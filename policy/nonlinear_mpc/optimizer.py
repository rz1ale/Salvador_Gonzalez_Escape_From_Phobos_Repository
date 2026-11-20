"""Numeric-only coupled 6DOF direct shooting. Never connects to the simulator.

The canonical model supplies allocation, rigid-body dynamics and observations.
Only integration mesh, finite-horizon objective and policy constraints live here.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from system_identification.model import Model, matrix, gravity
from obstacle_characterization import predict_objects


def settings(overrides=None):
    value = json.loads(Path(__file__).with_name('settings.json').read_text(encoding='utf-8-sig'))
    overrides = overrides or {}
    if not isinstance(overrides, dict) or set(overrides)-set(value):
        raise ValueError('Unknown NMPC settings override')
    value.update(copy.deepcopy(overrides))
    positive = ('horizon_s','stage_s','prediction_step_s','integration_step_s','update_period_s',
        'activation_lead_s','solve_wall_s','worker_timeout_wall_s','monitor_period_s',
        'guard_horizon_s','vehicle_radius_m','max_rate_rad_s','terminal_rate_rad_s','tail_acceleration_m_s2',
        'tail_fuel_multiplier','mission_timeout_s','plan_position_tolerance_m',
        'plan_attitude_tolerance_rad','plan_velocity_tolerance_m_s','plan_rate_tolerance_rad_s',
        'command_bound_scale','finite_difference_step','score_time_floor_s','recovery_timeout_s')
    if any(not isinstance(value[k], (int,float)) or not np.isfinite(value[k]) or value[k]<=0 for k in positive):
        raise ValueError('NMPC positive settings must be finite and >0')
    if value['schema_version']!=1 or value['formulation']!='direct_shooting':
        raise ValueError('Unsupported NMPC formulation')
    n=value['horizon_s']/value['stage_s']
    if abs(n-round(n))>1e-8 or not 2<=round(n)<=12:
        raise ValueError('Need 2..12 equal control stages')
    if not value['integration_step_s']<=value['prediction_step_s']<=value['stage_s']:
        raise ValueError('Require integration_step <= prediction_step <= stage')
    if value['activation_lead_s']+value['horizon_s']>9.5:
        raise ValueError('Lead + horizon must leave margin inside shared 10 s radar forecast')
    if value['solve_wall_s']>=value['activation_lead_s'] or value['worker_timeout_wall_s']<=value['solve_wall_s']:
        raise ValueError('Require solve budget < activation lead; worker timeout > solve budget')
    if not 0<value['command_bound_scale']<=1 or value['guard_horizon_s']>value['horizon_s']:
        raise ValueError('Invalid command bound or guard horizon')
    if value['terminal_rate_rad_s']>value['max_rate_rad_s']:
        raise ValueError('Terminal rate bound must not exceed path rate bound')
    nonnegative=('clearance_m','trajectory_chord_padding_m','launch_floor_tolerance_m',
        'rate_regularization','slew_regularization','arrival_margin_m','arrival_dwell_s')
    if any(not np.isfinite(value[k]) or value[k]<0 for k in nonnegative):
        raise ValueError('Invalid nonnegative NMPC setting')
    if type(value['max_iterations']) is not int or value['max_iterations']<1:
        raise ValueError('max_iterations must be a positive integer')
    return value


def score_win(elapsed_s, fuel_kg, damage_J, constants):
    """Exact supplied official winning formula; time origin is caller evidence."""
    ts=constants['NominalTime_s']/elapsed_s if elapsed_s!=0 else 0.
    return 50.+100.*(ts-2.*damage_J/constants['DamageThreshold_J']-
                    max(fuel_kg/constants['NominalFuel_kg'],0.))


def reported(model, state):
    x=np.asarray(state,float)
    B=matrix(x[6:10])
    return x[:3]+B.T@model.ell, x[3:6]+B.T@np.cross(x[10:13],model.ell)


def packet_row(packet):
    q=np.asarray(packet['q_PB_xyzw_reported'],float)
    return np.r_[packet['position_P_m'],packet['velocity_P_m_s'],q,
                 matrix(q)@np.asarray(packet['omega_relP_P_rad_s'])]


class Predictor:
    def __init__(self, model, integration_step_s):
        self.model=model
        self.step=float(integration_step_s)
        self.scale=np.zeros((2,6))
        for sign in (-1,1):
            for axis in range(6):
                ray=np.eye(6)[axis]*sign
                self.scale[int(sign>0),axis]=model.h/model.allocator.solve(ray).max()

    def decode(self,z):
        z=np.asarray(z,float)
        return z*self.scale[(z>0).astype(int),np.arange(6)]

    def encode(self,u):
        u=np.asarray(u,float)
        return u/self.scale[(u>0).astype(int),np.arange(6)]

    def allocation(self,u):
        b=self.model.allocator.clip(u)
        lam=self.model.allocator.solve(b)
        denom=max(self.model.h,float(np.max(lam)))
        base=b/denom
        # Exact structural fuel law, using the identified effective mean period.
        # This is a source-informed estimate, NOT an identified fuel model.
        fuel=float(self.model.allocator.forces@lam/denom/(80.*9.8))
        return base,fuel

    def rollout(self,x0,t0,times,command_times,commands,*,step=None):
        times=np.asarray(times,float); ct=np.asarray(command_times,float); u=np.asarray(commands,float)
        if times.ndim!=1 or not len(times) or times[0]<t0-1e-8 or np.any(np.diff(times)<0):
            raise ValueError('Invalid prediction times')
        if len(ct)==0: ct=np.array([t0-1.]);u=np.zeros((1,6))
        if u.shape!=(len(ct),6) or np.any(np.diff(ct)<0) or not np.isfinite(u).all():
            raise ValueError('Invalid command history')
        values=[self.allocation(a) for a in u]
        base=np.array([v[0] for v in values]);rates=np.array([v[1] for v in values])
        dl=self.model.p['linear_effect_delay_s']; da=self.model.p['angular_effect_delay_s']
        edges=np.unique(np.r_[ct,ct+dl,ct+da])
        x=np.asarray(x0,float).copy();clock=float(t0); fuel=0.;xs=[];fs=[]
        mesh=self.step if step is None else float(step)
        for end in times:
            while end-clock>1e-9:
                j=np.searchsorted(edges,clock+1e-8,side='right')
                edge=edges[j] if j<len(edges) else np.inf
                dt=min(mesh,end-clock,edge-clock); mid=clock+dt/2
                il=np.searchsorted(ct,mid-dl,side='right')-1
                ia=np.searchsorted(ct,mid-da,side='right')-1
                ir=np.searchsorted(ct,mid,side='right')-1
                b=np.r_[base[il,:3] if il>=0 else np.zeros(3),base[ia,3:] if ia>=0 else np.zeros(3)]
                x=self.model._advance(x,dt,b)
                fuel+=dt*(rates[ir] if ir>=0 else 0.)
                clock+=dt
            xs.append(x.copy());fs.append(fuel)
        return np.asarray(xs),np.asarray(fs)


def command_at(plan,when):
    if plan is None or when<float(plan['activation_sim_s'])-1e-8 or when>=float(plan['expires_sim_s'])-1e-8:
        return None
    i=np.searchsorted(plan['command_times_sim_s'],when+1e-8,side='right')-1
    return np.asarray(plan['commands'][max(0,i)],float)


def future_schedule(history,plan,now,until):
    """Keep actual request timestamps; append only committed, unexecuted changes."""
    # A snapshot can precede the latest completed send. Its actual timestamp is
    # known when this function runs and must remain in that snapshot's future.
    h=[(float(t),np.asarray(u,float)) for t,u in history if t<=until+1e-8]
    if not h: h=[(now-1.,np.zeros(6))]
    first=max(0,np.searchsorted([t for t,_ in h],now-.2,side='right')-1)
    h=h[first:]
    if plan is not None:
        for t,u in zip(plan['command_times_sim_s'],plan['commands']):
            if now+1e-8<t<=until+1e-8: h.append((float(t),np.asarray(u,float)))
        if now<plan['expires_sim_s']<until:
            raise ValueError('Committed plan expires before new activation')
    h.sort(key=lambda v:v[0])
    return [t for t,_ in h],[u.tolist() for _,u in h]


def recovery_request(predictor,x,target,mode='progress',bound=1.):
    """Constructive bounded seed/recovery only; not the primary NMPC controller.

    Six simultaneous requests always pass through the same shared allocator.
    No claim that this seed is optimal or collision-free.
    """
    m=predictor.model; B=matrix(x[6:10]);p,v=reported(m,x)
    delta=np.asarray(target)-p; direction=delta/max(np.linalg.norm(delta),1e-9)
    g=gravity(x[:3],m.p['phobos_mu_prior_m3_s2'],m.p['uniform_external_acceleration_P_m_s2'])
    if mode=='brake': desired=-v/3.-g
    elif mode=='coast': desired=-g
    else:
        cross_v=v-direction*(v@direction)
        desired=.075*direction-.3*cross_v-g
    desired_alpha=-x[10:13]/1.5-np.asarray(m.p['angular_bias_B_rad_s2'])
    raw=m.h*np.r_[np.linalg.solve(m.Kl,B@desired),np.linalg.solve(m.Ka,desired_alpha)]
    z=predictor.encode(raw)
    z*=min(1.,bound/max(np.max(np.abs(z)),1e-12))
    return predictor.decode(z)


def obstacle_arrays(frame,times,vehicle_radius,clearance,chord_padding):
    forecast=predict_objects(frame,times,array_output=True)
    if not forecast['objects']:return np.empty((0,len(times),3)),np.empty((0,len(times)))
    return (np.asarray([o['center_P_m'] for o in forecast['objects']]),
            np.asarray([o['inflated_obstacle_radius_m'] for o in forecast['objects']])+
            vehicle_radius+clearance+chord_padding)


def swept_clearances(positions,centers,radii):
    """All supplied objects; relative closest approach on every entire segment."""
    if not len(centers):return np.empty(0)
    r=positions[None,:-1,:]-centers[:,:-1,:]
    d=np.diff(positions,axis=0)[None,:,:]-np.diff(centers,axis=1)
    den=np.sum(d*d,axis=2)
    f=np.clip(np.divide(-np.sum(r*d,axis=2),den,out=np.zeros_like(den),where=den>1e-18),0,1)
    distance=np.linalg.norm(r+f[:,:,None]*d,axis=2)
    # Inflated radii are monotone bounds, so use the larger endpoint per segment.
    return (distance-np.maximum(radii[:,:-1],radii[:,1:])).ravel()


def tail_estimate(model,x,target,elapsed,fuel,damage,s,c):
    """Explicit obstacle-free 1D accelerate/coast continuation approximation.

    Enumerate future acceleration durations; no terminal stopping requirement.
    Current lateral velocity is charged an optimistic delta-v correction.
    This is an approximate value function, not global trajectory optimization.
    """
    p,v=reported(model,x); dvec=np.asarray(target)-p; distance=np.linalg.norm(dvec)
    remaining=max(distance-c['KeepOutRadius_m']+s['arrival_margin_m'],0.)
    if remaining<=0:return score_win(max(elapsed,s['score_time_floor_s']),fuel,damage,c),0.,0.
    direction=dvec/max(distance,1e-9);speed=float(v@direction)
    B=matrix(x[6:10])
    g=gravity(x[:3],model.p['phobos_mu_prior_m3_s2'],model.p['uniform_external_acceleration_P_m_s2'])
    # Homogeneous LP ray: delivered translation is direction/max(lambda).
    # Its total impulse price and directional plateau depend on TERMINAL q.
    forward_ray=np.r_[np.linalg.solve(model.Kl,B@direction),np.zeros(3)]
    forward_allocation=model.allocator.solve(forward_ray)
    axial_plateau=1./max(float(forward_allocation.max()),1e-9)
    a=max(.005,min(s['tail_acceleration_m_s2'],.9*axial_plateau)+float(g@direction))
    full=(np.sqrt(speed*speed+2*a*remaining)-speed)/a
    tau=np.linspace(0.,full,33)
    end_v=speed+a*tau
    d_acc=speed*tau+.5*a*tau*tau
    coast=np.maximum(remaining-d_acc,0)/np.maximum(end_v,1e-4)
    travel=tau+coast
    weight=model.allocator.forces/(80.*9.8)
    forward_price=float(weight@forward_allocation)
    hover_ray=np.r_[np.linalg.solve(model.Kl,B@(-g)),np.linalg.solve(model.Ka,-np.asarray(model.p['angular_bias_B_rad_s2']))]
    hover_fuel_rate=float(weight@model.allocator.solve(hover_ray))
    lateral=v-direction*speed
    correction_ray=np.r_[np.linalg.solve(model.Kl,B@(-lateral)),np.linalg.solve(model.Ka,-x[10:13])]
    correction_fuel=float(weight@model.allocator.solve(correction_ray))
    extra_fuel=s['tail_fuel_multiplier']*(forward_price*a*tau+hover_fuel_rate*travel+correction_fuel)
    scores=50+100*(c['NominalTime_s']/np.maximum(elapsed+travel,s['score_time_floor_s'])-
        2*damage/c['DamageThreshold_J']-(fuel+extra_fuel)/c['NominalFuel_kg'])
    i=int(np.argmax(scores))
    return float(scores[i]),float(travel[i]),float(extra_fuel[i])


class BudgetExpired(Exception):pass


def solve(predictor,job):
    s=job['settings']; c=job['score_constants'];start=time.monotonic()
    deadline=start+s['solve_wall_s'];m=predictor.model
    t0=float(job['source_sim_s']);activation=float(job['activation_sim_s'])
    prefix_t=job['committed_times'];prefix_u=job['committed_commands']
    xstart,fp= predictor.rollout(job['state'],t0,[activation],prefix_t,prefix_u)
    xstart=xstart[0];fuel0=job['fuel_kg']+float(fp[-1])
    n=round(s['horizon_s']/s['stage_s']);bound=s['command_bound_scale']
    times=activation+np.linspace(0.,s['horizon_s'],int(np.ceil(s['horizon_s']/s['prediction_step_s']))+1)
    ct=activation+np.arange(n)*s['stage_s']
    centers,radii=obstacle_arrays(job['obstacles'],times,s['vehicle_radius_m'],s['clearance_m'],s['trajectory_chord_padding_m'])
    prefix=[(t,u) for t,u in zip(prefix_t,prefix_u) if t<activation-1e-8]
    if not prefix:prefix=[(t0-1.,np.zeros(6).tolist())]
    cache={};best=None;best_infeasible=None;evals=0
    up=np.asarray(job['launch_up_P']);floor=np.asarray(job['launch_position_P'])

    def evaluate(z,enforce_budget=True):
        nonlocal best,best_infeasible,evals
        if enforce_budget and time.monotonic()>=deadline:raise BudgetExpired()
        z=np.asarray(z,float).reshape(n,6);key=z.tobytes()
        if key in cache:return cache[key]
        controls=predictor.decode(z)
        all_t=[t for t,_ in prefix]+ct.tolist();all_u=[u for _,u in prefix]+controls.tolist()
        xs,fs=predictor.rollout(xstart,activation,times,all_t,all_u)
        positions=np.asarray([reported(m,x)[0] for x in xs])
        distances=np.linalg.norm(positions-np.asarray(job['target']),axis=1)
        arrival=np.flatnonzero(distances<=c['KeepOutRadius_m']-s['arrival_margin_m'])
        if len(arrival):
            i=int(arrival[0]);val=score_win(max(times[i],s['score_time_floor_s']),fuel0+fs[i],job['damage_J'],c)
            tail_time=tail_fuel=0.
        else:
            val,tail_time,tail_fuel=tail_estimate(m,xs[-1],job['target'],times[-1],fuel0+fs[-1],job['damage_J'],s,c)
        regularizer=s['rate_regularization']*np.sum(xs[:,10:13]**2)/len(xs)
        regularizer+=s['slew_regularization']*np.sum(np.diff(z,axis=0)**2)
        clearance=swept_clearances(positions,centers,radii)
        angular=s['max_rate_rad_s']-np.linalg.norm(xs[:,10:13],axis=1)
        # Tail holds terminal attitude. Do not let a spinning terminal state
        # obtain its benefits without planning the angular stopping maneuver.
        terminal_angular=np.array([s['terminal_rate_rad_s']-np.linalg.norm(xs[-1,10:13])])
        # A local launch-plane guard, not a claimed Phobos terrain model.
        altitude=(positions-floor)@up+s['launch_floor_tolerance_m']
        constraint=np.r_[clearance,angular,terminal_angular,altitude]
        violation=float(max(0.,-np.min(constraint)))
        result=dict(z=z.copy(),commands=controls,states=xs,positions=positions,
            fuel=fs,objective=float(-val+regularizer),score_estimate=float(val),
            tail_time_s=tail_time,tail_fuel_kg=tail_fuel,constraint=constraint,
            violation=violation,minimum_clearance_m=float(clearance.min()) if len(clearance) else None)
        cache[key]=result;evals+=1
        if violation<=1e-6 and np.isfinite(result['objective']):
            if best is None or result['objective']<best['objective']:best=result
        elif best_infeasible is None or violation<best_infeasible['violation']:best_infeasible=result
        return result

    seeds=[]
    previous=job.get('warm_plan')
    def seed(mode,previous=None):
        temp=xstart.copy();controls=[];st=[t for t,_ in prefix];su=[u for _,u in prefix]
        for t in ct:
            u=command_at(previous,t) if previous else None
            if u is None:u=recovery_request(predictor,temp,job['target'],mode,bound)
            z=np.clip(predictor.encode(u),-bound,bound);u=predictor.decode(z)
            controls.append(z);st.append(float(t));su.append(u.tolist())
            temp=predictor.rollout(temp,t,[t+s['stage_s']],st,su)[0][0]
        return np.asarray(controls)
    if previous:seeds.append(seed('coast',previous))
    for mode in ('progress','coast','brake'):seeds.append(seed(mode))
    message='budget_expired';success=False;iterations=0
    def iteration(_):
        nonlocal iterations
        iterations+=1
    try:
        for z in seeds:evaluate(z)
        initial=best if best is not None else best_infeasible
        fit=minimize(lambda z:evaluate(z)['objective'],initial['z'].ravel(),method='SLSQP',
            bounds=[(-bound,bound)]*(6*n),
            constraints=[dict(type='ineq',fun=lambda z:evaluate(z)['constraint'])],callback=iteration,
            options=dict(maxiter=s['max_iterations'],ftol=.01,eps=s['finite_difference_step']))
        evaluate(fit.x);success=bool(fit.success);iterations=int(fit.nit);message=str(fit.message)
    except BudgetExpired:pass
    elapsed=time.monotonic()-start
    result=dict(job_id=job['job_id'],generation=job['generation'],source_sim_s=t0,
        activation_sim_s=activation,expires_sim_s=activation+s['horizon_s'],
        status='feasible' if best is not None else 'no_feasible_plan',
        solve_wall_s=elapsed,evaluations=evals,converged=success,iterations=iterations,
        solver_message=message,official_score_captured=False,
        objective_kind='official_score_with_approximate_accelerate_coast_tail')
    if best is not None:
        result.update(command_times_sim_s=ct.tolist(),commands=best['commands'].tolist(),
            prediction_times_sim_s=times.tolist(),predicted_states=best['states'].tolist(),
            predicted_positions_P_m=best['positions'].tolist(),predicted_fuel_kg=best['fuel'].tolist(),
            score_estimate=best['score_estimate'],tail_time_s=best['tail_time_s'],tail_fuel_kg=best['tail_fuel_kg'],
            minimum_inflated_clearance_m=best['minimum_clearance_m'])
    elif best_infeasible is not None:result['constraint_violation']=best_infeasible['violation']
    return result
