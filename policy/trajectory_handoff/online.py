"""Event-triggered LTV MPC and complete nonlinear/statistical post-screening.

Translation-only corrections are a selectable hypothesis, not independent
Cartesian thrust limits. Each rollout uses the shared six-axis allocator.
Obstacle exteriors are nonconvex: several local corridor sides are attempted.
An unsuccessful QP is not evidence that the physical problem is infeasible.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from obstacle_characterization.collision import (check_trajectory, predict_objects,
    position_support_sigma, ellipsoid_separating_normal)
from .reference import error, retract, interpolate_state, clip_request, schedule_control, plan_end
from .core import score


class Deadline(Exception):pass


def check_time(deadline):
    if time.monotonic()>=deadline:raise Deadline()


def qp(H,q,C,lo,hi,*,iterations=200,tolerance=.005,deadline=float('inf'),warm_start=None):
    """Scaled dense ADMM QP, l<=Cz<=u. Returns residuals, not a certificate."""
    H=(np.asarray(H)+np.asarray(H).T)/2; q=np.asarray(q); C=np.asarray(C)
    lo=np.asarray(lo);hi=np.asarray(hi)
    if np.any(lo>hi):return None,dict(status='contradictory_bounds',iterations=0)
    zero=np.linalg.norm(C,axis=1)<1e-10
    if np.any(zero & ((lo>tolerance)|(hi < -tolerance))):return None,dict(status='fixed_state_outside_corridor',iterations=0)
    C=C[~zero];lo=lo[~zero];hi=hi[~zero]
    if not len(C):C=np.zeros((1,len(q)));lo=np.array([-np.inf]);hi=np.array([np.inf])
    row=np.maximum(np.linalg.norm(C,axis=1),1e-6);C=C/row[:,None];lo=lo/row;hi=hi/row
    obj=max(float(np.linalg.norm(H,ord=np.inf)),float(np.max(abs(q))),1e-3)
    H=H/obj;q=q/obj
    rho=2.;sigma=1e-7
    factor=cho_factor(H+rho*C.T@C+sigma*np.eye(len(q)),lower=True,check_finite=False)
    x=np.asarray(warm_start,float).copy() if warm_start is not None and np.shape(warm_start)==np.shape(q) else np.zeros(len(q))
    z=np.clip(C@x,lo,hi);dual=np.zeros(len(z));status='iteration_limit'
    primal=dual_res=float('inf')
    for it in range(int(iterations)):
        if it%8==0:check_time(deadline)
        x=cho_solve(factor,rho*C.T@(z-dual)-q+sigma*x,check_finite=False)
        cx=C@x;previous=z.copy();z=np.clip(cx+dual,lo,hi);dual+=cx-z
        primal=float(np.max(abs(cx-z)));dual_res=float(np.max(abs(rho*C.T@(z-previous))))
        if primal<=tolerance and dual_res<=tolerance:status='solved';break
    physical=float(max(np.max(np.maximum(lo-C@x,0)),np.max(np.maximum(C@x-hi,0))))
    valid=np.isfinite(x).all() and physical<=tolerance
    return (x if valid else None),dict(status=status,iterations=it+1,primal_residual=primal,
        dual_residual=dual_res,normalized_bound_violation=physical,feasible_iterate=bool(valid))


def law(reference,phase,active,state,when,s):
    return schedule_control(active,state,when,reference,phase,s)


def forecast(plant,reference,x,start,end,phase,active,old,s,step=.25,deadline=None):
    ts=np.unique(np.r_[start,np.arange(start+step,end,step),end]);xs=[x.copy()];us=[];prev=np.asarray(old)
    for a,b in zip(ts[:-1],ts[1:]):
        if deadline is not None:check_time(deadline)
        u=law(reference,phase+a-start,active,xs[-1],a,s)
        xs.append(plant.transition(xs[-1],prev,u,b-a));us.append(u);prev=u
    return ts,np.asarray(xs),np.asarray(us),prev


def readiness(frame,now,s):
    if frame is None:return False,'radar_not_initialized'
    if frame['generated_sim_s']>now+1e-8:return False,'radar_frame_from_future'
    if now-frame['generated_sim_s']>s['max_radar_frame_age_s']:return False,'radar_frame_stale'
    if not frame.get('objects'):return False,'radar_no_usable_objects'
    diagnostics=frame.get('diagnostics',{})
    # Duplicate packets can retain live tracks, but not indefinitely. The shared
    # tracker carries coasting/stale/ambiguous hypotheses into every query.
    if any(c in frame['coverage'].get('conditions',[]) for c in
           ('empty_radar_packet','no_assessed_objects','clock_unregistered','unresolved_capacity_loss',
            'retained_occupancy_capacity_loss','current_own_pose_unavailable')):
        return False,'radar_coverage_unavailable'
    return True,None


def assess(plant,frame,ts,xs,s,launch,up):
    # Freshness is checked at the measurement/source epoch in solve(). A future
    # prediction is not a new sensor read; its uncertainty is propagated by the query.
    ok,reason=readiness(frame,float(frame['generated_sim_s']) if frame is not None else float(ts[0]),s)
    if not ok:return dict(admissible=False,reason=reason,certified_safe=False)
    pp=plant.reported_positions(xs)
    query=check_trajectory(frame,ts,pp,vehicle_radius_m=s['vehicle_radius_m'],
        clearance_m=s['clearance_m']+s['chord_padding_m']+s['plan_position_tolerance_m'],
        uncertainty_geometry=s.get('obstacle_uncertainty_geometry','sphere'))
    floor=float(np.min((pp-launch)@up+s['launch_floor_tolerance_m']))
    rates=float(np.max(np.linalg.norm(xs[:,10:13],axis=1)))
    ok=query['assessment']=='no_predicted_intersection' and floor>=0 and rates<=s['max_rate_rad_s']
    return dict(admissible=bool(ok),minimum_inflated_clearance_m=query['minimum_inflated_clearance_m'],
        critical_object_id=query.get('critical_object_id'),critical_object=query['objects'][0] if query.get('objects') else None,
        collision_assessment=query['assessment'],uncertainty_geometry=query.get('uncertainty_geometry'),launch_plane_margin_m=floor,
        max_rate_rad_s=rates,objects_assessed=len(frame['objects']),conditions=query['conditions'],
        retained_expired_tracks=frame['coverage'].get('retained_expired_tracks',0),
        retained_expired_unresolved_observations=frame['coverage'].get('retained_expired_unresolved_observations',0),
        reason=(None if ok else 'radar_coverage_incomplete' if query['assessment']=='unassessed' else 'forecast_or_policy_bound'),certified_safe=False)


def validate_plan(plant,reference,plan,x,old,s,frame,launch,up,deadline):
    # Simulate the actual exported FF+FB law, not merely the optimizer knot line.
    start=float(plan['times'][0]);end=plan_end(plan);phase=plan['phase']
    ts,xs,_,_=forecast(plant,reference,x,start,end,phase,plan,old,s,step=.25,deadline=deadline)
    check_time(deadline)
    return assess(plant,frame,ts,xs,s,launch,up),ts,xs


def make_plan(reference,plant,x,when,phase,us,old,s):
    ts=when+np.arange(len(us)+1)*s['mpc_stage_s'];xs=plant.rollout(x,ts,us,old)
    gains=np.array([reference.at(phase+t-when)[2] for t in ts[:-1]])
    return dict(times=ts.tolist(),states=xs.tolist(),commands=np.asarray(us).tolist(),gains=gains.tolist(),phase=float(phase))


def tail_estimate(plant,reference,plan,fuel,damage,constants):
    xs=np.asarray(plan['states']);ts=np.asarray(plan['times']);phase=plan['phase']+ts[-1]-ts[0]
    p=reference.closest_phase(xs[-1],phase,window=20.)
    # Tangent continuation from the cached whole-mission solution; never assume
    # a made-up scalar tail thrust/fuel efficiency. Large departures remain local approximations.
    remaining=max(0.,reference.duration-p)
    used=sum(plant.fuel_rate(u)*dt for u,dt in zip(plan['commands'],np.diff(ts)))
    arrival=max(ts[-1]+remaining,1.)
    return score(arrival,fuel+used+reference.remaining(p),damage,constants)


def feasibility_seed(plant,reference,job,frame,x,old,activation,phase,s,deadline,nominal_clear):
    """Find a nonlinear-verified incumbent BEFORE spending the QP budget.

    This changes the search initialization, not the accepted collision set or
    public actuation law. Every seed is a complete coupled 13-state FF+FB plan.
    """
    reports=[]
    scales=[1.] if nominal_clear else s.get('feasible_start_translation_scales',[0.,-.5])
    for factor in scales:
        check_time(deadline);start=time.monotonic()
        us=np.array([reference.at(phase+k*s['mpc_stage_s'])[1] for k in range(s['mpc_stages'])])
        us[:,:3]*=float(factor);us=np.array([plant.canonical_request(u) for u in us])
        plan=make_plan(reference,plant,x,activation,phase,us,old,s)
        check,_,_=validate_plan(plant,reference,plan,x,old,s,frame,np.asarray(job['launch_position_P']),np.asarray(job['launch_up_P']),deadline)
        reports.append(dict(translation_scale=float(factor),assessment=check,wall_s=time.monotonic()-start))
        if check['admissible']:
            plan['source']='verified_feasible_start'
            value=tail_estimate(plant,reference,plan,job['fuel_kg'],job['damage_J'],job['score_constants'])
            return (value,plan,check),reports
    return None,reports


def planning_grid(reference,phase,activation,s,arrival_scale=1.):
    near=s['mpc_stages']*s['mpc_stage_s']
    if not s.get('tail_stages',0):return activation+np.arange(s['mpc_stages']+1)*s['mpc_stage_s']
    remaining=max(near+4.,(reference.duration-phase)*arrival_scale)
    advisory=min(remaining, s['advisory_horizon_s'])
    knots=list(np.arange(s['mpc_stages']+1)*s['mpc_stage_s'])
    # Fine near field, two advisory intervals, remaining coarse mission tail.
    if remaining>advisory+.01:
        first=min(2,s['tail_stages']-1)
        knots.extend(np.linspace(near,advisory,first+1)[1:])
        knots.extend(np.linspace(advisory,remaining,s['tail_stages']-first+1)[1:])
    else:knots.extend(np.linspace(near,remaining,s['tail_stages']+1)[1:])
    return activation+np.asarray(knots)


def goal_plan(reference,plant,x,ts,phase,us,old,s,source):
    xs=plant.rollout(x,ts,us,old)
    gains=np.array([reference.at(phase+t-ts[0])[2] for t in ts[:-1]])
    return dict(times=ts.tolist(),states=xs.tolist(),commands=np.asarray(us).tolist(),gains=gains.tolist(),
        phase=float(phase),source=source,kind='mission_mpc',
        execute_until_sim_s=float(min(ts[-1],ts[0]+s['mpc_stages']*s['mpc_stage_s'])),
        arrival_sim_s=float(ts[-1]),tail_prediction='coarse nonlinear shooting with frozen-affine sensitivities',
        full_tail_clearance_verified=False)


def mpc(plant,reference,job,frame,x,old,activation,phase,s,deadline,incumbent=None):
    best=None;reports=[]
    warm=getattr(plant,'planning_warm',{});plant.planning_warm=warm
    for scale in s.get('arrival_time_scales',[1.]):
        try:
            check_time(deadline);ts=planning_grid(reference,phase,activation,s,float(scale));hs=np.diff(ts);N=len(hs)
            axes=np.arange(3 if s['correction_axes']=='translation' else 6);d=len(axes);nz=N*d
            us=[];xs=[x.copy()];prev=old;prior=job.get('active_plan')
            warp=max(reference.duration-phase,1.)/max(ts[-1]-activation,1.)
            for k,dt in enumerate(hs):
                check_time(deadline)
                ph=phase+(ts[k]-activation)*warp
                if prior is not None and prior.get('kind') not in ('coast','local_avoidance') and ts[k]<plan_end(prior):
                    raw=law(reference,ph,prior,xs[-1],ts[k],s)
                else:raw=reference.at(ph)[1]
                raw=plant.canonical_request(raw);us.append(raw);xs.append(plant.transition(xs[-1],prev,raw,dt));prev=raw
            us=np.asarray(us);xs=np.asarray(xs)
            S=[np.zeros((12,nz))];previous_selector=np.zeros((6,nz));prev=old
            for k,dt in enumerate(hs):
                check_time(deadline)
                if s.get('affine_planning',True):A,B,D=plant.affine_linearize(xs[k],prev,us[k],dt)
                else:_,A,B,D=plant.linearize(xs[k],prev,us[k],dt)
                selector=np.zeros((6,nz));selector[axes,k*d+np.arange(d)]=1.
                S.append(A@S[-1]+B@selector+D@previous_selector);previous_selector=selector;prev=us[k]
            xref,_,_,lam,P=reference.at(phase+ts[-1]-activation)
            P=s['mpc_curvature_weight']*P
            H=S[-1].T@P@S[-1]+s['mpc_input_regularization']*np.eye(nz)
            q=S[-1].T@(lam+P@error(xs[-1],xref))
            for k,u in enumerate(us):
                for a,axis in enumerate(axes):
                    perturb=np.eye(6)[axis]*plant.scales[axis]*.004
                    gradient=(plant.fuel_rate(u+perturb)-plant.fuel_rate(u-perturb))/.008
                    q[k*d+a]+=100./job['score_constants']['NominalFuel_kg']*hs[k]*gradient
            if N>1:
                L=np.kron(np.diff(np.eye(N),axis=0),np.eye(d));H+=s['mpc_slew_regularization']*L.T@L
            positions=plant.reported_positions(xs);maps=[]
            for k,state in enumerate(xs):
                M=np.zeros((3,12));M[:,:3]=np.eye(3)
                for j in range(3):
                    delta=np.eye(12)[6+j]*1e-4
                    pp=plant.reported_positions([retract(state,delta),retract(state,-delta)])
                    M[:,6+j]=(pp[0]-pp[1])/2e-4
                maps.append(M@S[k])
            maps=np.asarray(maps);near=min(s['mpc_stages'],N)
            far=np.flatnonzero((ts>ts[near]+1e-8)&(ts<=job['source_sim_s']+s['advisory_horizon_s']))
            if len(far):
                advisory_objects=predict_objects(frame,ts[:far[-1]+1],array_output=True)['objects']
                for node in far:
                    for ob in advisory_objects:
                        relative=positions[node]-np.asarray(ob['center_P_m'][node]);cov=np.asarray(ob['position_covariance_P'][node])
                        normal=ellipsoid_separating_normal(relative,cov,frame['default_query']['uncertainty_multiplier'])
                        required=ob['extent_radius_with_margins_m']+s['vehicle_radius_m']+s['clearance_m']+s['chord_padding_m']+s['plan_position_tolerance_m']
                        required+=frame['default_query']['uncertainty_multiplier']*float(position_support_sigma(cov,normal))
                        overlap=max(0.,required-float(normal@relative))
                        if overlap:q-=s['advisory_weight']*overlap*(normal@maps[node])
            # Only the near horizon is a hard obstacle set. Far hypotheses contribute
            # advisory cost; they never certify a coarse tail as free space.
            objects=predict_objects(frame,ts[:near+1],array_output=True)['objects'];count=len(objects)
            centers=np.asarray([o['center_P_m'] for o in objects]).reshape(count,near+1,3)
            rel=positions[None,:near+1]-centers;mid=.5*(rel[:,:-1]+rel[:,1:]);length=np.linalg.norm(mid,axis=2)
            extra=s['vehicle_radius_m']+s['clearance_m']+s['chord_padding_m']+s['plan_position_tolerance_m']+s['linearization_margin_m']
            covariance=np.asarray([o['position_covariance_P'] for o in objects]).reshape(count,near+1,3,3)
            extent=np.array([o['extent_radius_with_margins_m'] for o in objects])[:,None]+extra
            sigma=frame['default_query']['uncertainty_multiplier']
            n0=ellipsoid_separating_normal(mid,covariance[:,1:],sigma)
            radii=extent+sigma*np.maximum(position_support_sigma(covariance[:,:-1],n0),position_support_sigma(covariance[:,1:],n0))
            risk=np.min(length-radii,axis=1) if count else np.empty(0)
            ids=np.argsort(risk)[:s.get('planner_obstacles',12)]
            forward=positions[near]-positions[0];forward/=max(np.linalg.norm(forward),1e-9)
            side=np.cross(forward,np.asarray(job['launch_up_P']));side/=max(np.linalg.norm(side),1e-9)
            if np.linalg.norm(side)<.5:side=np.array([1.,0.,0.])
            vertical=np.cross(side,forward);vertical/=max(np.linalg.norm(vertical),1e-9)
            preferred=int(job.get('preferred_side',1) or 1)
            directions=[np.zeros(3),preferred*side,-preferred*side,vertical,-vertical][:s['mpc_candidate_count']]
            for branch,direction in enumerate(directions):
                if time.monotonic()>=deadline:break
                rows=[np.eye(nz)];lower=[np.full(nz,-s['request_trust_radius'])];upper=[np.full(nz,s['request_trust_radius'])]
                if count:
                    rr=rel[ids];cv=covariance[ids];ex=extent[ids];rad=radii[ids];ll=length[ids]
                    n=n0[ids]*ll[:,:,None]+2*direction[None,None,:]*rad[:,:,None]
                    nn=np.linalg.norm(n,axis=2);n=np.divide(n,nn[:,:,None],out=np.zeros_like(n),where=nn[:,:,None]>1e-12);n[nn<1e-12]=side
                    required=ex+sigma*np.maximum(position_support_sigma(cv[:,:-1],n),position_support_sigma(cv[:,1:],n))
                    for node,mapping in ((rr[:,:-1],maps[:near]),(rr[:,1:],maps[1:near+1])):
                        rows.append(np.einsum('mki,kij->mkj',n,mapping).reshape(len(ids)*near,nz))
                        lower.append((required-np.sum(n*node,axis=2)).ravel());upper.append(np.full(len(ids)*near,np.inf))
                up=np.asarray(job['launch_up_P'])
                rows.append(np.einsum('i,kij->kj',up,maps[1:near+1]));lower.append(-s['launch_floor_tolerance_m']-(positions[1:near+1]-job['launch_position_P'])@up);upper.append(np.full(near,np.inf))
                if s.get('tail_stages',0):
                    delta=positions[-1]-np.asarray(reference.manifest['target_P_m']);bound=s['goal_tolerance_m']/np.sqrt(3.)
                    rows.append(maps[-1]);lower.append(-bound-delta);upper.append(bound-delta)
                z,report=qp(H,q,np.vstack(rows),np.concatenate(lower),np.concatenate(upper),iterations=s['qp_iterations'],
                    tolerance=s['qp_tolerance'],deadline=deadline,warm_start=warm.get((N,branch)))
                report.update(arrival_scale=float(scale),arrival_sim_s=float(ts[-1]),stages=N,near_stages=near,
                    objects_considered=len(ids),objects_postscreened=count,branch=branch)
                reports.append(report)
                if z is None:continue
                warm[(N,branch)]=z.copy();candidate=us.copy();candidate[:,axes]+=z.reshape(N,d)*plant.scales[axes]
                candidate=np.array([plant.canonical_request(clip_request(u,reference.caps)) for u in candidate])
                check_time(deadline);plan=goal_plan(reference,plant,x,ts,phase,candidate,old,s,'linear_mpc')
                check,_,_=validate_plan(plant,reference,plan,x,old,s,frame,np.asarray(job['launch_position_P']),up,deadline)
                if not check['admissible']:report['nonlinear_rejected']=check;continue
                used=sum(plant.fuel_rate(u)*dt for u,dt in zip(candidate,hs))
                value=score(float(ts[-1]),job['fuel_kg']+used,job['damage_J'],job['score_constants'])
                if branch==2:value-=s.get('side_switch_cost',.5)
                plan.update(estimated_fuel_kg=used,preferred_side=preferred if branch!=2 else -preferred)
                plan['affine_terminal_position_P_m']=(positions[-1]+maps[-1]@z).tolist()
                plan['affine_terminal_distance_m']=float(np.linalg.norm(positions[-1]+maps[-1]@z-reference.manifest['target_P_m']))
                if best is None or value>best[0]:best=(value,plan,check)
        except Deadline:
            reports.append(dict(status='numeric_budget_exhausted',arrival_scale=float(scale)))
            break
    return best or incumbent,reports


def solve(plant,reference,job,frame,s,*,on_current=None):
    budget=float(job.get('solve_wall_s',s['solve_wall_s']))
    if not np.isfinite(budget) or not 0<budget<=s['solve_wall_s']:raise ValueError('Invalid solve budget')
    start=time.monotonic();deadline=start+budget;now=float(job['source_sim_s']);phase=float(job['phase'])
    active=job.get('active_plan');x=np.asarray(job['state']);old=np.asarray(job['last_command'])
    activation=max(now+s['activation_lead_s'],float(job.get('committed_until_sim_s',now)))
    launch=np.asarray(job['launch_position_P']);up=np.asarray(job['launch_up_P'])
    result=dict(job_id=job['job_id'],generation=job['generation'],source_sim_s=now,activation_sim_s=activation,
        status='blocked',certified_safe=False,solve_budget_wall_s=budget,worker_role='planner')
    ok,reason=readiness(frame,now,s)
    if not ok:return dict(result,status='data_unavailable',reason=reason,wall_s=time.monotonic()-start)
    try:
        pt,px,_,prev=forecast(plant,reference,x,now,activation,phase,active,old,s,deadline=deadline)
        prefix=assess(plant,frame,pt,px,s,launch,up)
        result.update(prefix_times=pt.tolist(),prefix_states=px.tolist(),prefix_assessment=prefix,
            valid_until_sim_s=activation+.4,radar=dict(sequence=frame['sequence'],generated_sim_s=frame['generated_sim_s'],objects=len(frame['objects']),coverage=frame['coverage']))
        if not prefix['admissible']:
            missing=prefix.get('collision_assessment')=='unassessed'
            return dict(result,status='data_unavailable' if missing else 'blocked',
                reason='radar_coverage_incomplete' if missing else 'committed_prefix_conflict',wall_s=time.monotonic()-start)
        horizon=s['mpc_stages']*s['mpc_stage_s'];end=activation+horizon
        nt,nx,_,_=forecast(plant,reference,px[-1],activation,end,phase+activation-now,active,prev,s,deadline=deadline)
        nominal=assess(plant,frame,nt,nx,s,launch,up);result['continued_assessment']=nominal
        at,ax,_,_=forecast(plant,reference,px[-1],activation,now+s['advisory_horizon_s'],phase+activation-now,active,prev,s,step=.75,deadline=deadline)
        advisory=assess(plant,frame,at,ax,s,launch,up);result['advisory_assessment']=advisory
        if active is None and nominal['admissible'] and advisory['admissible'] and not job.get('force_mpc',False):
            return dict(result,status='nominal_clear',wall_s=time.monotonic()-start)
        if active is not None and not job.get('force_mpc',False):
            rejoin_phase=reference.closest_phase(px[-1],phase+activation-now,s['phase_search_window_s'])
            e=error(px[-1],reference.at(rejoin_phase)[0])
            if np.linalg.norm(e[:3])<=s['rejoin_position_m'] and np.linalg.norm(e[3:6])<=s['rejoin_velocity_m_s'] and np.linalg.norm(e[6:9])<=s['rejoin_attitude_rad']:
                rt,rx,_,_=forecast(plant,reference,px[-1],activation,end,rejoin_phase,None,prev,s,deadline=deadline)
                rc=assess(plant,frame,rt,rx,s,launch,up)
                if rc['admissible']:return dict(result,status='rejoin_clear',rejoin_phase=rejoin_phase,rejoin_assessment=rc,wall_s=time.monotonic()-start)
        plan_phase=reference.closest_phase(px[-1],phase+activation-now,s['phase_search_window_s']) if active else phase+activation-now
        incumbent,reports=feasibility_seed(plant,reference,job,frame,px[-1],prev,activation,plan_phase,s,deadline,nominal['admissible'])
        result['feasible_start_checks']=reports
        try:best,reports=mpc(plant,reference,job,frame,px[-1],prev,activation,plan_phase,s,deadline,incumbent)
        except Deadline:best=incumbent;reports=[];result['optimization_budget_exhausted']=True
        result['qp']=reports
        if best is not None:
            value,plan,check=best
            return dict(result,status='mpc_feasible',plan=plan,plan_assessment=check,estimated_continuation_score=value,
                selected_candidate=plan.get('source','linear_mpc'),wall_s=time.monotonic()-start)
        result['reason']='no_feasible_corridor_candidate'
    except Deadline:result['reason']='numeric_budget_exhausted'
    return dict(result,wall_s=time.monotonic()-start)
