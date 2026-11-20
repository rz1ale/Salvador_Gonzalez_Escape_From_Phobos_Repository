"""Numerical helpers; offline-safe and no simulator SDK access.

State: P-frame position/velocity, passive P->B xyzw attitude, P-frame omega.
Actuator: public 6-axis body acceleration REQUESTS; not achieved acceleration.
"""
from __future__ import annotations
import itertools
import math
import numpy as np


def unit(x, fallback=None):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n < 1e-10:
        if fallback is None:
            raise ValueError('Cannot normalize vector')
        return unit(fallback)
    return x / n


def cap(x, magnitude):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x * min(1., float(magnitude) / n) if n > 0 else x.copy()


def axes_for_direction(x):
    """Columns of passive-B inverse: body axes expressed in P coordinates."""
    x = unit(x)
    ref = np.array([0., 0., 1.]) if abs(x[2]) < .85 else np.array([0., 1., 0.])
    y = unit(np.cross(ref, x))
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def estimate_target_velocity(history, max_speed):
    """Use only previously observed public target positions and read times."""
    if len(history) < 2:
        return np.zeros(3)
    times = np.asarray([i[0] for i in history], float)
    ps = np.asarray([i[1] for i in history], float)
    delta = times[-1]-times[0]
    if delta < .3:
        return np.zeros(3)
    X = times-times.mean()
    v = (X[:, None] * (ps-ps.mean(axis=0))).sum(axis=0) / max(float(X @ X),1e-12)
    return cap(v,max_speed)


class TargetMotion:
    def __init__(self, position, velocity, mode='linear'):
        self.p = np.asarray(position,float).copy()
        self.v = np.asarray(velocity,float).copy()
        self.mode = mode
        if mode not in ('fixed','linear','angular'):
            raise ValueError('target.mode must be fixed, linear, or angular')
        self.w = np.cross(self.p,self.v) / max(float(self.p@self.p),1e-9)

    def at(self, dt):
        t = np.asarray(dt,float)
        if self.mode == 'fixed':
            return np.broadcast_to(self.p, t.shape+(3,)).copy()
        if self.mode == 'linear':
            return self.p+t[...,None]*self.v
        omega = np.linalg.norm(self.w)
        if omega < 1e-12:
            return self.p+t[...,None]*self.v
        axis=self.w/omega
        angle=omega*t
        # Uniform angular-motion hypothesis; inferred solely from measured public target positions.
        return (np.cos(angle)[...,None]*self.p +
                np.sin(angle)[...,None]*np.cross(axis,self.p) +
                (1-np.cos(angle))[...,None]*(axis@self.p)*axis +
                t[...,None]*axis*(axis@self.v))


def latent_from_reported(state, model):
    from system_identification.model import matrix
    q=np.asarray(state['q_PB_xyzw_reported'],float)
    C=matrix(q)  # P->B
    p=np.asarray(state['position_P_m'],float)-C.T@model.ell
    v=np.asarray(state['velocity_P_m_s'],float)-model.vbias
    w=C@np.asarray(state['omega_relP_P_rad_s'],float)-model.wbias
    return p,v,q,w,C


def fuel_per_second(model, request):
    """One-tick nominal fuel law from native minimum-sum LP + normalized on-times.
    Timing uses the model's *fitted* effective period, not verified simulator clock.
    """
    u=model.allocator.clip(np.asarray(request,float))
    if np.linalg.norm(u)<1e-12:
        return 0.
    allocation=model.allocator.solve(u)
    L=float(np.max(allocation))
    if L<=1e-12:
        return 0.
    h=model.h
    on=np.maximum(allocation,0.)*(min(h,L)/L)
    per_tick=float(model.allocator.forces @ on)/(80.*9.8)
    return per_tick/h


def estimated_bangbang_turn(current_x, target_x, cfg):
    phi=math.acos(float(np.clip(unit(current_x)@unit(target_x),-1,1)))
    a=float(cfg['attitude']['estimated_effective_ang_accel_rad_s2'])
    return 2.*math.sqrt(phi/a) if phi>0 else 0., phi


def model_gravity_batch(p, model):
    radius=np.linalg.norm(p,axis=1)
    if np.any(radius < 1.):
        raise ValueError('Invalid position relative to Phobos center')
    return (-float(model.p['phobos_mu_prior_m3_s2'])*p/radius[:,None]**3 +
            np.asarray(model.p['uniform_external_acceleration_P_m_s2']))


def basis_around(x):
    M=axes_for_direction(x)
    return M[:,0],M[:,1],M[:,2]


def run_candidates(state, model, target, cfg, candidates, base, start_sim_s,
                   force_direction=None, known_aligned=False):
    """Vectorized nominal fixed-attitude +X burn followed by gravity coast.

    candidates: N x 3 = yaw_deg, pitch_deg, burn_s; when force_direction is set,
    the first two dimensions are ignored. No radar/rotation dynamics here.
    """
    cand=np.asarray(candidates,float)
    if cand.ndim!=2 or cand.shape[1]!=3 or len(cand)==0:
        raise ValueError('Expected Nx3 candidate table')
    n=len(cand)
    p0,v0,q,w,C=latent_from_reported(state,model)
    x_now=C.T@np.array([1.,0.,0.])
    if force_direction is None:
        _,ey,ez=basis_around(base)
        offsets=np.tan(np.deg2rad(cand[:,:2]))
        axes=base[None,:]+offsets[:,0,None]*ey+offsets[:,1,None]*ez
        directions=axes/np.linalg.norm(axes,axis=1)[:,None]
    else:
        directions=np.broadcast_to(unit(force_direction),(n,3)).copy()
    turns=np.arccos(np.clip(directions@x_now,-1,1))
    estimated_align=np.zeros(n) if known_aligned else 2*np.sqrt(turns/cfg['attitude']['estimated_effective_ang_accel_rad_s2'])
    max_align=cfg['attitude']['max_align_s']
    estimated_align=np.minimum(estimated_align,max_align)

    burns=cand[:,2].clip(0.,cfg['planning']['burn_max_s'])
    mu=float(model.p['phobos_mu_prior_m3_s2'])
    bias=np.asarray(model.p['uniform_external_acceleration_P_m_s2'],float)
    g0=(-mu*p0/np.linalg.norm(p0)**3+bias)
    p=p0+estimated_align[:,None]*v0+.5*estimated_align[:,None]**2*g0
    v=v0+estimated_align[:,None]*g0

    # Final body frame includes arbitrary but fixed roll; x-axis is what matters.
    # Translational cross-axis nominal effects are included with this chosen roll.
    x= directions
    zref=np.where((abs(x[:,2])<.85)[:,None],np.array([0.,0.,1.]),np.array([0.,1.,0.]))
    y=np.cross(zref,x)
    y/=np.maximum(np.linalg.norm(y,axis=1)[:,None],1e-12)
    z=np.cross(x,y)
    effect_body=model.mean_acceleration(np.array([cfg['burn']['request_x_m_s2'],0,0,0,0,0]))[:3]
    applied=x*effect_body[0]+y*effect_body[1]+z*effect_body[2]
    report_offset=x*model.ell[0]+y*model.ell[1]+z*model.ell[2]
    burn_fuel_rate=fuel_per_second(model,[cfg['burn']['request_x_m_s2'],0,0,0,0,0])
    turn_fuel_rate=fuel_per_second(model,[0,0,0,0,cfg['attitude']['max_request_rad_s2'],0])

    horizon=cfg['planning']['horizon_s']
    h=cfg['planning']['integration_step_s']
    count=int(math.ceil(horizon/h))
    arrival=np.full(n,np.inf)
    arrival_speed=np.full(n,np.inf)
    min_dist=np.full(n,np.inf)
    when_closest=np.full(n,np.inf)
    valid=np.ones(n,bool)
    hit_radius=cfg['planning']['arrival_radius_m']

    for i in range(count):
        t=i*h
        dh=min(h,horizon-t)
        fraction=np.clip((burns-t)/dh,0.,1.)
        grav=model_gravity_batch(p,model)
        acc=grav+fraction[:,None]*applied
        p+=v*dh+.5*acc*dh*dh
        v+=acc*dh
        elapsed=t+dh
        goal=target.at(estimated_align+elapsed)
        dist=np.linalg.norm(p+report_offset-goal,axis=1)
        new_closest=dist<min_dist
        min_dist=np.minimum(min_dist,dist)
        when_closest=np.where(new_closest,elapsed,when_closest)
        newly=(dist<=hit_radius)&np.isinf(arrival)
        arrival=np.where(newly,elapsed,arrival)
        arrival_speed=np.where(newly,np.linalg.norm(v-target.v,axis=1),arrival_speed)
        # Invalid if approaching the origin unphysically; not a terrain collision test.
        valid &= (np.linalg.norm(p,axis=1)>100.)

    total_T=start_sim_s+estimated_align+arrival
    fuel0=float(state['fuel_used_kg'])
    nominal_fuel=(burn_fuel_rate*np.minimum(burns,np.where(np.isfinite(arrival),arrival,horizon))+
                  turn_fuel_rate*estimated_align)
    fuel=fuel0+nominal_fuel
    scorecfg=cfg['objective']['score']
    score=50.+100.*(scorecfg['nominal_time_s']/np.maximum(total_T,.01)-
                   fuel/scorecfg['nominal_fuel_kg']-
                   2.*float(state['damage_J'])/scorecfg['damage_threshold_J'])
    feasible= np.isfinite(arrival)&valid
    if cfg['objective']['mode']=='score':
        objective=-score
    else:
        w=cfg['objective']['quadratic']
        v_ref=max(w['velocity_scale_m_s'],1e-3)
        objective=(w['Q_time']*(total_T/scorecfg['nominal_time_s'])**2 +
                   w['Q_terminal_velocity']*(arrival_speed/v_ref)**2 +
                   w['R_fuel']*(fuel/scorecfg['nominal_fuel_kg'])**2 +
                   w['R_turn']*turns**2)
    objective=np.where(feasible,objective,1e6+np.maximum(0.,min_dist-hit_radius)*100. +fuel)
    return dict(cost=objective,feasible=feasible,aim=directions,burns=burns,
                arrival=arrival,score=score,fuel=fuel,min_dist=min_dist,
                estimated_align=estimated_align,turn_rad=turns,
                closest_at_s=when_closest)


def aim_base(state,model,target,cfg):
    p,v,_,_,_=latent_from_reported(state,model)
    T=cfg['planning']['initial_time_guess_s']
    mu=float(model.p['phobos_mu_prior_m3_s2'])
    g=-mu*p/np.linalg.norm(p)**3+np.asarray(model.p['uniform_external_acceleration_P_m_s2'])
    return unit(target.at(T)-(p+v*T+.5*g*T*T), target.p-p)


def select_plan(state,model,target,cfg,start_sim_s,force_direction=None,known_aligned=False):
    plan=cfg['planning']
    base=aim_base(state,model,target,cfg) if force_direction is None else unit(force_direction)
    bmins=plan['burn_min_s'];bmax=plan['burn_max_s']
    levels=plan['grid_refinement_levels']
    half=plan['cone_half_angle_deg']
    wide=np.linspace(-half,half,5)
    burns=np.linspace(bmins,bmax,plan['burn_samples'])
    if force_direction is not None:
        collection=np.array([(0.,0.,b) for b in burns])
    else:
        collection=np.array(list(itertools.product(wide,wide,burns)),float)
    best=None
    top=[]
    for level in range(levels+1):
        res=run_candidates(state,model,target,cfg,collection,base,start_sim_s,
                           force_direction,known_aligned)
        order=np.argsort(res['cost'])
        ix=int(order[0])
        if best is None or res['cost'][ix]<best[0]:
            best=(float(res['cost'][ix]),collection[ix].copy(),{k: (v[ix].copy() if isinstance(v,np.ndarray) and v.ndim>1 else v[ix]) for k,v in res.items()})
        if level==levels:
            break
        if force_direction is not None:
            stride=(bmax-bmins)/(plan['burn_samples']-1)/(3**level)
            center=float(collection[ix,2]); collection=np.array([(0.,0.,b) for b in
                 np.linspace(max(bmins,center-stride),min(bmax,center+stride),15)])
            continue
        # Refine both possible hitting solutions and closest misses. The success
        # basin can be extremely narrow relative to the coarse angular grid.
        step_a=half/(2.**(level+1))
        step_b=(bmax-bmins)/max(plan['burn_samples']-1,1)/(2.**level)
        selected=order[:min(5,len(order))]
        rows=[]
        for k in selected:
            cy,cp,cb=collection[int(k)]
            for oy,op,ob in itertools.product((-1.,0.,1.),(-1.,0.,1.),(-1.,0.,1.)):
                yy=float(np.clip(cy+oy*step_a,-half,half))
                pp=float(np.clip(cp+op*step_a,-half,half))
                bb=float(np.clip(cb+ob*step_b,bmins,bmax))
                rows.append((yy,pp,bb))
        collection=np.unique(np.asarray(rows),axis=0)
    entry=best[2]
    return dict(feasible=bool(entry['feasible']),aim_P=entry['aim'].tolist(),
                burn_s=float(entry['burns']),estimated_align_s=float(entry['estimated_align']),
                estimated_score=float(entry['score']) if np.isfinite(entry['score']) else None,
                estimated_fuel_kg=float(entry['fuel']),
                estimated_arrival_sim_s=(start_sim_s+float(entry['estimated_align'])+float(entry['arrival'])
                  if np.isfinite(entry['arrival']) else None),
                predicted_closest_m=float(entry['min_dist']),objective_value=float(best[0]),
                nominal_assumptions='fixed aim/roll; continuous mean allocator; target extrapolation; no debris or actual attitude transient')


def point_command(q,omegaP,aim_P,model,cfg):
    from system_identification.model import matrix
    C=matrix(q)
    desired=unit(C@aim_P)
    current=np.array([1.,0.,0.])
    cross=np.cross(current,desired)
    theta=math.atan2(np.linalg.norm(cross),float(np.clip(current@desired,-1,1)))
    if theta>math.pi-1e-6:
        cross=np.array([0.,0.,1.]);theta=math.pi
    error=unit(cross,[0.,0.,1.])*theta if theta>1e-10 else np.zeros(3)
    omega=C@np.asarray(omegaP,float)
    # Rest-to-rest bang-bang switching surface, with bounded near-target damping.
    a=cfg['attitude']['estimated_effective_ang_accel_rad_s2']
    surface=error-omega*np.abs(omega)/(2*a)
    dead_rad=math.radians(cfg['attitude']['angle_tolerance_deg'])
    rate_tol=cfg['attitude']['rate_tolerance_rad_s']
    if theta<dead_rad and np.linalg.norm(omega)<rate_tol:
        return np.zeros(3),theta,float(np.linalg.norm(omega)),True
    cmd=np.where(np.abs(surface)>cfg['attitude']['surface_deadband_rad'],np.sign(surface),
                 surface/max(cfg['attitude']['surface_deadband_rad'],1e-8))
    # Apply finite angular acceleration norm and reduce roll only if nonzero rate.
    cmd=cap(cmd*cfg['attitude']['max_request_rad_s2'],cfg['attitude']['max_request_rad_s2'])
    return cmd,theta,float(np.linalg.norm(omega)),False


def radar_lateral_body(rows, C, now, settings, velocity_P=None):
    """Predict nearby closing threats with *public* radar observations only.

    The radar provides a relative center, radial speed and average diameter, but
    no reliable tangential debris velocity or persistent object ID. We assume
    debris's unknown transverse velocity is zero in the P frame, then correct
    the estimated radial component to match the measured radar speed. This
    yields a heuristic closest-approach estimate, NOT a collision guarantee.

    Importantly, we do not discard objects behind the body +X hemisphere.
    The output dodge is always a body-frame Y/Z request, bounded to 0.5 m/s^2.
    """
    cfg=settings['avoidance']
    base={'hazards':0,'min_margin_m':None,'stale':0,'rear_hazards':0,
          'closest_time_s':None,'emergency':False,'assessment':'public_radar_heuristic'}
    if not cfg['enabled']:
        return np.zeros(3),base
    C=np.asarray(C,float)
    vel=np.zeros(3) if velocity_P is None else np.asarray(velocity_P,float)
    if vel.shape!=(3,) or not np.isfinite(vel).all():
        vel=np.zeros(3)
    min_margin=np.inf
    candidates=[]
    for row in rows:
        z=np.asarray(row,float)
        if z.shape!=(3,3) or not np.isfinite(z).all():continue
        age=max(0.,float(now)-float(z[2,0]))
        if age>cfg['max_radar_age_s']:
            base['stale']+=1
            continue
        rP=z[0,:];distance=float(np.linalg.norm(rP))
        if distance<1e-6:continue
        radial=float(z[1,1])
        closing=-radial
        if closing<cfg['min_closing_speed_m_s']:
            continue
        line=rP/distance
        # Public radar radial relative speed; unknown tangential debris speed.
        # In P-frame assume zero transverse debris velocity for prediction.
        v_rel=-vel
        v_rel=v_rel+(radial-float(v_rel@line))*line
        if not np.isfinite(v_rel).all():continue
        r_now=rP+age*v_rel
        v2=float(v_rel@v_rel)
        if v2<1e-10:continue
        t_cpa=float(np.clip(-float(r_now@v_rel)/v2,0.,cfg['lookahead_s']))
        if t_cpa<=0:continue
        r_at_cpa=r_now+t_cpa*v_rel
        # Diameter and surface-range are independently noisy. Use the larger
        # implied radius, then subtract configured padding and age uncertainty.
        radius=max(.5*max(0.,float(z[1,2])),
                   distance-max(0.,float(z[1,0])))
        surface_at_cpa=float(np.linalg.norm(r_at_cpa))-radius
        margin=surface_at_cpa-cfg['clearance_padding_m']-age*cfg['radar_age_uncertainty_m_s']
        min_margin=min(min_margin,margin)
        if margin>0:continue
        # Treat a projected near miss as relevant in ANY body hemisphere.
        rB=C@r_now
        futureB=C@r_at_cpa
        sepYZ=futureB[1:]
        if np.linalg.norm(sepYZ)<1e-8:
            sepYZ=rB[1:]
        away=-unit(sepYZ,[0.,1.])
        surface_now=max(0.,float(z[1,0])-closing*age)
        t_surface=surface_now/max(closing,1e-6)
        emergency=(min(t_cpa,t_surface)<=cfg['emergency_time_to_threat_s']
                   and surface_now<=cfg['emergency_surface_range_m'])
        base['rear_hazards']+=int(rB[0]<=0.)
        base['hazards']+=1
        candidates.append((margin, min(t_surface,t_cpa), away, emergency,
                           surface_now, float(rB[0])))
    if np.isfinite(min_margin):base['min_margin_m']=float(min_margin)
    if not candidates:return np.zeros(3),base
    # Choose most urgent threat instead of cancelling opposing avoidance vectors.
    # Under multiple hazards this is only a heuristic; no global clearance proof.
    candidates.sort(key=lambda v:(v[1] if v[3] else cfg['lookahead_s']+v[1],v[0]))
    margin,tgo,away,emergency,surface_now,body_x=candidates[0]
    base.update(closest_time_s=float(tgo),emergency=bool(emergency),
                selected_surface_range_m=float(surface_now),
                selected_body_x_m=float(body_x))
    magnitude=(cfg['emergency_lateral_request_m_s2'] if emergency
               else cfg['max_lateral_request_m_s2'])
    magnitude=float(np.clip(magnitude,0.,.5))
    return np.r_[0.,magnitude*away],base


def linearized_qr_gain(p, aim_P,model,cfg):
    """Local 6D translation LQR about a fixed final pointing orientation.

    State e=[delta p_P;delta v_P], input delta achieved acceleration_B.
    A includes Phobos gravity Jacobian; B uses fixed final body->P rotation.
    The inverse allocator is applied separately to the requested correction.
    """
    from scipy.linalg import solve_continuous_are
    p=np.asarray(p,float);rho=np.linalg.norm(p);mu=model.p['phobos_mu_prior_m3_s2']
    G=-mu*(np.eye(3)/rho**3-3*np.outer(p,p)/rho**5)
    A=np.block([[np.zeros((3,3)),np.eye(3)],[G,np.zeros((3,3))]])
    B=np.vstack((np.zeros((3,3)),axes_for_direction(aim_P)))
    qr=cfg['objective']['quadratic']
    Q=np.diag([qr['Q_position']]*3+[qr['Q_velocity']]*3)
    R=np.diag([qr['R_correction']]*3)
    P=solve_continuous_are(A,B,Q,R)
    return np.linalg.solve(R,B.T@P)


def coast_trim(state,model,target,cfg,when,aim_P,qr_gain=None):
    """Compute a small body-frame acceleration request for terminal correction."""
    from system_identification.model import matrix
    p,v,q,_,C=latent_from_reported(state,model)
    goal=np.asarray(state['target_P_m'],float)
    d=goal-np.asarray(state['position_P_m'],float)
    target_velocity=target.v if target.mode != 'fixed' else np.zeros(3)
    closing=np.dot(v-target_velocity,unit(d))
    togo=np.clip(np.linalg.norm(d)/max(closing,1.0),
                   cfg['feedback']['min_time_to_go_s'],cfg['feedback']['max_time_to_go_s'])
    if qr_gain is not None and np.linalg.norm(d)<cfg['feedback']['qr_activation_range_m']:
        desired_B=-qr_gain@np.r_[np.asarray(state['position_P_m'],float)-goal,v-target_velocity]
        desired_B=cap(desired_B,cfg['feedback']['max_delivered_trim_m_s2'])
        req=model.request_for_mean_acceleration(np.r_[desired_B,np.zeros(3)])['request'][:3]
        return cap(req,cfg['feedback']['max_request_trim_m_s2']),{'type':'qr','togo_s':float(togo)}
    if not cfg['feedback']['coast_trim_enabled']:
        return np.zeros(3),{'type':'none','togo_s':float(togo)}
    n=max(1,int(math.ceil(togo/2.)))
    h=togo/n
    p_future=p.copy();v_future=v.copy()
    for _ in range(n):
        grav=model_gravity_batch(p_future[None,:],model)[0]
        p_future+=v_future*h+.5*grav*h*h
        v_future+=grav*h
    future_goal=target.at(togo)
    p_future_report=p_future+axes_for_direction(aim_P)@model.ell
    miss=future_goal-p_future_report
    if np.linalg.norm(miss)<cfg['feedback']['trim_deadband_m']:
        return np.zeros(3),{'type':'coast','togo_s':float(togo),'predicted_miss_m':float(np.linalg.norm(miss))}
    accel_P=cap(2*miss/max(togo*togo,1.),cfg['feedback']['max_delivered_trim_m_s2'])
    desired_B=C@accel_P
    req=model.request_for_mean_acceleration(np.r_[desired_B,np.zeros(3)])['request'][:3]
    return cap(req,cfg['feedback']['max_request_trim_m_s2']),{'type':'coast','togo_s':float(togo),'predicted_miss_m':float(np.linalg.norm(miss))}
