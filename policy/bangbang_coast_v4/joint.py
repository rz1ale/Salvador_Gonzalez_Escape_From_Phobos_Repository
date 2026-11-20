"""V3 joint longitudinal burn / short body-YZ trim optimization.

Physics assumptions are explicit: held current attitude and native-fitted mean
jet allocator, approximate Phobos gravity, public current own/target state.
Every candidate is a bang-off +X burn, optionally with an end-of-burn
body-YZ request. Terminal state is the FIRST INTERSECTION with the capture
sphere, NOT an imposed fixed x_f or a target-centered stopping condition.
No terminal velocity penalty. All algorithms are offline-safe / SDK-free.
"""
from __future__ import annotations

import math
import numpy as np
from system_identification.model import matrix
from policy.bangbang_coast_v4.core import latent_from_reported, fuel_per_second


def first_sphere_entry(relative_before, relative_after, radius):
    """Fraction [0,1] of the segment entering the sphere; NaN if none.

    Both relative vectors are Nx3. Segment interpolation resolves boundary
    crossings between numerical samples; does not replace integration checks.
    """
    r0=np.asarray(relative_before,float)
    r1=np.asarray(relative_after,float)
    if r0.shape!=r1.shape or r0.ndim!=2 or r0.shape[1]!=3:
        raise ValueError('Expected matching Nx3 relative positions')
    d=r1-r0
    aa=np.einsum('ij,ij->i',d,d)
    bb=2*np.einsum('ij,ij->i',r0,d)
    cc=np.einsum('ij,ij->i',r0,r0)-float(radius)**2
    ans=np.full(len(r0),np.nan)
    ans[cc<=0]=0.
    good=(cc>0)&(aa>1e-15)
    disc=bb**2-4*aa*cc
    good &= (disc>=0)&(bb<0)
    indices=np.flatnonzero(good)
    if len(indices):
        root=(-bb[indices]-np.sqrt(np.maximum(disc[indices],0)))/(2*aa[indices])
        fits=(root>=-1e-10)&(root<=1+1e-10)
        ans[indices[fits]]=np.clip(root[fits],0,1)
    return ans


def _directions():
    """Unit lateral requests; no longitudinal braking."""
    inv=1/math.sqrt(2)
    return [(0.,0.),(1.,0.),(-1.,0.),(0.,1.),(0.,-1.),
            (inv,inv),(inv,-inv),(-inv,inv),(-inv,-inv)]


def search_joint_burn(state, model, target, cfg, now, remaining_burn_s):
    """Choose max-score near-term +X/off or +X+Y/Z/off schedule from *current* state.

    Returns a dict of scalar/list values safe for JSON event logging.
    This is an OPTIMISTIC prediction, not a collision-free trajectory guarantee.
    """
    jc=cfg['burn']['joint_optimizer']
    if not jc['enabled']:
        return None
    horizon=min(float(jc['prediction_horizon_s']),
                float(cfg['safety']['mission_timeout_s']-(now-cfg.get('_start_sim',0.))))
    if horizon<=1:
        return None
    remaining=max(0.,float(remaining_burn_s))
    if not np.isfinite(remaining):
        raise ValueError('Invalid remaining burn allowance')
    max_burn=min(float(jc['future_burn_window_s']),remaining)
    if max_burn<=0:
        burns=np.array([0.])
    else:
        burns=np.unique(np.r_[0.,np.linspace(0,max_burn,int(jc['burn_samples'])),
                                   np.minimum(max_burn,float(jc['short_burn_sample_s']))])
    lateral_directions=_directions() if jc['include_diagonals'] else _directions()[:5]
    profiles=[(0.,0.,0.)]
    for mag in jc['lateral_request_magnitudes_m_s2']:
        for yy,zz in lateral_directions[1:]:
            profiles.append((float(mag)*yy,float(mag)*zz,1.))
    candidates=[]
    for burn in burns:
        candidates.append((float(burn),0.,0.,0.))
        if burn <=0:continue
        for yy,zz,_ in profiles[1:]:
            for duration in jc['lateral_pulse_durations_s']:
                if duration>burn+1e-9:continue
                candidates.append((float(burn),float(yy),float(zz),float(duration)))
    if len(candidates)>jc['max_candidates']:
        raise ValueError('Joint candidate count exceeds configured max_candidates')
    table=np.asarray(candidates,float)
    n=len(table)
    btime=table[:,0]
    lat=table[:,1:3]
    pulse=table[:,3]
    pure_end=np.maximum(btime-pulse,0.)
    p0,v0,q,w,C=latent_from_reported(state,model)
    p=np.broadcast_to(p0,(n,3)).copy()
    v=np.broadcast_to(v0,(n,3)).copy()
    report=C.T@model.ell
    mx=float(cfg['burn']['request_x_m_s2'])
    if not 0<mx<=.5:
        raise ValueError('Unexpected forward request cap')
    xmix=np.minimum(mx,np.sqrt(np.maximum(0.,.5**2-np.einsum('ij,ij->i',lat,lat))))
    if np.any(np.linalg.norm(lat,axis=1)>.5+1e-10):
        raise ValueError('Invalid lateral request')
    pure_u=np.r_[mx,0,0,0,0,0]
    pure_acc=C.T@model.mean_acceleration(pure_u)[:3]
    pure_rate=fuel_per_second(model,pure_u)
    uniq={}
    acc_mix=np.zeros((n,3));rate_mix=np.zeros(n)
    for i in range(n):
        key=(float(xmix[i]),float(lat[i,0]),float(lat[i,1]))
        if key not in uniq:
            bcmd=np.r_[list(key),[0.,0.,0.]]
            uniq[key]=(C.T@model.mean_acceleration(bcmd)[:3],fuel_per_second(model,bcmd))
        acc_mix[i],rate_mix[i]=uniq[key]
    # Score objective; source objective uses nominal T/F and accumulated actual fuel.
    scorecfg=cfg['objective']['score']
    fuel0=float(state['fuel_used_kg'])
    damage=float(state['damage_J'])
    fuel=np.zeros(n)
    hit=np.full(n,np.nan)
    closest=np.full(n,np.inf)
    closest_time=np.zeros(n)
    hit_point=np.full((n,3),np.nan)
    at_closest=np.full((n,3),np.nan)
    entered=np.zeros(n,dtype=bool)
    t=0.
    step=float(jc['integration_step_s'])
    mu=float(model.p['phobos_mu_prior_m3_s2'])
    bias=np.asarray(model.p['uniform_external_acceleration_P_m_s2'],float)
    capture=float(jc['capture_radius_m'])
    g0=np.asarray(target.at(0.),float)
    relative=p+report-g0
    distance=np.linalg.norm(relative,axis=1)
    closest=np.minimum(closest,distance)
    at_closest[:]=p+report
    immediate=np.flatnonzero(distance<=capture)
    if len(immediate):
        hit[immediate]=0.
        hit_point[immediate]=p[immediate]+report
        entered[immediate]=True
    while t<horizon-1e-8 and not np.all(entered):
        h=min(step,horizon-t)
        # Fraction of interval belonging to each actuator phase.
        pure_frac=np.maximum(0.,np.minimum(t+h,pure_end)-t)/h
        mix_frac=np.maximum(0.,np.minimum(t+h,btime)-np.maximum(t,pure_end))/h
        # Gravity midpoint rule; omit angular evolution under nominal held orientation.
        midpoint=p+v*(h/2.)
        radius=np.linalg.norm(midpoint,axis=1)
        if np.any(radius<100.):
            raise ValueError('Nominal joint rollout passed below 100 m radius')
        grav=-mu*midpoint/radius[:,None]**3+bias
        a=grav+pure_frac[:,None]*pure_acc+mix_frac[:,None]*acc_mix
        new_p=p+v*h+.5*a*h*h
        new_v=v+a*h
        new_fuel=fuel+h*(pure_frac*pure_rate+mix_frac*rate_mix)
        rel_new=new_p+report-np.asarray(target.at(t+h),float)
        dist=np.linalg.norm(rel_new,axis=1)
        improved=dist<closest
        closest[improved]=dist[improved]
        closest_time[improved]=t+h
        at_closest[improved]=new_p[improved]+report
        entering=first_sphere_entry(relative,rel_new,capture)
        newly=(~entered)&np.isfinite(entering)
        if np.any(newly):
            frac=entering[newly]
            hit[newly]=t+h*frac
            hit_point[newly]=p[newly]+(new_p[newly]-p[newly])*frac[:,None]+report
            fuel[newly]=fuel[newly]+frac*(new_fuel[newly]-fuel[newly])
        continuing=~entered & ~newly
        fuel[continuing]=new_fuel[continuing]
        entered |= newly
        p=new_p
        v=new_v
        relative=rel_new
        t+=h
    feasible=np.isfinite(hit)
    score=50.+100.*(float(scorecfg['nominal_time_s'])/np.maximum(now+hit,1e-6)-
                    (fuel0+fuel)/float(scorecfg['nominal_fuel_kg'])-
                    2.*damage/float(scorecfg['damage_threshold_J']))
    # Constrained terminal region, no terminal velocity or exact point penalty.
    if cfg['objective']['mode']=='score':
        cost=-score
    else:
        weights=cfg['objective']['quadratic']
        cost=(weights['Q_time']*((now+hit)/scorecfg['nominal_time_s'])**2+
              weights['R_fuel']*((fuel0+fuel)/scorecfg['nominal_fuel_kg'])**2+
              weights['R_lateral']*(pulse/3.)**2)
    cost=np.where(feasible,cost,np.inf)
    if np.any(feasible):
        idx=int(np.argmin(cost))
    else:
        idx=int(np.argmin(closest+jc['unreachable_fuel_weight_m_per_kg']*fuel))
    def record(k):
        return {'feasible':bool(feasible[k]),'future_burn_s':float(btime[k]),
            'lateral_request_B':[0.,float(lat[k,0]),float(lat[k,1])],
            'lateral_pulse_s':float(pulse[k]),
            'lateral_start_s':float(pure_end[k]),
            'predicted_capture_sim_s':float(now+hit[k]) if feasible[k] else None,
            'predicted_entry_position_P_m':(hit_point[k].tolist() if feasible[k] else None),
            'predicted_closest_m':float(closest[k]),
            'predicted_closest_position_P_m':at_closest[k].tolist(),
            'predicted_total_fuel_kg':float(fuel0+fuel[k]),
            'predicted_score':float(score[k]) if feasible[k] else None,
            'objective':float(cost[k]) if feasible[k] else None}
    pure=(lat[:,0]==0)&(lat[:,1]==0)&(pulse==0)
    purefeasible=np.flatnonzero(pure&feasible)
    pure_index=int(purefeasible[np.argmin(cost[purefeasible])]) if len(purefeasible) else None
    pure_miss_index=int(np.flatnonzero(pure)[np.argmin(closest[pure])])
    return {'best':record(idx),'best_pure':record(pure_index) if pure_index is not None else None,
            'best_pure_miss':record(pure_miss_index),
            'immediate_coast':record(0),
            'n_candidates':int(n),'n_feasible':int(feasible.sum()),
            'prediction_horizon_s':float(horizon),'capture_radius_m':capture,
            'model_assumption':'held measured orientation; actual body 6D mean allocator; midpoint gravity; first 3D capture-boundary entry; no angular corrections, obstacles or report latency'}
