"""V4 SDK-free startup strategy selection and nominal fixed-direction interception.

A strategy changes the BODY thrust direction, desired inertial direction, and
whether translation is requested DURING the closed-loop attitude turn.

The predictor intentionally does not integrate full attitude feedback, jet delay,
or radar obstacles. It predicts one fixed attitude after alignment, with an
approximate integrated attitude transition for combined-turn strategies. All
choices are reevaluated after the actual turn using public measurements.
"""
from __future__ import annotations
import math
import numpy as np
from system_identification.model import matrix
from policy.bangbang_coast_v4.core import unit, cap, fuel_per_second, latent_from_reported, aim_base
from policy.bangbang_coast_v4.joint import first_sphere_entry

MODES=('no_orient','partial_turn','full_turn','rotate_and_thrust','auto_compare')


def check_startup(cfg):
    d=cfg['startup']
    if d['mode'] not in MODES:raise ValueError('Unsupported startup.mode')
    if d['no_orient_body_selection'] not in ('fixed','target_los','cone_search'):
        raise ValueError('Invalid no_orient_body_selection')
    if len(d['cone_offsets_deg'])<1 or len(d['cone_offsets_deg'])>9 or any(abs(float(x))>25 for x in d['cone_offsets_deg']):
        raise ValueError('Invalid cone_offsets_deg')
    if (not isinstance(d['search_body_candidates'],bool) or
        not all(m in MODES[:-1] for m in d['auto_modes']) or not d['auto_modes']):
        raise ValueError('Invalid startup mode/candidate search')
    if not 0<=d['partial_fraction']<=1 or not 0<=d['combined_turn_fraction']<=1:
        raise ValueError('Turn fraction must be in [0,1]')
    if not 0<=d['turn_translation_fraction']<=1:
        raise ValueError('Simultaneous translational thrust fraction must be in [0,1]')
    if (not 0<d['candidate_burn_min_s']<d['candidate_burn_max_s']<=200 or
        not 5<=d['candidate_burn_samples']<=150 or
        not 0<d['capture_radius_m']<50 or
        not 0<d['predict_step_s']<=5 or not 0<d['predict_horizon_s']<=1000 or
        not 1<=d['turn_duration_margin']<=3 or
        not 0<d['fallback_burn_s']<=200):
        raise ValueError('Invalid startup optimization bounds')
    def body(v):
        a=np.asarray(v,float)
        if a.shape!=(3,) or not np.isfinite(a).all() or np.linalg.norm(a)<1e-7:
            raise ValueError('Each startup body direction must be a finite nonzero 3-vector')
    body(d['body_direction_B'])
    for v in d['body_candidates_B']:body(v)
    for frac in d['auto_partial_fractions']:
        if not 0<=frac<=1:raise ValueError('Invalid auto partial fraction')
    online=d['online_burn']
    if (not isinstance(online['enabled'],bool) or
        not 0<online['check_period_s']<=30 or
        not 0<=online['min_elapsed_s']<=200 or
        not 0<online['future_window_s']<=200 or
        not 3<=online['candidate_samples']<=101 or
        online['stop_confirmations']<1 or
        not 0<=online['score_tolerance_points']<=10 or
        not 0<=online['max_extension_s']<=100 or
        not 0<online['max_total_burn_s']<=200):
        raise ValueError('Invalid startup online burn optimizer')
    return cfg


def rotation_between(start,end):
    """Proper minimal active rotation taking start P-vector onto end P-vector."""
    a=unit(start);b=unit(end)
    cross=np.cross(a,b)
    sn=float(np.linalg.norm(cross));co=float(np.clip(a@b,-1.,1.))
    angle=math.atan2(sn,co)
    if angle<1e-12:return np.eye(3),0.
    if sn<1e-12:
        ref=np.eye(3)[int(np.argmin(np.abs(a)))]
        axis=unit(np.cross(a,ref))
    else:axis=cross/sn
    K=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    R=np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
    return R,angle


def slerp_direction(a,b,fraction):
    """Minimal-angle partial reorientation; unlike Cartesian mix, well-defined near anti-parallel."""
    R,angle=rotation_between(a,b)
    if angle<1e-12:return unit(a)
    axis=unit(np.cross(unit(a),unit(b)), [0.,1.,0.])
    # For exact opposite directions reconstruct axis consistently with rotation_between.
    if abs(angle-math.pi)<1e-8:
        ref=np.eye(3)[int(np.argmin(np.abs(unit(a))))]
        axis=unit(np.cross(unit(a),ref))
    t=float(np.clip(fraction,0,1))*angle
    start=unit(a)
    return unit(start*math.cos(t)+np.cross(axis,start)*math.sin(t)+axis*(axis@start)*(1-math.cos(t)))


def candidate(mode,body_axis_B,turn_fraction,los,measured_C, cfg):
    body=unit(body_axis_B)
    current=unit(measured_C.T@body)
    if mode=='no_orient':frac=0.
    elif mode=='full_turn':frac=1.
    elif mode in ('partial_turn','rotate_and_thrust'):frac=float(turn_fraction)
    else:raise ValueError('Unknown candidate mode')
    aim=slerp_direction(current,los,frac)
    R,angle=rotation_between(current,aim)
    overlap= mode=='rotate_and_thrust' and angle>1e-6
    return {'mode':mode,'body_direction_B':body.tolist(),'aim_P':aim.tolist(),
            'turn_fraction':frac,'turn_angle_deg':math.degrees(angle),
            'overlap_translation':overlap}


def _gravity(pos,model):
    mu=float(model.p['phobos_mu_prior_m3_s2'])
    r=float(np.linalg.norm(pos))
    if r<1.:raise ValueError('Trajectory entered invalid central body radius')
    return -mu*pos/r**3+np.asarray(model.p['uniform_external_acceleration_P_m_s2'],float)


def predict(state,model,target,cfg,now,body_direction_B,aim_P,burns,turn_duration_s=0.,
            turn_fuel_kg=0.,overlap_fraction=0.,capture_radius_m=None):
    """Nominal first-capture integration for arbitrary body thrust with fixed final attitude.

    Returns per-burn trajectory metrics. Velocity at capture is intentionally unconstrained.
    Turn is approximated by a minimal active rotation, not a full dynamics replay.
    """
    p0,v0,_,_,C=latent_from_reported(state,model)
    uB=unit(body_direction_B);start=C.T@uB
    aim=unit(aim_P);R,angle=rotation_between(start,aim)
    CendT=R@C.T
    accB=model.mean_acceleration(np.r_[.5*uB,np.zeros(3)])[:3]
    accel_P=CendT@accB
    overlap_accB=(model.mean_acceleration(np.r_[.5*float(overlap_fraction)*uB,np.zeros(3)])[:3]
                  if overlap_fraction else np.zeros(3))
    fuel_rate=fuel_per_second(model,np.r_[.5*uB,np.zeros(3)])
    offset=CendT@model.ell
    dt=float(cfg['startup']['predict_step_s'])
    horizon=min(float(cfg['startup']['predict_horizon_s']),
                float(cfg['safety']['mission_timeout_s']-(now-cfg.get('_start_sim',0.))))
    burn=np.atleast_1d(np.asarray(burns,float))
    if np.any(~np.isfinite(burn)) or np.any(burn<0):raise ValueError('Invalid candidate burns')
    N=len(burn)
    p=np.repeat(np.asarray(p0,float)[None,:],N,axis=0)
    v=np.repeat(np.asarray(v0,float)[None,:],N,axis=0)
    t0=max(0.,float(turn_duration_s))
    # Approximate the alignment interval; thrust throughout at overlap_fraction.
    if t0>0:
        elapsed=0.
        while elapsed<t0-1e-8:
            h=min(dt,t0-elapsed)
            f=min(1.,(elapsed+.5*h)/t0)
            af=slerp_direction(start,aim,f)
            Rt,_=rotation_between(start,af)
            for i in range(N):
                grav=_gravity(p[i],model)
                applied=grav+(Rt@C.T@overlap_accB)
                p[i]+=v[i]*h+.5*applied*h*h
                v[i]+=applied*h
            elapsed+=h
    elapsed=0.
    first=np.full(N,np.inf)
    closest=np.full(N,np.inf)
    closest_at=np.full(N,np.nan)
    entry=np.full((N,3),np.nan)
    valid=np.ones(N,dtype=bool)
    rel_prev=p+offset-target.at(t0)
    already=np.linalg.norm(rel_prev,axis=1)<=float(capture_radius_m or cfg['startup']['capture_radius_m'])
    first[already]=0.;entry[already]=p[already]+offset
    while elapsed<horizon-t0-1e-8:
        h=min(dt,horizon-t0-elapsed)
        frac=np.clip((burn-elapsed)/h,0.,1.)
        p_next=p.copy()
        for i in range(N):
            grav=_gravity(p[i],model)
            a=grav+frac[i]*accel_P
            p_next[i]+=v[i]*h+.5*a*h*h
            v[i]+=a*h
        rel_next=p_next+offset-target.at(t0+elapsed+h)
        dist=np.linalg.norm(rel_next,axis=1)
        improve=dist<closest
        closest[improve]=dist[improve]
        closest_at[improve]=t0+elapsed+h
        cross=first_sphere_entry(rel_prev,rel_next,float(capture_radius_m or cfg['startup']['capture_radius_m']))
        found=np.isfinite(cross)&np.isinf(first)
        first[found]=elapsed+h*cross[found]
        entry[found]=p[found]+(p_next[found]-p[found])*cross[found,None]+offset
        p=p_next;rel_prev=rel_next;elapsed+=h
        valid &= np.linalg.norm(p,axis=1)>100.
    # Fuel should stop when first capture occurs, not continue accumulating to the horizon.
    # Recover analytical fuel from scheduled active durations and measured starting fuel.
    used_until=np.minimum(burn,np.where(np.isfinite(first),first,horizon))
    overlap_rate=(fuel_per_second(model,np.r_[.5*float(overlap_fraction)*uB,np.zeros(3)])
                  if overlap_fraction else 0.)
    spent=overlap_rate*t0+fuel_rate*used_until+float(turn_fuel_kg)
    totalfuel=float(state['fuel_used_kg'])+spent
    arrival=now+t0+first
    weights=cfg['objective']['score']
    score=50.+100.*weights['nominal_time_s']/np.maximum(arrival,1e-6) - 100.*totalfuel/weights['nominal_fuel_kg'] -2.*float(state['damage_J'])/weights['damage_threshold_J']
    feasible=np.isfinite(first)&valid
    if cfg['objective']['mode']=='score':
        cost=np.where(feasible,-score,1e6+np.maximum(0.,closest-float(capture_radius_m or cfg['startup']['capture_radius_m']))*100.)
    else:
        w=cfg['objective']['quadratic']
        cost=np.where(feasible,w['Q_time']*(arrival/weights['nominal_time_s'])**2+w['R_fuel']*(totalfuel/weights['nominal_fuel_kg'])**2,
                      1e6+np.maximum(0.,closest-float(capture_radius_m or cfg['startup']['capture_radius_m']))*100.)
    return {'feasible':feasible,'score':score,'cost':cost,'arrival_sim_s':arrival,
            'fuel_kg':totalfuel,'closest_m':closest,'entry_P_m':entry,'burn_s':burn,
            'turn_time_s':t0,'turn_angle_deg':math.degrees(angle),'predicted_coast_only':False}


def select_startup(state,model,target,cfg,now):
    d=cfg['startup'];mode=d['mode']
    C=matrix(np.asarray(state['q_PB_xyzw_reported'],float))
    axis_candidates=([unit(x) for x in d['body_candidates_B']] if d['search_body_candidates']
                     else [unit(d['body_direction_B'])])
    los=aim_base(state,model,target,cfg)
    specs=[]
    modes=d['auto_modes'] if mode=='auto_compare' else [mode]
    for m in modes:
        vectors=axis_candidates
        if m=='no_orient' and d['no_orient_body_selection']!='fixed':
            from policy.bangbang_coast_v4.core import basis_around
            center=unit(C@los)
            if d['no_orient_body_selection']=='target_los':
                vectors=[center]
            else:
                _,ey,ez=basis_around(center)
                vectors=[unit(center+math.tan(math.radians(float(y)))*ey+math.tan(math.radians(float(z)))*ez)
                          for y in d['cone_offsets_deg'] for z in d['cone_offsets_deg']]
        fractions=d['auto_partial_fractions'] if m in ('partial_turn','rotate_and_thrust') and mode=='auto_compare' else [d['partial_fraction'] if m=='partial_turn' else d['combined_turn_fraction']]
        for body in vectors:
            for fraction in fractions:
                specs.append(candidate(m,body,fraction,los,C,cfg))
    burns=np.linspace(d['candidate_burn_min_s'],d['candidate_burn_max_s'],d['candidate_burn_samples'])
    best=None;top=[]
    for spec in specs:
        angle=math.radians(spec['turn_angle_deg'])
        tturn=(2*math.sqrt(angle/max(cfg['attitude']['estimated_effective_ang_accel_rad_s2'],0.005))*d['turn_duration_margin'] if angle>1e-6 else 0.)
        # Estimate fuel for sustained symmetric torques; measured native may differ substantially.
        if angle>1e-6:
            axisP=np.cross(C.T@unit(spec['body_direction_B']),unit(spec['aim_P']))
            axisB=unit(C@axisP,[0.,1.,0.])
            a=np.r_[np.zeros(3),cfg['attitude']['max_request_rad_s2']*axisB]
            turn_rate=.5*(fuel_per_second(model,a)+fuel_per_second(model,-a))
        else:turn_rate=0.
        turn_fuel=turn_rate*tturn
        overlap=(d['turn_translation_fraction'] if spec['overlap_translation'] else 0.)
        result=predict(state,model,target,cfg,now,spec['body_direction_B'],spec['aim_P'],burns,tturn,turn_fuel,overlap)
        j=int(np.argmin(result['cost']))
        record=dict(spec,planned_burn_s=float(burns[j]),predicted_feasible=bool(result['feasible'][j]),
                    predicted_score=float(result['score'][j]) if result['feasible'][j] else None,
                    predicted_fuel_kg=float(result['fuel_kg'][j]),
                    predicted_arrival_sim_s=float(result['arrival_sim_s'][j]) if result['feasible'][j] else None,
                    predicted_closest_m=float(result['closest_m'][j]),
                    turn_duration_pred_s=float(tturn),turn_fuel_pred_kg=float(turn_fuel),
                    objective_value=float(result['cost'][j]),
                    objective_note='nominal 3D free-capture, approximate turn, no debris; no velocity penalty')
        top.append(record)
        if best is None or record['objective_value']<best['objective_value']:best=record
    if best is None:raise ValueError('No startup candidates')
    if not best['predicted_feasible']:
        best=dict(best,planned_burn_s=d['fallback_burn_s'],prediction_fallback=True)
    best=dict(best,considered=len(top),compared=[{'mode':e['mode'],
          'axis_B':e['body_direction_B'],'turn_fraction':e['turn_fraction'],
          'score':e['predicted_score'],'closest_m':e['predicted_closest_m'],
          'burn_s':e['planned_burn_s']} for e in sorted(top,key=lambda r:r['objective_value'])[:12]])
    return best


def evaluate_remaining(state,model,target,cfg,now,body_B,aim_P,max_remaining_s):
    d=cfg['startup']['online_burn']
    limit=max(0.,min(float(max_remaining_s),float(d['future_window_s'])))
    burns=np.linspace(0.,limit,int(d['candidate_samples']))
    # Burn forecast must use the ACTUAL measured body direction, not an
    # artificial instantaneous rotation back to the planned aim.
    current_aim=matrix(np.asarray(state['q_PB_xyzw_reported'],float)).T@unit(body_B)
    result=predict(state,model,target,cfg,now,body_B,current_aim,burns,
                   capture_radius_m=cfg['startup']['capture_radius_m'])
    feasible=np.flatnonzero(result['feasible'])
    i=int(feasible[np.argmin(result['cost'][feasible])]) if len(feasible) else int(np.argmin(result['cost']))
    def item(j):
        return {'future_burn_s':float(burns[j]),'feasible':bool(result['feasible'][j]),
                'predicted_score':float(result['score'][j]) if result['feasible'][j] else None,
                'predicted_capture_sim_s':float(result['arrival_sim_s'][j]) if result['feasible'][j] else None,
                'predicted_fuel_kg':float(result['fuel_kg'][j]),
                'predicted_closest_m':float(result['closest_m'][j])}
    return {'best':item(i),'immediate_coast':item(0),'n_feasible':int(len(feasible)),
            'capture_radius_m':float(cfg['startup']['capture_radius_m']),
            'model_assumption':'fixed measured attitude and thrust body vector; gravity; no radar/debris; terminal velocity free'}
