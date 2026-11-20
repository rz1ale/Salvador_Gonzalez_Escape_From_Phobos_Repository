"""Low-dimensional body-frame impulse-equivalent correction search; public observations only.

Predict each candidate against the unchanged nominal nonlinear mean-actuator model.
This does not certify debris clearance or exact native impulse timing.
"""
from __future__ import annotations
import math
import numpy as np

from policy.bangbang_coast_v3.coast_monitor import predict_ballistic
from policy.bangbang_coast_v3.core import TargetMotion, fuel_per_second, latent_from_reported, unit
from system_identification.model import matrix


def _reported_state_after_step(state, model, command, duration):
    p,v,q,w,C=latent_from_reported(state,model)
    x=np.r_[p,v,q,w]
    next_x=model.step(x,np.r_[command,np.zeros(3)],duration)
    R=matrix(next_x[6:10]);out=dict(state)
    out['position_P_m']=(next_x[:3]+R.T@model.ell).tolist()
    out['velocity_P_m_s']=(next_x[3:6]+model.vbias).tolist()
    out['q_PB_xyzw_reported']=next_x[6:10].tolist()
    out['omega_relP_P_rad_s']=(R.T@(next_x[10:13]+model.wbias)).tolist()
    return out


def _directions(err_B,allow_longitudinal):
    directions=[]
    if allow_longitudinal:
        directions.extend([unit(err_B,[0,1,0]),-unit(err_B,[0,1,0])])
    lateral=np.r_[0.,err_B[1:]]
    if np.linalg.norm(lateral)>1e-8:
        for angle in (0.,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3):
            d=unit(lateral)
            directions.append([0., math.cos(angle)*d[1]-math.sin(angle)*d[2],
                                math.sin(angle)*d[1]+math.cos(angle)*d[2]])
    directions.extend([[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    if allow_longitudinal:
        directions.extend([[1,0,0],[-1,0,0]])
    results=[]
    for dr in directions:
        dr=unit(dr)
        if not any(np.dot(dr,existing)>0.999 for existing in results):
            results.append(dr)
    return results


def search_body_correction(state,model,target,cfg,now,baseline=None):
    """Select shortest-duration body-frame pulse yielding robust predicted capture.

    Returns None when no candidate in the constrained search predicts capture.
    Required inputs are the same public own-state and target fields as Flight.snapshot().
    """
    opt=cfg['coast']['direct_recovery']
    if not opt['enabled']:
        return None
    remaining=cfg['safety']['mission_timeout_s']-(now-cfg.get('_start_sim',0.))
    horizon=min(float(cfg['coast']['miss_monitor']['horizon_s']), max(0.,remaining))
    if horizon<=opt['min_time_to_closest_s']:
        return None
    if baseline is None:
        baseline=predict_ballistic(state,model,target,horizon,opt['integration_step_s'],opt['arrival_radius_m'])
    if baseline['predicted_capture']:
        return None
    tgo=float(baseline['time_to_closest_s'])
    if tgo<opt['min_time_to_closest_s'] or tgo>horizon+1e-6:
        return None
    # A shrinking time-to-go reduces our ability to demand a large capture margin.
    # Never exceed the official 50 m boundary. More conservative for long coasts.
    capture_radius=(opt['urgent_arrival_radius_m'] if tgo<=opt['urgent_time_to_closest_s']
                    else opt['arrival_radius_m'])
    miss_P=np.asarray(baseline['closest_target_P_m'])-np.asarray(baseline['closest_position_P_m'])
    _,_,_,_,C=latent_from_reported(state,model)
    miss_B=C@miss_P
    raw=[]
    for d in _directions(miss_B,opt['allow_longitudinal']):
        for amplitude in opt['request_magnitudes_m_s2']:
            request=np.asarray(d)*float(amplitude)
            delivered=model.mean_acceleration(np.r_[request,np.zeros(3)])[:3]
            accel_P=C.T@delivered
            effective=float(accel_P@miss_P)
            if effective<=0:continue
            time_need=(effective/max(float(accel_P@accel_P),1e-12))/max(tgo,1.)
            times=set(float(np.clip(time_need*f,opt['pulse_min_s'],opt['pulse_max_s'])) for f in (.7,1.,1.35))
            times.update([opt['pulse_min_s'],opt['pulse_max_s']])
            rate=fuel_per_second(model,np.r_[request,np.zeros(3)])
            for tau in sorted(times):
                displacement=accel_P*tau*max(tgo-.5*tau,0.)
                proxy=float(np.linalg.norm(miss_P-displacement))
                if proxy>opt['proxy_max_m']:
                    continue
                rank=(rate*tau + opt['proxy_violation_cost_kg_per_m']*max(0.,proxy-capture_radius))
                raw.append((rank,request,tau,proxy,rate*tau))
    raw.sort(key=lambda x:x[0])
    top=raw[:opt['max_detailed_candidates']]
    best=None
    for _,request,tau,proxy,estimated_fuel in top:
        try:
            updated=_reported_state_after_step(state,model,request,tau)
            future_target=TargetMotion(target.at(tau),target.v,target.mode)
            future=predict_ballistic(updated,model,future_target,horizon-tau,opt['integration_step_s'],capture_radius)
        except (ValueError,ArithmeticError,RuntimeError):
            continue
        if not future['predicted_capture']:
            continue
        arrival=now+tau+future['predicted_arrival_s']
        final_fuel=float(state['fuel_used_kg'])+estimated_fuel
        scorecfg=cfg['objective']['score']
        score=50+100*(scorecfg['nominal_time_s']/max(arrival,.01)-final_fuel/scorecfg['nominal_fuel_kg']-2*float(state['damage_J'])/scorecfg['damage_threshold_J'])
        if cfg['objective']['mode']=='score':
            objective=-score
        else:
            q=cfg['objective']['quadratic']; objective=q['Q_time']*(arrival/scorecfg['nominal_time_s'])**2+q['R_fuel']*(final_fuel/scorecfg['nominal_fuel_kg'])**2
        candidate=dict(kind='body_correction',request_B=request.tolist(),duration_s=float(tau),
                       estimated_fuel_delta_kg=float(estimated_fuel),predicted_score=float(score),
                       predicted_closest_m=future['predicted_closest_m'],
                       predicted_arrival_sim_s=float(arrival),objective=float(objective),
                       proxy_m=float(proxy))
        if best is None or candidate['objective']<best['objective']:
            best=candidate
    return best
