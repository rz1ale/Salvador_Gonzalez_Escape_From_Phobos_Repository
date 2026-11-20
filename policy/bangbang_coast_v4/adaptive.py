"""V2 offline-safe optimizers and feedback for the public 6-axis request interface.

Conventions: passive quaternion q_PB in xyzw, body +X is primary thrust axis.
No simulator objects or private telemetry are accessed here.
"""
from __future__ import annotations
import copy
import math
import numpy as np

from policy.bangbang_coast_v4.core import (cap, fuel_per_second, latent_from_reported,
                                         run_candidates, unit)


def pointing_geometry(state, aim_P, axis_B=(1.,0.,0.)):
    """Return body-frame aiming error, perpendicular angular rate, and angular norm.

    Body X roll is not a pointing error. During the main burn, do not spend fuel
    correcting roll unless it is needed to prevent another safety issue.
    """
    from system_identification.model import matrix
    C=matrix(np.asarray(state['q_PB_xyzw_reported'],float))
    d=unit(C @ unit(aim_P))
    axis=unit(axis_B)
    cross=np.cross(axis,d)
    theta=math.atan2(float(np.linalg.norm(cross)), float(np.clip(axis@d,-1.,1.)))
    if theta>math.pi-1e-7 and np.linalg.norm(cross)<1e-8:
        cross=np.array([0.,0.,1.])
    error=(unit(cross,[0.,0.,1.])*theta) if theta>1e-12 else np.zeros(3)
    wB=C@np.asarray(state['omega_relP_P_rad_s'],float)
    wP=wB-axis*float(wB@axis)
    return error,wP,float(np.linalg.norm(wB))


def choose_attitude_profile(state,aim_P,model,cfg,estimated_arrival_s,axis_B=(1.,0.,0.)):
    """Choose a single angular-request magnitude by approximate score-marginal cost.

    This is a nominal symmetric bang-bang estimate. Actual switching is closed-loop
    via measured quaternion/rate. Does not claim true fuel/time global optimum.
    """
    cfg_e=cfg['attitude']['economy']
    err,wperp,_=pointing_geometry(state,aim_P,axis_B)
    theta=np.linalg.norm(err)
    if theta<1e-7:
        return {'request_rad_s2':float(cfg['attitude']['max_request_rad_s2']),
                'effective_alpha_rad_s2':float(cfg['attitude']['estimated_effective_ang_accel_rad_s2']),
                'predicted_turn_s':0.,'predicted_turn_fuel_kg':0.,'choices':[]}
    axis=unit(err)
    score=cfg['objective']['score']
    # d(-S)/dT and d(-S)/dF at the expected mission arrival time.
    ref_T=max(30.,float(estimated_arrival_s or cfg_e['score_reference_arrival_s']))
    if cfg['objective']['mode']=='score':
        cost_t=(100*score['nominal_time_s']/ref_T**2)*cfg_e['time_cost_multiplier']
        cost_f=(100/score['nominal_fuel_kg'])*cfg_e['fuel_cost_multiplier']
    else:
        q=cfg['objective']['quadratic']
        cost_t=float(q['Q_time'])*cfg_e['time_cost_multiplier']/max(score['nominal_time_s'],1.)
        cost_f=float(q['R_fuel'])*cfg_e['fuel_cost_multiplier']/max(score['nominal_fuel_kg'],1.)
    choices=[]
    for raw in cfg_e['request_candidates_rad_s2']:
        raw=min(float(raw),float(cfg['attitude']['max_request_rad_s2']))
        aplus=np.r_[np.zeros(3),raw*axis]
        aminus=np.r_[np.zeros(3),-raw*axis]
        plus=float(axis@model.mean_acceleration(aplus)[3:])
        minus=float(-axis@model.mean_acceleration(aminus)[3:])
        a_eff=min(plus,minus)
        if not np.isfinite(a_eff) or a_eff<cfg_e['minimum_effective_accel_rad_s2']:
            continue
        # Conservative duration adjustment for nonzero perpendicular initial rate.
        duration=(2.*math.sqrt(theta/a_eff)+np.linalg.norm(wperp)/a_eff)*cfg_e['turn_duration_margin']
        if duration>cfg['attitude']['max_align_s']*0.95:
            continue
        fuel=duration*0.5*(fuel_per_second(model,aplus)+fuel_per_second(model,aminus))
        objective=cost_t*duration+cost_f*fuel
        choices.append({'request_rad_s2':raw,'effective_alpha_rad_s2':a_eff,
                        'predicted_turn_s':duration,'predicted_turn_fuel_kg':fuel,
                        'estimated_cost':objective})
    if not choices:
        return {'request_rad_s2':float(cfg['attitude']['max_request_rad_s2']),
                'effective_alpha_rad_s2':float(cfg['attitude']['estimated_effective_ang_accel_rad_s2']),
                'predicted_turn_s':None,'predicted_turn_fuel_kg':None,
                'choices':[],'fallback':True}
    # If magnitudes produce the SAME effective impulses/fuel (common with
    # normalized jet firing), keep the largest request to preserve the proven
    # nominal alignment behavior rather than 'optimizing' a fictitious gain.
    optimum=min(e['estimated_cost'] for e in choices)
    equivalent=[e for e in choices if e['estimated_cost'] <= optimum+1e-3]
    chosen=max(equivalent,key=lambda e:e['request_rad_s2'])
    return dict(chosen,choices=choices,fallback=False)


def alignment_command(state, aim_P, model, cfg, axis_B=(1.,0.,0.)):
    """Point +X using one vector braking surface; ignore irrelevant roll rate.

    The native jet allocator is nonlinear: tiny nonzero angular requests can still
    fire an appreciable impulse. Avoid analog gain tapering near the target.
    """
    att=cfg['attitude'];eco=att['alignment_control']
    error,wperp,_=pointing_geometry(state,aim_P)
    theta=float(np.linalg.norm(error));speed=float(np.linalg.norm(wperp))
    if theta<=math.radians(att['angle_tolerance_deg']) and speed<=att['rate_tolerance_rad_s']:
        return np.zeros(3),theta,speed,True
    # Approximate stopping distance at conservative attainable braking acceleration.
    axis=unit(-wperp,[0.,1.,0.]) if speed>1e-8 else unit(error,[0.,1.,0.])
    braking_command=np.r_[np.zeros(3),axis*att['max_request_rad_s2']]
    a_measured=float(axis @ model.mean_acceleration(braking_command)[3:])
    if not np.isfinite(a_measured) or a_measured<eco['minimum_braking_accel_rad_s2']:
        a_measured=float(att['estimated_effective_ang_accel_rad_s2'])
    a_switch=max(eco['minimum_braking_accel_rad_s2'],
                 min(a_measured*eco['braking_safety_factor'],eco['max_switch_accel_rad_s2']))
    switching=error-wperp*(speed/(2.*a_switch))
    if np.linalg.norm(switching)<eco['quiet_surface_rad'] and speed<eco['quiet_rate_rad_s']:
        return np.zeros(3),theta,speed,False
    command=unit(switching,[0.,1.,0.])*att['max_request_rad_s2']
    axis=unit(axis_B)
    command=command-axis*float(command@axis)
    command=cap(command,att['max_request_rad_s2'])
    return command,theta,speed,False


class AlignmentSwitchGuard:
    """Suppress rapid reversals of commanded torque during alignment.

    Does not enforce angular velocity = 0 in translation; only the existing
    pointing error and perpendicular angular-rate tolerances are used.
    On an early opposite command, send ZERO rather than continuing wrong torque.
    No prediction-only or simulator-private signals are used.
    """
    def __init__(self,cfg):
        self.p=cfg['attitude']['alignment_control']
        self.last_nonzero=np.zeros(3)
        self.last_switch=-1e20
        self.last_on=-1e20
        self.inhibited=0
        self.opposed_accepted=0
        self.reversals=0

    def filter(self,desired,now,settled=False):
        desired=np.asarray(desired,float).copy()
        if settled or np.linalg.norm(desired)<1e-10:
            return np.zeros(3)
        if np.linalg.norm(self.last_nonzero)<1e-10:
            self.last_nonzero=desired.copy()
            self.last_switch=float(now)
            self.last_on=float(now)
            return desired
        cosine=float(desired@self.last_nonzero /
                     max(np.linalg.norm(desired)*np.linalg.norm(self.last_nonzero),1e-12))
        if cosine<=self.p['opposing_direction_cosine']:
            if now-self.last_switch < self.p['minimum_reverse_dwell_s']:
                self.inhibited+=1
                return np.zeros(3)
            self.opposed_accepted+=1
            self.reversals+=1
            self.last_switch=float(now)
        self.last_nonzero=desired.copy()
        self.last_on=float(now)
        return desired


class BurnAttitudeHold:
    """Hysteretic, event-triggered attitude correction, pointing-only (no roll)."""
    def __init__(self,cfg):
        self.c=cfg['attitude']['burn_hold']
        self.active=False
        self.last_release=-1e20
        self.engagements=0
        self.active_updates=0
        self.total_updates=0
        self.burst_start=-1e20
        self.max_error_deg=0.

    def command(self,state,aim_P,model,cfg,now,axis_B=(1.,0.,0.)):
        self.total_updates+=1
        if not self.c['enabled']:
            from policy.bangbang_coast_v4.core import point_command
            a,theta,w,_=alignment_command(state,aim_P,model,cfg,axis_B)
            return a,{'active':True,'error_deg':math.degrees(theta),'rate_rad_s':w,
                      'predicted_drift_deg':None,'mode':'v1_continuous'}
        err,wperp,_=pointing_geometry(state,aim_P,axis_B)
        phi=float(np.linalg.norm(err));rate=float(np.linalg.norm(wperp))
        self.max_error_deg=max(self.max_error_deg,math.degrees(phi))
        # error_dot ~ -omega for small body-frame yaw/pitch excursions.
        predicted=float(np.linalg.norm(err-wperp*self.c['predicted_drift_s']))
        activate= (max(phi,predicted)>=math.radians(self.c['activate_error_deg']) or
                   rate>=self.c['activate_rate_rad_s'])
        release=(phi<=math.radians(self.c['release_error_deg']) and
                 rate<=self.c['release_rate_rad_s'] and
                 predicted<=math.radians(self.c['activate_error_deg'])*1.25)
        emergency=(phi>=math.radians(self.c['emergency_error_deg']) or
                   rate>=self.c['emergency_rate_rad_s'])
        if self.active:
            # Limit ordinary correction episodes; this is a deliberate fuel/direction
            # compromise, with an emergency override for large pointing/rate error.
            exhausted=now-self.burst_start>=self.c['max_burst_s']
            if release or (exhausted and not emergency):
                self.active=False
                self.last_release=now
        elif activate:
            if (now-self.last_release>=self.c['min_off_s'] or emergency):
                self.active=True
                self.engagements+=1
                self.burst_start=now
        angular=np.zeros(3)
        if self.active:
            a=self.c['estimated_effective_accel_rad_s2']
            # Same switching surface as v1, without roll correction.
            switch=err-wperp*np.abs(wperp)/(2*a)
            angular=np.sign(switch)*np.minimum(np.abs(switch)/max(cfg['attitude']['surface_deadband_rad'],1e-6),1.)
            angular=cap(angular*self.c['max_request_rad_s2'],self.c['max_request_rad_s2'])
            axis=unit(axis_B)
            angular=angular-axis*float(angular@axis)
            self.active_updates+=1
        return angular,{'active':self.active,'error_deg':math.degrees(phi),
                        'rate_rad_s':rate,'predicted_drift_deg':math.degrees(predicted),
                        'mode':'hysteretic_pointing_only'}


def evaluate_burn_stop(state,model,target,cfg,now,max_remaining_s):
    """Compare immediate coast to additional +X burn under CURRENT observed state.

    Uses nonlinear gravity and physical jet mean allocator from model.py.
    Unlike v1's one-time burn schedule, this is refreshed during the powered phase.
    Assumes future pure +X and fixed current attitude: a prediction, not guarantee.
    """
    from system_identification.model import matrix
    b=cfg['burn']['online_stop']
    limit=max(0.,min(float(max_remaining_s),b['future_burn_window_s']))
    burns=np.linspace(0.,limit,max(2,int(b['candidate_samples'])))
    C=matrix(np.asarray(state['q_PB_xyzw_reported'],float))
    aim=C.T@np.array([1.,0.,0.])
    local=copy.deepcopy(cfg)
    local['planning']['arrival_radius_m']=b['arrival_radius_m']
    local['planning']['integration_step_s']=b['integration_step_s']
    local['planning']['burn_max_s']=max(local['planning']['burn_max_s'],limit)
    candidates=np.c_[np.zeros((len(burns),2)),burns]
    pred=run_candidates(state,model,target,local,candidates,aim,now,
                        force_direction=aim,known_aligned=True)
    feasible=np.flatnonzero(pred['feasible'])
    zero_feasible=bool(pred['feasible'][0])
    if len(feasible):
        i=int(feasible[np.argmin(pred['cost'][feasible])])
        best={'additional_burn_s':float(burns[i]),
              'arrival_sim_s':float(now+pred['arrival'][i]),
              'predicted_total_fuel_kg':float(pred['fuel'][i]),
              'predicted_score':float(pred['score'][i]),
              'predicted_closest_m':float(pred['min_dist'][i]),
              'cost':float(pred['cost'][i])}
    else:
        best=None
    closest_index=int(np.argmin(pred['min_dist']))
    extra_fuel=float(burns[closest_index])*fuel_per_second(model,[cfg['burn']['request_x_m_s2'],0,0,0,0,0])
    best_miss={'estimated_extra_fuel_kg':extra_fuel,
               'improvement_per_kg':float((pred['min_dist'][0]-pred['min_dist'][closest_index])/max(extra_fuel,1e-6)),
               'additional_burn_s':float(burns[closest_index]),
               'predicted_closest_m':float(pred['min_dist'][closest_index]),
               'immediate_closest_m':float(pred['min_dist'][0]),
               'improvement_m':float(pred['min_dist'][0]-pred['min_dist'][closest_index])}
    if zero_feasible:
        immediate={'arrival_sim_s':float(now+pred['arrival'][0]),
                   'predicted_total_fuel_kg':float(pred['fuel'][0]),
                   'predicted_score':float(pred['score'][0]),
                   'predicted_closest_m':float(pred['min_dist'][0]),
                   'cost':float(pred['cost'][0])}
    else:
        immediate=None
    # Quadratic objective is not measured in score points.
    if immediate is not None and best is not None:
        if cfg['objective']['mode']=='score':
            margin=best['predicted_score']-immediate['predicted_score']
            stop=margin<=b['score_tolerance_points']
        else:
            margin=immediate['cost']-best['cost']
            stop=margin<=b['score_tolerance_points']
    else:
        stop=False; margin=None
    return {'recommend_stop':bool(stop),'coast_feasible_now':zero_feasible,
            'best':best,'best_miss':best_miss,'immediate_coast':immediate,
            'relative_cost_of_stop':float(margin) if margin is not None else None,
            'future_burn_window_s':float(limit),'candidate_count':int(len(burns))}
