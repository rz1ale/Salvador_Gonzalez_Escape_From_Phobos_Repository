"""Simulator entry logic. All SDK interaction goes through run(context) on its thread."""
from __future__ import annotations
import json
import math
import copy
from pathlib import Path
import numpy as np
from policy.bangbang_coast_v3.adaptive import (
    BurnAttitudeHold, AlignmentSwitchGuard, alignment_command,
    choose_attitude_profile, evaluate_burn_stop)
from policy.bangbang_coast_v3.coast_monitor import CoastMissMonitor, predict_ballistic
from policy.bangbang_coast_v3.recovery import search_body_correction
from policy.bangbang_coast_v3.joint import search_joint_burn
from policy.bangbang_coast_v3.core import (
    TargetMotion, cap, coast_trim, estimate_target_velocity, latent_from_reported,
    linearized_qr_gain, point_command, radar_lateral_body, select_plan, unit)


def load_settings(path):
    settings=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if settings.get('schema_version')!=3:
        raise ValueError('Unsupported V3 settings schema')
    if settings['objective']['mode'] not in ('score','quadratic'):
        raise ValueError('objective.mode must be score or quadratic')
    if settings['target']['mode'] not in ('fixed','linear','angular'):
        raise ValueError('target.mode invalid')
    if not 0<settings['burn']['request_x_m_s2']<=.5:
        raise ValueError('burn.request_x_m_s2 must be within official cap (0,0.5]')
    if not 0<settings['attitude']['max_request_rad_s2']<=.05:
        raise ValueError('attitude.max_request_rad_s2 must be within official cap (0,0.05]')
    if not 0<settings['planning']['arrival_radius_m']<50:
        raise ValueError('planning.arrival_radius_m must be inside official 50 m zone')
    if not 0<settings['planning']['burn_min_s']<settings['planning']['burn_max_s']:
        raise ValueError('Invalid burn search bounds')
    if settings['planning']['burn_samples']<4 or not 0<=settings['planning']['grid_refinement_levels']<=5:
        raise ValueError('Invalid planning sample/refinement settings')
    if not 0<settings['planning']['integration_step_s']<=5:
        raise ValueError('Invalid prediction integration step')
    if settings['control']['period_s']<=0 or settings['control']['radar_period_s']<=0:
        raise ValueError('Control/radar periods must be positive')
    if settings.get('coast', {}).get('mode') not in ('ballistic', 'corrected'):
        raise ValueError('coast.mode must be ballistic or corrected')
    monitor=settings['coast']['miss_monitor']
    if not isinstance(monitor['enabled'],bool):
        raise ValueError('coast.miss_monitor.enabled must be Boolean')
    if (not 0<monitor['trigger_closest_m']<50 or
        not 0<monitor['capture_radius_m']<=monitor['trigger_closest_m'] or
        not 0<monitor['check_period_s']<=30 or
        not 0<monitor['integration_step_s']<=5 or
        not 0<monitor['horizon_s']<=1000 or
        not 0<=monitor['entry_grace_s']<=60 or
        monitor['confirmations_required']<1 or
        monitor['max_reinterceptions']<0 or
        monitor['attempt_cooldown_s']<0):
        raise ValueError('Invalid ballistic coast miss monitor settings')
    econ_align=settings['attitude']['alignment_control']
    if not (0<econ_align['braking_safety_factor']<=1 and
            0<econ_align['minimum_braking_accel_rad_s2']<=.2 and
            0<econ_align['max_switch_accel_rad_s2']<=.2 and
            econ_align['quiet_surface_rad']>=0 and econ_align['quiet_rate_rad_s']>=0):
        raise ValueError('Invalid alignment controller settings')
    recovery=settings['coast']['direct_recovery']
    if not isinstance(recovery['enabled'],bool) or not isinstance(recovery['allow_longitudinal'],bool):
        raise ValueError('Invalid direct recovery toggles')
    if (not 0<recovery['arrival_radius_m']<settings['coast']['miss_monitor']['capture_radius_m'] or
        not recovery['arrival_radius_m']<=recovery['urgent_arrival_radius_m']<50 or
        not recovery['urgent_time_to_closest_s']>0 or
        not 0<recovery['integration_step_s']<=5 or
        not 0<recovery['min_time_to_closest_s'] or
        not 0<recovery['pulse_min_s']<recovery['pulse_max_s']<=15 or
        not recovery['request_magnitudes_m_s2'] or
        any(not 0<float(a)<=.5 for a in recovery['request_magnitudes_m_s2']) or
        recovery['max_detailed_candidates']<1 or recovery['proxy_max_m']<=0 or
        recovery['proxy_violation_cost_kg_per_m']<0 or
        not 0<recovery['max_fallback_turn_deg']<=180 or
        not -1000<=recovery['min_fallback_predicted_score']<=100):
        raise ValueError('Invalid direct recovery settings')
    avoidance=settings['avoidance']
    if (not 0<avoidance['lookahead_s']<=30 or
        not 0<avoidance['max_radar_age_s']<=10 or
        not 0<avoidance['clearance_padding_m']<=100 or
        not 0<=avoidance['radar_age_uncertainty_m_s']<=20 or
        not 0<avoidance['min_closing_speed_m_s']<=5 or
        not 0<=avoidance['max_lateral_request_m_s2']<=.5 or
        not 0<=avoidance['emergency_lateral_request_m_s2']<=.5 or
        not 0<avoidance['emergency_time_to_threat_s']<=30 or
        not 0<avoidance['emergency_surface_range_m']<=100):
        raise ValueError('Invalid radar avoidance settings')
    if (not 0<econ_align['minimum_reverse_dwell_s']<=2 or
        not -1<=econ_align['opposing_direction_cosine']<0):
        raise ValueError('Invalid alignment switching guard settings')
    if settings['safety']['mission_timeout_s']<=0:
        raise ValueError('Invalid mission timeout')
    joint=settings['burn']['joint_optimizer']
    if not isinstance(joint['enabled'],bool) or not isinstance(joint['include_diagonals'],bool):
        raise ValueError('Joint optimizer flags must be booleans')
    if (not 0<joint['check_period_s']<=30 or
        not 0<=joint['min_burn_elapsed_s']<=settings['planning']['burn_max_s'] or
        not 0<joint['future_burn_window_s']<=180 or
        not 3<=joint['burn_samples']<=40 or
        not 0<joint['short_burn_sample_s']<=joint['future_burn_window_s'] or
        not 0<joint['prediction_horizon_s']<=600 or
        not 0<joint['integration_step_s']<=5 or
        not 0<joint['capture_radius_m']<50 or
        not 0<=joint['stop_score_tolerance_points']<=10 or
        joint['stop_confirmations_required']<1 or
        joint['min_score_gain_for_lateral_points']<0 or
        not 0<joint['pulse_start_lead_s']<=5 or
        joint['max_lateral_fuel_kg']<0 or
        joint['max_lateral_episodes_per_burn']<0 or
        joint['min_time_between_lateral_episodes_s']<0 or
        joint['unreachable_fuel_weight_m_per_kg']<0 or
        not 0<joint['max_candidates']<=3000 or
        not 0<joint['coast_radar_log_period_s']<=60 or
        not isinstance(joint['coast_radar_observation'],bool) or
        not isinstance(joint['log_forecasts'],bool) or
        not joint['lateral_request_magnitudes_m_s2'] or
        not joint['lateral_pulse_durations_s'] or
        any(not 0<float(v)<=0.49 for v in joint['lateral_request_magnitudes_m_s2']) or
        any(not 0<float(v)<=10 for v in joint['lateral_pulse_durations_s'])):
        raise ValueError('Invalid V3 joint burn settings')
    if settings['objective']['quadratic']['Q_terminal_velocity']!=0:
        raise ValueError('V3 forbids terminal translational velocity penalties')
    econ=settings['attitude']['economy']
    if not isinstance(econ['enabled'],bool):
        raise ValueError('attitude.economy.enabled must be Boolean')
    if not econ['request_candidates_rad_s2'] or not all(
        0<float(x)<=settings['attitude']['max_request_rad_s2'] for x in econ['request_candidates_rad_s2']):
        raise ValueError('Invalid attitude economy request candidates')
    if not 0<econ['turn_duration_margin']<=3 or not 0<econ['minimum_effective_accel_rad_s2']<=.1:
        raise ValueError('Invalid attitude economy parameters')
    for key in ('time_cost_multiplier','fuel_cost_multiplier'):
        if econ[key]<0:raise ValueError('Invalid attitude economy cost multiplier')
    hold=settings['attitude']['burn_hold']
    if not 0<hold['release_error_deg']<hold['activate_error_deg']<=45:
        raise ValueError('Attitude hold error hysteresis invalid')
    if not 0<hold['release_rate_rad_s']<hold['activate_rate_rad_s']<=.2:
        raise ValueError('Attitude hold rate hysteresis invalid')
    if not 0<hold['max_request_rad_s2']<=.05 or hold['min_off_s']<0 or hold['predicted_drift_s']<0 or hold['max_burst_s']<=0 or hold['emergency_error_deg']<=hold['activate_error_deg'] or hold['emergency_rate_rad_s']<=hold['activate_rate_rad_s']:
        raise ValueError('Attitude hold parameters invalid')
    online=settings['burn']['online_stop']
    if online['enabled']:
        if not 0<online['check_period_s']<=30 or online['candidate_samples']<3:
            raise ValueError('Online burn-stop schedule invalid')
        if not 0<online['arrival_radius_m']<50 or online['arrival_radius_m']>settings['planning']['arrival_radius_m']:
            raise ValueError('Online arrival radius must be conservative and <50 m')
        if not 0<online['integration_step_s']<=5 or not 0<online['future_burn_window_s']<=180:
            raise ValueError('Online burn predictor settings invalid')
        if online['confirmations_required']<1 or online['max_extension_s']<0:
            raise ValueError('Online stop confirmation/extension invalid')
        if not 0<online['enforce_max_burn_s']<=settings['planning']['burn_max_s']:
            raise ValueError('Online hard burn maximum outside planning bound')
        if (online['score_tolerance_points']<0 or online['min_burn_elapsed_s']<0 or
            online['min_miss_improvement_m_per_kg']<0 or online['min_extension_predicted_s']<0):
            raise ValueError('Online score tolerance/min elapsed invalid')
    return settings


class Flight:
    def __init__(self,ctx,model,cfg):
        self.ctx=ctx;self.model=model;self.cfg=cfg
        self.history=[]
        self.initial_radius=None
        self.last_radar=-1e10
        self.last_rows=[]
        self.last_event=-1e10
        self.phase='startup'
        self.start_sim=ctx.clock()
        self.last_trim=-1e10
        self.trim_request=np.zeros(3)
        self.qr_gain=None
        self.reinterceptions=0
        self.last_recovery_attempt=-1e20

    def event(self, name, **data):
        self.ctx.event(name,phase=self.phase,**data)

    def snapshot(self):
        s=self.ctx.state()
        t=.5*(s['read_begin_sim_s']+s['read_end_sim_s'])
        self.history.append((t,np.asarray(s['target_P_m'],float)))
        # Local velocity estimate; never fetch hidden target velocity.
        self.history=[v for v in self.history if v[0]>=t-4.]
        self.history=self.history[-24:]
        return s

    def target(self,s):
        velocity=estimate_target_velocity(self.history,self.cfg['target']['max_estimated_speed_m_s'])
        return TargetMotion(s['target_P_m'],velocity,self.cfg['target']['mode'])

    def health(self,s):
        now=self.ctx.clock()
        if now-self.start_sim>self.cfg['safety']['mission_timeout_s']:
            self.ctx.finish(outcome='failure',reason='bangbang_mission_timeout',elapsed_sim_s=now-self.start_sim)
        if s['damage_J']>=self.cfg['safety']['max_damage_J']:
            self.ctx.finish(outcome='failure',reason='bangbang_damage_limit',damage_J=s['damage_J'])
        radius=np.linalg.norm(s['position_P_m'])
        if self.initial_radius is None:
            self.initial_radius=radius
        delta=radius-self.initial_radius
        radial_limit=self.cfg['safety']['radial_drop_abort_m']
        if radial_limit is not None and delta<radial_limit:
            self.ctx.finish(outcome='failure',reason='bangbang_radial_drop',drop_m=float(delta))

    def arrived(self,s):
        distance=float(np.linalg.norm(np.asarray(s['target_P_m'])-np.asarray(s['position_P_m'])))
        if distance<=self.cfg['safety']['arrival_observation_radius_m']:
            self.event('arrival_observed',sim_s=self.ctx.clock(),distance_m=distance,fuel_used_kg=s['fuel_used_kg'],damage_J=s['damage_J'])
            self.ctx.finish(outcome='success',reason='bangbang_observed_arrival',distance_m=distance,
                            fuel_used_kg=s['fuel_used_kg'],damage_J=s['damage_J'],
                            sim_s=self.ctx.clock(),note='Native scorer remains authoritative')
        return distance

    def radar(self,now):
        if now-self.last_radar>=self.cfg['control']['radar_period_s']:
            self.last_rows=self.ctx.radar(self.cfg['avoidance']['radar_count'])
            self.last_radar=now
        return self.last_rows

    def avoidance(self,s,now):
        from system_identification.model import matrix
        C=matrix(np.asarray(s['q_PB_xyzw_reported']))
        return radar_lateral_body(self.radar(now),C,now,self.cfg,
                                  velocity_P=s['velocity_P_m_s'])

    def log_periodic(self,s, distance, avoidance, angle_deg=0, **other):
        now=self.ctx.clock()
        if now-self.last_event<self.cfg['control']['event_period_s']:
            return
        self.last_event=now
        self.event('status',sim_s=now,distance_m=float(distance),fuel_kg=s['fuel_used_kg'],
                   damage_J=s['damage_J'],avoidance=avoidance,pointing_error_deg=float(angle_deg),**other)

    def tick(self):
        self.ctx.sleep(self.cfg['control']['period_s'])

    def launch_boost(self):
        if not self.cfg['launch']['enabled']:
            return
        self.phase='launch_boost'
        duration=self.cfg['launch']['duration_s']
        req=np.asarray(self.cfg['launch']['body_request_m_s2'],float)
        req=cap(req,.5)
        t0=self.ctx.clock()
        self.event('launch_boost_started',duration_s=duration,request_B=req.tolist())
        while self.ctx.clock()-t0<duration:
            s=self.snapshot();self.health(s)
            self.ctx.command(req,np.zeros(3))
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        self.event('launch_boost_ended',sim_s=self.ctx.clock())

    def collect_target(self):
        self.phase='target_observation'
        self.ctx.command(np.zeros(3),np.zeros(3))
        s=self.snapshot();self.health(s)
        start=self.ctx.clock()
        while self.ctx.clock()-start<self.cfg['target']['observation_span_s']:
            self.tick()
            s=self.snapshot();self.health(s)
        return s

    def align(self,aim_P,profile=None):
        self.phase='bangbang_align'
        t0=self.ctx.clock();stable=0
        first=self.snapshot()
        fuel0=float(first['fuel_used_kg'])
        # Use one cost-selected angular amplitude, with feedback determining switch times.
        # Do not use predicted alignment duration as an open-loop command schedule.
        cfg_align=copy.deepcopy(self.cfg)
        if profile and self.cfg['attitude']['economy']['enabled']:
            cfg_align['attitude']['max_request_rad_s2']=profile['request_rad_s2']
            cfg_align['attitude']['estimated_effective_ang_accel_rad_s2']=profile['effective_alpha_rad_s2']
        switch_guard=AlignmentSwitchGuard(cfg_align)
        self.event('alignment_started',sim_s=t0,angular_request_cap_rad_s2=cfg_align['attitude']['max_request_rad_s2'],
                   estimated_angular_authority_rad_s2=cfg_align['attitude']['estimated_effective_ang_accel_rad_s2'])
        while self.ctx.clock()-t0<self.cfg['attitude']['max_align_s']:
            s=self.snapshot();self.health(s);distance=self.arrived(s)
            angular,error,rate,settled=alignment_command(s,aim_P,self.model,cfg_align)
            angular=switch_guard.filter(angular,self.ctx.clock(),settled)
            lateral,info=self.avoidance(s,self.ctx.clock())
            self.ctx.command(lateral,angular)
            self.log_periodic(s,distance,info,angle_deg=math.degrees(error),angular_rate_rad_s=rate)
            stable=stable+1 if settled else 0
            if stable>=self.cfg['attitude']['stable_cycles']:
                self.ctx.command(np.zeros(3),np.zeros(3))
                final=self.snapshot()
                self.event('alignment_complete',elapsed_s=self.ctx.clock()-t0,
                           angle_deg=math.degrees(error),rate_rad_s=rate,
                           fuel_delta_kg=float(final['fuel_used_kg'])-fuel0,
                           inhibited_reversals=switch_guard.inhibited,
                           accepted_reversals=switch_guard.opposed_accepted)
                return final
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        self.ctx.finish(outcome='failure',reason='bangbang_alignment_timeout',
                        align_elapsed_s=self.ctx.clock()-t0)

    def burn(self,aim_P,burn_s):
        """V3 online +X/off optimization with opportunistic short body Y/Z correction.

        All choices use the current public state and a free terminal point on the
        capture boundary. No terminal velocity tracking/braking is requested.
        """
        self.phase='burn_x'
        t0=self.ctx.clock()
        cfg=self.cfg
        online=cfg['burn']['online_stop']
        jc=cfg['burn']['joint_optimizer']
        hard_end=(min(float(cfg['planning']['burn_max_s']),
                      float(online['enforce_max_burn_s']),
                      float(burn_s)+float(online['max_extension_s']))
                  if online['enabled'] else float(burn_s))
        hold=BurnAttitudeHold(cfg)
        last_eval=-1e20
        confirmations=0
        n_checks=0
        fuel0=float(self.snapshot()['fuel_used_kg'])
        reason='planned_duration'
        last_joint=None
        last_legacy=None
        last_prediction_at=-1e20
        pulse_until=-1e20
        last_pulse=-1e20
        lateral_active=np.zeros(3)
        pulse_episodes=0
        self.event('burn_started',sim_s=t0,duration_s=burn_s,
                   max_duration_s=hard_end,request_x=cfg['burn']['request_x_m_s2'],
                   joint_optimizer=bool(jc['enabled']),online_burn_stop=bool(online['enabled']))
        while True:
            s=self.snapshot();self.health(s);distance=self.arrived(s)
            now=self.ctx.clock();elapsed=now-t0
            if elapsed>=hard_end:
                reason='hard_stop'
                break
            lateral_avoid,info=self.avoidance(s,now)
            hazards=bool(info.get('hazards',0))
            if hazards and now<pulse_until:
                pulse_until=-1e20
                lateral_active[:]=0
                self.event('joint_lateral_cancelled_for_radar',sim_s=now,avoidance=info)
            pulse_running=now<pulse_until
            if (online['enabled'] and elapsed>=max(online['min_burn_elapsed_s'],
                                                  jc['min_burn_elapsed_s'] if jc['enabled'] else 0.) and
                    now-last_eval >= (jc['check_period_s'] if jc['enabled'] else online['check_period_s']) and
                    not pulse_running):
                last_eval=now
                last_prediction_at=now
                n_checks+=1
                try:
                    if jc['enabled']:
                        # Include current absolute mission time for remaining horizon.
                        local=dict(cfg,_start_sim=self.start_sim)
                        last_joint=search_joint_burn(s,self.model,self.target(s),local,now,
                                                     max(0.,hard_end-elapsed))
                        best=last_joint['best'] if last_joint is not None else None
                        immediate=last_joint['immediate_coast'] if last_joint is not None else None
                        pure=last_joint['best_pure'] if last_joint is not None else None
                        if (immediate is not None and immediate['feasible'] and
                                best is not None and best['feasible']):
                            benefit=(float(best['predicted_score'])-float(immediate['predicted_score'])
                                     if cfg['objective']['mode']=='score' else
                                     float(immediate['objective'])-float(best['objective']))
                            confirmations=(confirmations+1 if benefit<=jc['stop_score_tolerance_points']
                                           and not hazards else 0)
                        else:
                            confirmations=0
                        if jc['log_forecasts']:
                            self.event('joint_burn_forecast',sim_s=now,burn_elapsed_s=elapsed,
                                       confirmations=confirmations,forecast=last_joint)
                        if confirmations>=jc['stop_confirmations_required']:
                            reason='online_joint_score_stop'
                            break
                        # A lateral correction is considered ONLY when it can start
                        # now, the predicted full schedule reaches the capture zone,
                        # and its fuel/score tradeoff merits intervention.
                        if (best is not None and best['feasible'] and
                                best['lateral_pulse_s']>0 and
                                best['lateral_start_s']<=jc['pulse_start_lead_s'] and
                                not hazards and pulse_episodes<jc['max_lateral_episodes_per_burn'] and
                                now-last_pulse>=jc['min_time_between_lateral_episodes_s']):
                            lateral=np.asarray(best['lateral_request_B'],float)
                            strength=float(np.linalg.norm(lateral))
                            x_req=min(cfg['burn']['request_x_m_s2'],
                                      math.sqrt(max(0.,.5**2-strength**2)))
                            full=np.r_[x_req,lateral[1:],0.,0.,0.]
                            from policy.bangbang_coast_v3.core import fuel_per_second
                            pulse_fuel=(fuel_per_second(self.model,full)*
                                        float(best['lateral_pulse_s']))
                            if pure is None:
                                gain=float('inf')
                            else:
                                gain=(float(best['predicted_score'])-float(pure['predicted_score'])
                                      if cfg['objective']['mode']=='score' else
                                      float(pure['objective'])-float(best['objective']))
                            if (pulse_fuel<=jc['max_lateral_fuel_kg'] and
                                    gain>=jc['min_score_gain_for_lateral_points']):
                                lateral_active=lateral
                                pulse_until=now+min(best['lateral_pulse_s'],hard_end-elapsed)
                                last_pulse=now
                                pulse_episodes+=1
                                self.event('joint_lateral_started',sim_s=now,
                                           predicted_gain=gain,estimated_fuel_kg=pulse_fuel,
                                           pulse_number=pulse_episodes,plan=best)
                    else:
                        last_legacy=evaluate_burn_stop(s,self.model,self.target(s),cfg,now,
                                            max(0.,hard_end-elapsed))
                        if last_legacy['recommend_stop'] and not hazards:
                            confirmations+=1
                        else:
                            confirmations=0
                        if online['log_checks']:
                            self.event('online_burn_forecast',sim_s=now,burn_elapsed_s=elapsed,
                                       confirmations=confirmations,forecast=last_legacy)
                        if confirmations>=online['confirmations_required']:
                            reason='online_score_stop'
                            break
                except (ValueError,RuntimeError,ArithmeticError) as exc:
                    last_joint=None
                    last_legacy=None
                    confirmations=0
                    self.event('online_burn_forecast_failed',sim_s=now,error=str(exc))
            # Avoid unbounded extension merely because a coast forecast misses.
            if elapsed>=burn_s and now>=pulse_until:
                fresh=now-last_prediction_at<=2.5*(jc['check_period_s'] if jc['enabled']
                                                     else online['check_period_s'])
                if jc['enabled']:
                    best=last_joint['best'] if last_joint is not None and fresh else None
                    worthwhile=(best is not None and best['feasible'] and
                        best['future_burn_s']>=online['min_extension_predicted_s'] and
                        best['predicted_capture_sim_s'] is not None and
                        best['predicted_capture_sim_s']>now)
                else:
                    best=last_legacy.get('best') if last_legacy is not None and fresh else None
                    worthwhile=(best is not None and
                        best['additional_burn_s']>=online['min_extension_predicted_s'])
                if not worthwhile or hazards or not online['enabled']:
                    reason='no_feasible_extension' if not worthwhile else 'planned_duration'
                    break
            ang,hold_info=hold.command(s,aim_P,self.model,cfg,now)
            # Emergency radar correction always takes priority over economic trim.
            lateral=lateral_avoid if hazards else (lateral_active if now<pulse_until
                                                     else np.zeros(3))
            if now>=pulse_until and np.linalg.norm(lateral_active)>0:
                self.event('joint_lateral_completed',sim_s=now,pulse_number=pulse_episodes)
                lateral_active[:]=0
            strength=float(np.linalg.norm(lateral[1:]))
            x_req=min(cfg['burn']['request_x_m_s2'],math.sqrt(max(0.,.5**2-strength**2)))
            u=np.r_[x_req,lateral[1:]]
            self.ctx.command(u,ang)
            self.log_periodic(s,distance,info,angle_deg=hold_info['error_deg'],
                              x_request=x_req,hold=hold_info,
                              joint_lateral_active=bool(now<pulse_until),
                              online_stop_confirmations=confirmations)
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        final=self.snapshot()
        self.event('burn_ended',elapsed_s=self.ctx.clock()-t0,planned_duration_s=burn_s,
                   reason=reason,fuel_delta_kg=float(final['fuel_used_kg'])-fuel0,
                   hold_engagements=hold.engagements,hold_max_error_deg=hold.max_error_deg,
                   hold_active_fraction=hold.active_updates/max(1,hold.total_updates),
                   online_forecast_checks=n_checks,lateral_episodes=pulse_episodes,
                   final_joint_forecast=last_joint)

    def body_correction(self, action):
        """One bounded open-loop pulse, followed by zero-thrust coast and reassessment."""
        self.phase='body_correction'
        req=np.asarray(action['request_B'],float)
        duration=float(action['duration_s'])
        if not np.isfinite(req).all() or np.linalg.norm(req)>.5+1e-8 or duration<=0:
            raise ValueError('Invalid body-frame recovery pulse')
        t0=self.ctx.clock();f0=float(self.snapshot()['fuel_used_kg'])
        self.event('body_correction_started',sim_s=t0,plan=action)
        while self.ctx.clock()-t0<duration:
            s=self.snapshot();self.health(s);self.arrived(s)
            self.ctx.command(req,np.zeros(3))
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        final=self.snapshot()
        self.event('body_correction_completed',sim_s=self.ctx.clock(),
                   duration_s=self.ctx.clock()-t0,
                   fuel_delta_kg=float(final['fuel_used_kg'])-f0)

    def coast(self,aim_P):
        """Zero-thrust coast. Returns a feasible recovery plan only on confirmed miss."""
        self.phase='coast'
        coast_mode=self.cfg['coast']['mode']
        zero=np.zeros(3)
        self.ctx.command(zero,zero)
        first=self.snapshot()
        fuel_at_entry=float(first['fuel_used_kg'])
        entry_time=self.ctx.clock()
        monitor_cfg=self.cfg['coast']['miss_monitor']
        monitor=CoastMissMonitor(monitor_cfg)
        last_check=-1e20
        budget_logged=False
        coast_last_radar=-1e20
        self.event('coast_started',sim_s=entry_time,mode=coast_mode,
                   fuel_kg=fuel_at_entry,miss_monitor=monitor_cfg['enabled'],
                   reinterceptions_used=self.reinterceptions)

        while True:
            s=self.snapshot();self.health(s);distance=self.arrived(s)
            now=self.ctx.clock()
            # Always keep the body-linear and body-angular commands at ZERO in
            # ballistic mode; public state reads and forecasts do not fire jets.
            if coast_mode=='ballistic':
                self.ctx.command(zero,zero)
            else:
                ang,error,rate,_=point_command(s['q_PB_xyzw_reported'],s['omega_relP_P_rad_s'],
                                               aim_P,self.model,self.cfg)
                lateral,info=self.avoidance(s,now)
                target=self.target(s)
                if now-self.last_trim>=self.cfg['feedback']['trim_period_s']:
                    self.trim_request,trim_info=coast_trim(s,self.model,target,self.cfg,now,aim_P,self.qr_gain)
                    self.last_trim=now
                    self.event('trim_update',sim_s=now,request_B=self.trim_request.tolist(),detail=trim_info)
                self.ctx.command(cap(self.trim_request+lateral,.5),ang)

            # Radar observation during ballistic coast is deliberately PASSIVE.
            # It flags possible risk for analysis but NEVER commands thrusters.
            joint=self.cfg['burn']['joint_optimizer']
            if (coast_mode=='ballistic' and joint['coast_radar_observation'] and
                    now-coast_last_radar>=joint['coast_radar_log_period_s']):
                coast_last_radar=now
                _,risk=self.avoidance(s,now)
                if risk.get('hazards',0):
                    self.event('coast_radar_risk_observed',sim_s=now,risk=risk,
                               note='Monitoring only; no flight command or collision guarantee')
            if (coast_mode=='ballistic' and monitor_cfg['enabled'] and
                    now-entry_time>=monitor_cfg['entry_grace_s'] and
                    now-last_check>=monitor_cfg['check_period_s']):
                last_check=now
                remaining=self.cfg['safety']['mission_timeout_s']-(now-self.start_sim)
                horizon=min(monitor_cfg['horizon_s'],max(0.,remaining))
                if horizon>1.:
                    try:
                        forecast=predict_ballistic(s,self.model,self.target(s),horizon,
                            monitor_cfg['integration_step_s'],monitor_cfg['capture_radius_m'])
                        status=monitor.update(forecast,
                            horizon_covers_deadline=(horizon>=remaining-1e-5))
                        self.event('coast_intercept_forecast',sim_s=now,forecast=forecast,
                                   monitor=status,reinterceptions_used=self.reinterceptions)
                        if status['trigger_replan']:
                            if self.reinterceptions>=monitor_cfg['max_reinterceptions']:
                                if not budget_logged:
                                    self.event('coast_recovery_budget_exhausted',sim_s=now,forecast=forecast)
                                    budget_logged=True
                            elif now-self.last_recovery_attempt>=monitor_cfg['attempt_cooldown_s']:
                                self.last_recovery_attempt=now
                                # FIRST try a low-cost body-frame translational correction.
                                # Full reorientation is a fallback, never the default.
                                try:
                                    local_cfg=dict(self.cfg,_start_sim=self.start_sim)
                                    direct=search_body_correction(s,self.model,self.target(s),local_cfg,now)
                                    if direct is not None:
                                        self.event('coast_direct_correction_selected',sim_s=now,
                                                   miss_forecast=forecast,proposed_action=direct,
                                                   recovery_number=self.reinterceptions+1)
                                        self.ctx.command(zero,zero)
                                        return direct
                                    self.event('coast_no_feasible_body_correction',sim_s=now,
                                               miss_forecast=forecast)
                                except (ValueError,RuntimeError,ArithmeticError) as exc:
                                    self.event('coast_direct_correction_failed',sim_s=now,error=str(exc))
                                try:
                                    fresh=select_plan(s,self.model,self.target(s),self.cfg,now)
                                    from system_identification.model import matrix
                                    forward=matrix(np.asarray(s['q_PB_xyzw_reported'],float)).T@np.array([1.,0.,0.])
                                    angle_deg=math.degrees(math.acos(float(np.clip(
                                        forward@unit(fresh['aim_P']),-1.,1.))))
                                    recovery_cfg=self.cfg['coast']['direct_recovery']
                                    allowed=(fresh['feasible'] and
                                             angle_deg<=recovery_cfg['max_fallback_turn_deg'] and
                                             fresh['estimated_score'] is not None and
                                             fresh['estimated_score']>=recovery_cfg['min_fallback_predicted_score'])
                                    if allowed:
                                        self.event('coast_reinterception_trigger',sim_s=now,
                                            miss_forecast=forecast,proposed_plan=fresh,
                                            angle_deg=angle_deg,
                                            reinterception_number=self.reinterceptions+1)
                                        self.ctx.command(zero,zero)
                                        return dict(kind='turn_reintercept',plan=fresh)
                                    self.event('coast_reinterception_rejected',sim_s=now,
                                        forecast=forecast,planner_result=fresh,turn_deg=angle_deg)
                                except (ValueError,RuntimeError,ArithmeticError) as exc:
                                    self.event('coast_reinterception_plan_failed',sim_s=now,error=str(exc))
                    except (ValueError,RuntimeError,ArithmeticError) as exc:
                        monitor.consecutive=0
                        self.event('coast_prediction_failed',sim_s=now,error=str(exc))
            self.log_periodic(s,distance,{'mode':coast_mode},
                coast_mode=coast_mode,coast_fuel_delta_kg=float(s['fuel_used_kg'])-fuel_at_entry)
            self.tick()


    def execute_interception(self,s,plan=None,recovery=False):
        """Run one planned alignment / +X burn; callable after coast recovery."""
        self.phase='recovery_planning' if recovery else 'planning'
        if plan is None:
            plan=select_plan(s,self.model,self.target(s),self.cfg,self.ctx.clock())
        self.event('reinterception_plan' if recovery else 'nominal_plan',**plan)
        if not plan['feasible'] and self.cfg['planning']['require_feasible_plan']:
            self.ctx.command(np.zeros(3),np.zeros(3))
            self.ctx.finish(outcome='failure',reason='bangbang_no_feasible_nominal_arrival',
                            predicted_closest_m=plan['predicted_closest_m'])
        aim=np.asarray(plan['aim_P'],float)
        profile=None
        if self.cfg['attitude']['economy']['enabled']:
            profile=choose_attitude_profile(s,aim,self.model,self.cfg,plan['estimated_arrival_sim_s'])
            self.event('attitude_economy_profile',profile=profile)
        self.align(aim,profile)
        if self.cfg['planning']['refit_after_alignment']:
            self.phase='refit_after_alignment'
            s=self.snapshot();self.health(s)
            from system_identification.model import matrix
            C=matrix(np.asarray(s['q_PB_xyzw_reported'],float))
            x_direction=C.T@np.array([1.,0.,0.])
            refined=select_plan(s,self.model,self.target(s),self.cfg,self.ctx.clock(),
                force_direction=x_direction,known_aligned=True)
            self.event('post_alignment_refit',**refined)
            if refined['feasible']:
                plan=refined
            elif self.cfg['planning']['require_feasible_post_align_refit']:
                self.ctx.command(np.zeros(3),np.zeros(3))
                self.ctx.finish(outcome='failure',reason='bangbang_post_align_infeasible',
                    predicted_closest_m=refined['predicted_closest_m'])
        if self.cfg['objective']['mode']=='quadratic' and self.cfg['feedback']['qr_feedback_enabled']:
            s=self.snapshot()
            p,_,_,_,_=latent_from_reported(s,self.model)
            self.qr_gain=linearized_qr_gain(p,aim,self.model,self.cfg)
            self.event('qr_gain_scheduled',K=self.qr_gain.tolist(),
                       note='6-state local LQR about pointing; allocator applied afterward')
        self.burn(aim,plan['burn_s'])
        return aim


def run(context):
    from system_identification.model import Model
    cfg=load_settings(context.root/'policy'/'bangbang_coast_v3'/'settings.json')
    if context.model_path is None:
        raise ValueError('Select root nominal_model.json using --config/--model')
    model=Model.load(context.model_path)
    if context.parameters:
        raise ValueError('This policy is configured in settings.json; per-trial overrides unsupported')
    (context.output_dir/'settings_used.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    flight=Flight(context,model,cfg)
    flight.event('policy_started',mode=cfg['objective']['mode'],model_path=str(context.model_path),
        model_status=model.definition.get('status','unknown'),
        note='Fitted local model; no native score or safety guarantee')
    flight.launch_boost()
    s=flight.collect_target()
    pending_action=None
    aim=None
    while True:
        if pending_action is not None:
            flight.reinterceptions+=1
            flight.event('recovery_started',sim_s=context.clock(),
                         recovery_number=flight.reinterceptions,kind=pending_action['kind'])
            if pending_action['kind']=='body_correction':
                flight.body_correction(pending_action)
                pending_action=flight.coast(aim)
                continue
            if pending_action['kind']!='turn_reintercept':
                raise ValueError('Unsupported recovery action')
            s=flight.snapshot()
            plan=pending_action['plan']
            aim=flight.execute_interception(s,plan=plan,recovery=True)
        else:
            aim=flight.execute_interception(s)
        pending_action=flight.coast(aim)
