"""V4 configurable thrust-axis / reorientation startup.

Separate install from V2/V3. All simulator interaction is via public context.
V3-derived coast / recovery methods are reused through legacy_flight in the
V4 namespace; the nominal V4 startup and main burn use this module only.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import numpy as np
from system_identification.model import matrix, Model
from policy.bangbang_coast_v4.core import cap, unit
from policy.bangbang_coast_v4.adaptive import (alignment_command, choose_attitude_profile,
                                                    AlignmentSwitchGuard, BurnAttitudeHold)
from policy.bangbang_coast_v4.legacy_flight import Flight as BaseFlight, load_settings as load_base
from policy.bangbang_coast_v4.startup import check_startup, select_startup, evaluate_remaining


def load_settings(path):
    cfg=load_base(path)  # V4-local copy of shared V3 safety/runner validation.
    if cfg.get('schema_version')!=4:
        raise ValueError('V4 requires settings schema_version 4')
    check_startup(cfg)
    if cfg['coast']['mode']!='ballistic':
        raise ValueError('V4 supports only ballistic coast; legacy corrected-coast geometry assumes body +X')
    if cfg['burn']['joint_optimizer']['enabled']:
        raise ValueError('V4 requires burn.joint_optimizer.enabled=false: V3 joint planner assumes body +X')
    return cfg


class Flight(BaseFlight):
    def __init__(self,ctx,model,cfg):
        super().__init__(ctx,model,cfg)
        self.thrust_axis_B=np.array([1.,0.,0.])

    def avoidance(self,s,now):
        from policy.bangbang_coast_v4.core import radar_lateral_body
        C=matrix(np.asarray(s['q_PB_xyzw_reported'],float))
        return radar_lateral_body(self.radar(now),C,now,self.cfg,
                                  velocity_P=s['velocity_P_m_s'],
                                  thrust_axis_B=self.thrust_axis_B)

    def align_axis(self,aim_P,body_B,turn_fraction=0.,combined=False,profile=None):
        self.thrust_axis_B=unit(body_B)
        self.phase='v4_combined_turn' if combined else 'v4_alignment'
        now=self.ctx.clock();f0=float(self.snapshot()['fuel_used_kg'])
        cfg_align=copy.deepcopy(self.cfg)
        if profile and self.cfg['attitude']['economy']['enabled']:
            cfg_align['attitude']['max_request_rad_s2']=float(profile['request_rad_s2'])
            cfg_align['attitude']['estimated_effective_ang_accel_rad_s2']=float(profile['effective_alpha_rad_s2'])
        guard=AlignmentSwitchGuard(cfg_align)
        stable=0
        thrust=(float(self.cfg['startup']['turn_translation_fraction'])*
                float(self.cfg['burn']['request_x_m_s2'])*self.thrust_axis_B if combined else np.zeros(3))
        self.event('v4_alignment_started',sim_s=now,body_axis_B=self.thrust_axis_B.tolist(),
                   desired_aim_P=unit(aim_P).tolist(),combined=combined,
                   translational_request_B=thrust.tolist(),requested_fraction=turn_fraction)
        while self.ctx.clock()-now < self.cfg['attitude']['max_align_s']:
            s=self.snapshot();self.health(s);distance=self.arrived(s)
            alpha,err,rate,settled=alignment_command(s,aim_P,self.model,cfg_align,self.thrust_axis_B)
            alpha=guard.filter(alpha,self.ctx.clock(),settled)
            dodge,info=self.avoidance(s,self.ctx.clock())
            self.ctx.command(cap(thrust+dodge,.5),cap(alpha,.05))
            self.log_periodic(s,distance,info,angle_deg=math.degrees(err),
                              angular_rate_rad_s=rate,v4_mode='combined' if combined else 'turn')
            stable=stable+1 if settled else 0
            if stable>=self.cfg['attitude']['stable_cycles']:
                self.ctx.command(np.zeros(3),np.zeros(3))
                final=self.snapshot()
                self.event('v4_alignment_completed',elapsed_s=self.ctx.clock()-now,
                           angle_deg=math.degrees(err),rate_rad_s=rate,
                           fuel_delta_kg=float(final['fuel_used_kg'])-f0,
                           inhibited_reversals=guard.inhibited,
                           accepted_reversals=guard.opposed_accepted,
                           combined=combined)
                return final
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        self.ctx.finish(outcome='failure',reason='v4_alignment_timeout',
                        align_elapsed_s=self.ctx.clock()-now,combined=combined)

    def burn_axis(self,aim_P,body_B,burn_duration_s):
        self.thrust_axis_B=unit(body_B)
        self.phase='v4_axis_burn'
        now=self.ctx.clock();t0=now
        f0=float(self.snapshot()['fuel_used_kg'])
        cfg=self.cfg;online=cfg['startup']['online_burn']
        planned=float(burn_duration_s)
        hard_end=min(online['max_total_burn_s'], planned+online['max_extension_s']) if online['enabled'] else planned
        hold=BurnAttitudeHold(cfg)
        confirmations=0;last_check=-1e20;checks=0;last_forecast=None
        reason='planned_duration'
        req=float(cfg['burn']['request_x_m_s2'])*self.thrust_axis_B
        self.event('v4_burn_started',sim_s=now,body_axis_B=self.thrust_axis_B.tolist(),
                   aim_P=unit(aim_P).tolist(),request_B=req.tolist(),planned_duration_s=planned,
                   hard_stop_s=hard_end,online_enabled=bool(online['enabled']))
        while True:
            s=self.snapshot();self.health(s);dist=self.arrived(s)
            now=self.ctx.clock();elapsed=now-t0
            if elapsed>=hard_end:
                reason='hard_stop';break
            dodge,risk=self.avoidance(s,now)
            hazards=bool(risk.get('hazards',0))
            if (online['enabled'] and elapsed>=online['min_elapsed_s'] and
                    now-last_check>=online['check_period_s']):
                last_check=now;checks+=1
                try:
                    local=dict(cfg,_start_sim=self.start_sim)
                    last_forecast=evaluate_remaining(s,self.model,self.target(s),local,now,
                                  self.thrust_axis_B,aim_P,max(0.,hard_end-elapsed))
                    zero=last_forecast['immediate_coast'];best=last_forecast['best']
                    if zero['feasible'] and best['feasible'] and not hazards:
                        if cfg['objective']['mode']=='score':
                            benefit=best['predicted_score']-zero['predicted_score']
                        else:
                            # No objective derived in forecast quadratic summary; require
                            # immediate coast to equal best zero-duration schedule.
                            benefit=0. if best['future_burn_s']<1e-8 else float('inf')
                        confirmations=confirmations+1 if benefit<=online['score_tolerance_points'] else 0
                    else: confirmations=0
                    self.event('v4_burn_forecast',sim_s=now,elapsed_s=elapsed,
                               confirmations=confirmations,forecast=last_forecast)
                    if confirmations>=online['stop_confirmations']:
                        reason='score_stop';break
                except (ArithmeticError,RuntimeError,ValueError) as exc:
                    confirmations=0;last_forecast=None
                    self.event('v4_burn_forecast_failed',sim_s=now,error=str(exc))
            if elapsed>=planned:
                best=last_forecast['best'] if last_forecast is not None else None
                if (not online['enabled'] or hazards or best is None or
                    not best['feasible'] or best['future_burn_s']<.8 or
                    now-last_check>2.5*online['check_period_s']):
                    reason='planned_duration';break
            ang,hold_info=hold.command(s,aim_P,self.model,cfg,now,self.thrust_axis_B)
            translation=cap(req+dodge,.5)
            self.ctx.command(translation,cap(ang,.05))
            self.log_periodic(s,dist,risk,angle_deg=hold_info['error_deg'],
                              v4_axis_B=self.thrust_axis_B.tolist(),
                              hold=hold_info,online_stop_confirmations=confirmations)
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        final=self.snapshot()
        self.event('v4_burn_ended',elapsed_s=self.ctx.clock()-t0,
                   planned_duration_s=planned,reason=reason,
                   fuel_delta_kg=float(final['fuel_used_kg'])-f0,
                   online_checks=checks,hold_engagements=hold.engagements,
                   hold_max_error_deg=hold.max_error_deg,
                   final_forecast=last_forecast)

    def execute_interception(self,s,plan=None,recovery=False):
        self.phase='v4_recovery_plan' if recovery else 'v4_startup_plan'
        if recovery:
            # Existing coast recovery planner supplies a +X interception. Keep
            # this as a documented *recovery-only* fallback, never silently use
            # +X for nominal no-turn / +Z runs.
            from policy.bangbang_coast_v4.core import select_plan
            if plan is None:plan=select_plan(s,self.model,self.target(s),self.cfg,self.ctx.clock())
            aim=unit(plan['aim_P']);axis=np.array([1.,0.,0.])
            self.event('v4_recovery_plus_x_fallback',proposed_plan=plan)
            self.align_axis(aim,axis,turn_fraction=1.,combined=False)
            self.burn_axis(aim,axis,float(plan['burn_s']))
            return aim
        local=dict(self.cfg,_start_sim=self.start_sim)
        plan=select_startup(s,self.model,self.target(s),local,self.ctx.clock())
        self.event('v4_strategy_selected',plan=plan)
        if not plan['predicted_feasible'] and self.cfg['startup']['required_prediction_feasible']:
            self.ctx.command(np.zeros(3),np.zeros(3))
            self.ctx.finish(outcome='failure',reason='v4_nominal_startup_infeasible',
                            predicted_closest_m=plan['predicted_closest_m'])
        aim=unit(plan['aim_P']);axis=unit(plan['body_direction_B'])
        self.thrust_axis_B=axis
        if plan['turn_angle_deg']>self.cfg['attitude']['angle_tolerance_deg']:
            profile=None
            if self.cfg['attitude']['economy']['enabled']:
                profile=choose_attitude_profile(s,aim,self.model,self.cfg,
                     plan['predicted_arrival_sim_s'],axis)
                self.event('v4_attitude_profile',profile=profile)
            self.align_axis(aim,axis,plan['turn_fraction'],plan['overlap_translation'],profile)
        else:
            self.event('v4_alignment_skipped',mode=plan['mode'],body_axis_B=axis.tolist(),
                       turn_angle_deg=plan['turn_angle_deg'])
        # Actual position and velocity after the alignment are used to revise
        # the burn duration, without constraining final capture position/velocity.
        s=self.snapshot();self.health(s)
        from policy.bangbang_coast_v4.startup import predict
        d=self.cfg['startup']
        burns=np.linspace(d['candidate_burn_min_s'],d['candidate_burn_max_s'],
                          int(d['candidate_burn_samples']))
        try:
            measured_direction=matrix(np.asarray(s['q_PB_xyzw_reported'],float)).T@axis
            revised=predict(s,self.model,self.target(s),local,self.ctx.clock(),axis,measured_direction,burns)
            valid=np.flatnonzero(revised['feasible'])
            if len(valid):
                i=int(valid[np.argmin(revised['cost'][valid])]);burn_s=float(burns[i])
                self.event('v4_post_turn_refit',feasible=True,burn_s=burn_s,
                           predicted_score=float(revised['score'][i]),
                           predicted_closest_m=float(revised['closest_m'][i]))
            else:
                burn_s=float(plan['planned_burn_s'])
                self.event('v4_post_turn_refit',feasible=False,
                           fallback_burn_s=burn_s,closest_m=float(min(revised['closest_m'])))
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            burn_s=float(plan['planned_burn_s'])
            self.event('v4_post_turn_refit_failed',error=str(exc),fallback_burn_s=burn_s)
        self.burn_axis(aim,axis,burn_s)
        return aim


def run(context):
    cfg=load_settings(context.root/'policy'/'bangbang_coast_v4'/'settings.json')
    if context.model_path is None:raise ValueError('V4 needs root nominal_model.json')
    if context.parameters:raise ValueError('V4 is configured only in settings.json')
    model=Model.load(context.model_path)
    (context.output_dir/'settings_used.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    flight=Flight(context,model,cfg)
    flight.event('v4_policy_started',strategy=cfg['startup']['mode'],
                 model_path=str(context.model_path),
                 warning='Model-based predictions have large demonstrated long-horizon position error; no safety guarantee')
    flight.launch_boost()
    s=flight.collect_target()
    pending=None;aim=None
    while True:
        if pending is not None:
            flight.reinterceptions+=1
            flight.event('v4_recovery_started',sim_s=context.clock(),
                         kind=pending['kind'],number=flight.reinterceptions)
            if pending['kind']=='body_correction':
                flight.body_correction(pending)
                pending=flight.coast(aim)
                continue
            if pending['kind']!='turn_reintercept':raise ValueError('Unsupported recovery action')
            s=flight.snapshot()
            aim=flight.execute_interception(s,plan=pending['plan'],recovery=True)
        else:
            aim=flight.execute_interception(s)
        pending=flight.coast(aim)
