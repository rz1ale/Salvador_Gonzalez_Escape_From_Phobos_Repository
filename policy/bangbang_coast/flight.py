"""Simulator entry logic. All SDK interaction goes through run(context) on its thread."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from policy.bangbang_coast.core import (
    TargetMotion, cap, coast_trim, estimate_target_velocity, latent_from_reported,
    linearized_qr_gain, point_command, radar_lateral_body, select_plan, unit)


def load_settings(path):
    settings=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if settings.get('schema_version')!=1:
        raise ValueError('Unsupported bangbang settings schema')
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
    if settings['safety']['mission_timeout_s']<=0:
        raise ValueError('Invalid mission timeout')
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
        return radar_lateral_body(self.radar(now),C,now,self.cfg)

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

    def align(self,aim_P):
        self.phase='bangbang_align'
        t0=self.ctx.clock();stable=0
        while self.ctx.clock()-t0<self.cfg['attitude']['max_align_s']:
            s=self.snapshot();self.health(s);distance=self.arrived(s)
            angular,error,rate,settled=point_command(s['q_PB_xyzw_reported'],s['omega_relP_P_rad_s'],
                                                     aim_P,self.model,self.cfg)
            lateral,info=self.avoidance(s,self.ctx.clock())
            self.ctx.command(lateral,angular)
            self.log_periodic(s,distance,info,angle_deg=math.degrees(error),angular_rate_rad_s=rate)
            stable=stable+1 if settled else 0
            if stable>=self.cfg['attitude']['stable_cycles']:
                self.ctx.command(np.zeros(3),np.zeros(3))
                self.event('alignment_complete',elapsed_s=self.ctx.clock()-t0,
                           angle_deg=math.degrees(error),rate_rad_s=rate)
                return self.snapshot()
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        self.ctx.finish(outcome='failure',reason='bangbang_alignment_timeout',
                        align_elapsed_s=self.ctx.clock()-t0)

    def burn(self,aim_P,burn_s):
        self.phase='burn_x'
        t0=self.ctx.clock()
        self.event('burn_started',sim_s=t0,duration_s=burn_s,request_x=self.cfg['burn']['request_x_m_s2'])
        while self.ctx.clock()-t0<burn_s:
            s=self.snapshot();self.health(s);distance=self.arrived(s)
            ang,error,rate,_=point_command(s['q_PB_xyzw_reported'],s['omega_relP_P_rad_s'],
                                           aim_P,self.model,self.cfg)
            lateral,info=self.avoidance(s,self.ctx.clock())
            # Preserve the 0.5 m/s^2 VECTOR-NORM request limit when adding body Y/Z avoidance.
            x_cap=self.cfg['burn']['request_x_m_s2']
            x_req=min(x_cap,math.sqrt(max(0.,.5**2-lateral[1]**2-lateral[2]**2)))
            u=np.array([x_req,lateral[1],lateral[2]])
            self.ctx.command(u,ang)
            self.log_periodic(s,distance,info,angle_deg=math.degrees(error),x_request=x_req)
            self.tick()
        self.ctx.command(np.zeros(3),np.zeros(3))
        self.event('burn_ended',elapsed_s=self.ctx.clock()-t0)

    def coast(self,aim_P):
        self.phase='coast'
        coast_mode=self.cfg['coast']['mode']
        zero=np.zeros(3)
        # Cancel every translational and angular request before coasting.
        self.ctx.command(zero,zero)
        first=self.snapshot()
        fuel_at_entry=float(first['fuel_used_kg'])
        self.event('coast_started',sim_s=self.ctx.clock(),mode=coast_mode,
                   fuel_kg=fuel_at_entry)

        if coast_mode=='ballistic':
            # Strictly ballistic: no attitude hold, no Y/Z avoidance, no trims.
            # Repeat zero requests to prevent a previous command from persisting.
            while True:
                s=self.snapshot();self.health(s);distance=self.arrived(s)
                self.ctx.command(zero,zero)
                self.log_periodic(s,distance,{'mode':'ballistic_disabled'},
                                  coast_mode=coast_mode,trim_norm=0.0,
                                  coast_fuel_delta_kg=float(s['fuel_used_kg'])-fuel_at_entry)
                self.tick()

        # Original corrected-coast behavior, unchanged aside from added logging.
        while True:
            s=self.snapshot();self.health(s);distance=self.arrived(s)
            now=self.ctx.clock()
            ang,error,rate,_=point_command(s['q_PB_xyzw_reported'],s['omega_relP_P_rad_s'],
                                           aim_P,self.model,self.cfg)
            lateral,info=self.avoidance(s,now)
            target=self.target(s)
            if now-self.last_trim>=self.cfg['feedback']['trim_period_s']:
                self.trim_request,trim_info=coast_trim(s,self.model,target,self.cfg,now,aim_P,self.qr_gain)
                self.last_trim=now
                self.event('trim_update',sim_s=now,request_B=self.trim_request.tolist(),detail=trim_info)
            # No forward burn during nominal coast except explicitly small correction/avoidance.
            u=cap(self.trim_request+lateral,.5)
            self.ctx.command(u,ang)
            self.log_periodic(s,distance,info,angle_deg=math.degrees(error),
                              trim_norm=float(np.linalg.norm(u)),coast_mode=coast_mode,
                              coast_fuel_delta_kg=float(s['fuel_used_kg'])-fuel_at_entry)
            self.tick()


def run(context):
    from system_identification.model import Model
    from policy.bangbang_coast.core import axes_for_direction
    cfg=load_settings(context.root/'policy'/'bangbang_coast'/'settings.json')
    if context.model_path is None:
        raise ValueError('Select the root nominal_model.json using --config/--model')
    model=Model.load(context.model_path)
    if context.parameters:
        raise ValueError('This policy is configured in settings.json; per-trial parameter overrides not supported')
    (context.output_dir/'settings_used.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    flight=Flight(context,model,cfg)
    flight.event('policy_started',mode=cfg['objective']['mode'],model_path=str(context.model_path),
                 model_status=model.definition.get('status','unknown'),
                 note='Not independently validated; no native score guarantee')
    flight.launch_boost()
    s=flight.collect_target()
    flight.phase='planning'
    plan=select_plan(s,model,flight.target(s),cfg,context.clock())
    flight.event('nominal_plan',**plan)
    if not plan['feasible'] and cfg['planning']['require_feasible_plan']:
        context.command(np.zeros(3),np.zeros(3))
        context.finish(outcome='failure',reason='bangbang_no_feasible_nominal_arrival',
                       predicted_closest_m=plan['predicted_closest_m'])
    aim=np.asarray(plan['aim_P'],float)
    flight.align(aim)

    if cfg['planning']['refit_after_alignment']:
        flight.phase='refit_after_alignment'
        s=flight.snapshot();flight.health(s)
        from system_identification.model import matrix
        C=matrix(np.asarray(s['q_PB_xyzw_reported'],float))
        x_direction=C.T@np.array([1.,0.,0.])
        refined=select_plan(s,model,flight.target(s),cfg,context.clock(),
                            force_direction=x_direction,known_aligned=True)
        flight.event('post_alignment_refit',**refined)
        if refined['feasible']:
            plan=refined
        elif cfg['planning']['require_feasible_post_align_refit']:
            context.command(np.zeros(3),np.zeros(3))
            context.finish(outcome='failure',reason='bangbang_post_align_infeasible',
                           predicted_closest_m=refined['predicted_closest_m'])

    if cfg['objective']['mode']=='quadratic' and cfg['feedback']['qr_feedback_enabled']:
        s=flight.snapshot()
        p,_,_,_,_=latent_from_reported(s,model)
        flight.qr_gain=linearized_qr_gain(p,aim,model,cfg)
        flight.event('qr_gain_scheduled',K=flight.qr_gain.tolist(),
                     note='6-state translational LQR about fixed pointing; nonlinear allocator inverse afterward')
    flight.burn(aim,plan['burn_s'])
    flight.coast(aim)
