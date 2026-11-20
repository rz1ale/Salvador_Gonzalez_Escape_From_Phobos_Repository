"""Legal public-I/O system identification. One run(context) for every JSON case."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from sim_core.common import digest, local_path, read_json, utc, write_json
from system_identification.campaigns.flight_math import (
    attitude_error, braking_distance, fit_coast, navigation, radar_threat,
    relative_geometry, rotation)

SMOKES = ('SMOKE01_LIFT', 'SMOKE02_PULSES', 'SMOKE03_COAST_MIXED')
SIGNATURE_FILES = ('system_identification/campaigns/identification.py',
    'system_identification/campaigns/flight_math.py', 'sim_core/runtime.py',
    'sim_core/common.py')

class GuardError(Exception):
    def __init__(self, reason, **details):
        self.reason, self.details = reason, details
        super().__init__(reason)

def signature(root, settings_path):
    files = {p: digest(local_path(root,p,must_exist=True)) for p in SIGNATURE_FILES}
    files[settings_path] = digest(local_path(root,settings_path,must_exist=True))
    official=Path(root).parent/'STU4-Sim'
    config=official/'Sims/COMPETITION_SIM.simconf'
    if not config.is_file(): raise ValueError('Official sibling configuration is missing')
    files['official/Sims/COMPETITION_SIM.simconf']=digest(config)
    # Read-only static source fingerprints, not live telemetry or ephemeris use.
    for folder,pattern in (('Systems','*.py'),('Data','*.csv')):
        if (official/folder).is_dir():
            for path in sorted((official/folder).rglob(pattern)):
                if folder=='Data' and not any(word in path.name.lower() for word in ('jet','thrust','rcs')):
                    continue  # never read private debris/ephemeris tables for runtime fingerprinting
                files['official/'+path.relative_to(official).as_posix()]=digest(path)
    files['numpy_version']=np.__version__
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()

def smoke_gate(root, expected):
    path = Path(root)/'system_identification'/'smoke_status.json'
    records = read_json(path) if path.is_file() else {}
    missing=[]
    for name in SMOKES:
        item=records.get(name)
        if not item or item.get('signature') != expected:
            missing.append(name)
            continue
        directory=local_path(root,item['attempt'])
        primary=directory/'result.json'
        if not primary.is_file():
            missing.append(name)
            continue
        value=read_json(primary)
        supervisor=directory/'supervisor_result.json'
        if supervisor.is_file():
            value=read_json(supervisor)
        if value.get('execution_status')!='completed' or value.get('task_outcome')!='success' or value.get('reason')!='sysid_'+name+'_complete':
            missing.append(name)
    return missing

def validate_settings(s):
    if s.get('schema_version')!=1:
        raise ValueError('Unsupported ID settings schema')
    numeric = [v for k,v in s.items() if k not in ('schema_version','height_definition','science_band_m')]
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in numeric):
        raise ValueError('ID settings must contain finite numeric values')
    for key,value in s.items():
        if key not in ('schema_version','height_definition','science_band_m','launch_height_min_m') and value<=0:
            raise ValueError('Positive ID setting required: '+key)
    lo,hi=s['science_band_m']
    if not 0 < s['airborne_height_min_m'] < lo < s['hover_height_m'] < hi < s['height_hard_max_m'] <= 35:
        raise ValueError('Invalid height ordering; historical 35 m ceiling may not be enlarged silently')
    if not .02 <= s['command_period_s'] <= .2 or not .005 <= s['native_step_prior_s'] <= .05:
        raise ValueError('Invalid control period or native-step prior')
    if s['brake_net_m_s2']<=0 or s['brake_angular_rad_s2']<=0 or s['reaction_delay_s']<0:
        raise ValueError('Invalid recovery assumptions')
    if s['launch_height_min_m']>0 or s['attitude_max_rad']>.6 or s['rate_max_rad_s']>.2:
        raise ValueError('Unexpected launch/rotation envelope')

def make_phases(p, settings):
    family=p['family']
    phases=[]
    def phase(name,duration,u=None,kind='science'):
        if not 0 < duration <= 20:
            raise ValueError('Invalid phase duration')
        return {'name':name,'duration':float(duration),'u':None if u is None else np.asarray(u,float), 'kind':kind}
    zero=np.zeros(6)
    if family=='smoke_lift':
        return [phase('hover_check',3.,None,'stabilization')]
    phases += [phase('baseline_coast',settings['baseline_coast_s'],zero,'coast'),
               {'name':'baseline_recovery','recover':True}]
    if family=='smoke_pulses':
        u=np.zeros(6);u[0]=.003
        phases += [phase('smoke_linear_pulse',.8,u,'isolated_pulse'),phase('smoke_linear_coast',1.,zero,'coast'),
                   {'name':'linear_recovery','recover':True}]
        u=np.zeros(6);u[3]=.003
        phases += [phase('smoke_angular_pulse',.15,u,'isolated_pulse'),phase('smoke_angular_coast',.6,zero,'coast')]
    elif family=='saturation':
        axis=p['axis']; amplitude=p['amplitude']; sign=p['sign']
        if type(axis) is not int or not 0<=axis<6 or sign not in (-1,1):
            raise ValueError('Invalid signed axis')
        cap=.5 if axis<3 else .05
        duration=float(p.get('pulse_s',1. if axis<3 else .2))
        bound=settings['linear_pulse_max_s'] if axis<3 else settings['angular_pulse_max_s']
        if not 0 < amplitude <= cap or not 0 < duration <= bound:
            raise ValueError('Pulse exceeds raw command/duration limits')
        u=np.zeros(6);u[axis]=sign*amplitude
        phases += [phase('axis_pulse',duration,u,'isolated_pulse'),phase('post_pulse_coast',1. if axis<3 else .6,zero,'coast')]
    elif family=='coast':
        phases += [phase('environment_coast',float(p.get('coast_s',5.)),zero,'coast')]
    elif family=='timing':
        axis=int(p['axis']); amp=float(p['amplitude'])
        if not 0<=axis<6 or not 0 < amp <= (.5 if axis<3 else .05):
            raise ValueError('Invalid timing axis/amplitude')
        durations=p['widths_s']
        for i,width in enumerate(durations):
            if not 0 < width <= (settings['linear_pulse_max_s'] if axis<3 else settings['angular_pulse_max_s']):
                raise ValueError('Timing width exceeds bounds')
            u=np.zeros(6);u[axis]=amp*(1 if i%2==0 else -1)
            phases += [phase('edge_pulse_'+str(i),width,u,'isolated_pulse'),
                       phase('edge_coast_'+str(i),.6,zero,'coast'),{'name':'edge_recovery_'+str(i),'recover':True}]
    elif family in ('smoke_coast_mixed','coupled','validation'):
        base=np.asarray(p.get('base_request',[.001,-.0008,.0012,.0005,.0003,-.0004]),float)
        if base.shape!=(6,) or not np.isfinite(base).all() or np.linalg.norm(base[:3])>.01 or np.linalg.norm(base[3:])>.003:
            raise ValueError('Mixed-request magnitude is outside this campaign domain')
        width=float(p.get('segment_s',.35 if family=='smoke_coast_mixed' else .5))
        patterns=np.asarray(p.get('patterns',[[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1],
                                             [1,-1,1,-1,1,-1],[-1,1,-1,1,-1,1]]),float)
        if patterns.ndim!=2 or patterns.shape[1]!=6 or np.max(np.abs(patterns))>1 or width*len(patterns)>settings['mixed_max_s']:
            raise ValueError('Invalid mixed six-input schedule')
        for i,pattern in enumerate(patterns):
            phases.append(phase('mixed_'+str(i),width,base*pattern,'coupled'))
        phases.append(phase('mixed_final_coast',1.,zero,'coast'))
    else:
        raise ValueError('Unknown experiment family: '+str(family))
    phases.append({'name':'final_recovery','recover':True})
    return phases

class Flight:
    def __init__(self,ctx):
        self.ctx=ctx
        self.p=dict(ctx.parameters)
        self.settings_path=self.p.get('settings_file','system_identification/campaigns/settings.json')
        self.s=read_json(local_path(ctx.root,self.settings_path,must_exist=True))
        validate_settings(self.s)
        self.phases=make_phases(self.p,self.s)
        self.signature=signature(ctx.root,self.settings_path)
        self._validate_parameters()
        self.launch=None;self.up=None;self.B0=None;self.Bref=None;self.Btarget=None;self.target=None
        self.g=None;self.start=None;self.last_t=None;self.last_phase=None;self.phase_start=None
        self.last_command=np.zeros(6);self.damage0=None;self.fuel0=None
        self.radar_rows=[];self.last_radar=-np.inf;self.valid_radar_seen=False
        self.nearest=None;self.airborne=False;self.first_motion=None;self.motion_test_done=False
        self.phase_records=[];self.samples=[];self.pulse_checks=[];self.abort_details=None
        self.baseline_g=None;self.last_settled=None
        self.log=ctx.output_dir/'telemetry.txt'
        self.columns=['t_sim_s','read_begin_s','read_end_s','send_begin_s','send_end_s','phase','kind',
            'h_launch_m','x_P_m','y_P_m','z_P_m','vx_P_m_s','vy_P_m_s','vz_P_m_s',
            'qPB_x','qPB_y','qPB_z','qPB_w','wx_B_rad_s','wy_B_rad_s','wz_B_rad_s',
            'u_ax','u_ay','u_az','u_alphax','u_alphay','u_alphaz','fuel_kg','damage_J','radar_clearance_margin_m']
        self.log.write_text('\t'.join(self.columns)+'\n',encoding='utf-8')

    def _validate_parameters(self):
        allowed={'family','split','settings_file','height_m','attitude_offset_rad','axis','sign','amplitude','pulse_s',
                 'coast_s','widths_s','base_request','patterns','segment_s','smoke_id'}
        unknown=set(self.p)-allowed
        if unknown: raise ValueError('Unknown ID parameters: '+','.join(sorted(unknown)))
        if self.p.get('split') not in ('smoke','training','validation','testing'):
            raise ValueError('Declare a flight-level split')
        if self.p['split']=='smoke' and self.p.get('smoke_id') not in SMOKES:
            raise ValueError('Unknown smoke ID')
        expected=dict(zip(SMOKES,('smoke_lift','smoke_pulses','smoke_coast_mixed')))
        if self.p['split']=='smoke' and expected[self.p['smoke_id']] != self.p['family']:
            raise ValueError('Smoke ID/family mismatch')
        if self.p['split']!='smoke' and 'smoke_id' in self.p:
            raise ValueError('Science cases cannot create smoke receipts')
        height=float(self.p.get('height_m',self.s['hover_height_m']))
        if not self.s['science_band_m'][0]+1 <= height <= self.s['science_band_m'][1]-1:
            raise ValueError('Reference height needs at least 1 m inside the science band')
        offset=np.asarray(self.p.get('attitude_offset_rad',[0,0,0]),float)
        if offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.21:
            raise ValueError('Attitude reference offset exceeds 0.21 rad')

    def sample(self):
        raw=self.ctx.state()
        t=.5*(raw['read_begin_sim_s']+raw['read_end_sim_s'])
        x={'t':t,'p':np.array(raw['position_P_m']),'v':np.array(raw['velocity_P_m_s']),
           'q':np.array(raw['q_PB_xyzw_reported']),'damage':raw['damage_J'],'fuel':raw['fuel_used_kg'],
           'read_begin':raw['read_begin_sim_s'],'read_end':raw['read_end_sim_s']}
        x['B']=rotation(x['q']); x['wB']=x['B']@np.array(raw['omega_relP_P_rad_s'])
        if self.launch is None:
            self.launch=x['p'].copy();self.up=self.launch/np.linalg.norm(self.launch);self.B0=x['B'].copy()
            offset=np.array(self.p.get('attitude_offset_rad',[0,0,0]))
            self.Btarget=Rotation.from_rotvec(-offset).as_matrix()@self.B0
            self.Bref=self.B0.copy()
            self.target=self.launch+self.up*float(self.p.get('height_m',self.s['hover_height_m']))
            self.g=-self.up*self.s['gravity_prior_m_s2']
            self.damage0=x['damage'];self.fuel0=x['fuel'];self.start=t
        x['h'],x['horizontal']=relative_geometry(x['p'],self.launch,self.up)
        # The airborne floor cannot arm below itself during the launch transit.
        if x['h']>=self.s['airborne_height_min_m']+1.: self.airborne=True
        if self.first_motion is None and x['h']>self.s['motion_min_m']:
            self.first_motion=t-self.start
            self.ctx.event('first_upward_motion',elapsed_sim_s=self.first_motion,height_m=x['h'])
        if t-self.last_radar>=self.s['radar_period_s']:
            self.radar_rows=self.ctx.radar(500);self.last_radar=t
        issue,self.nearest,count=radar_threat(self.radar_rows,t,x,self.s)
        self.valid_radar_seen=self.valid_radar_seen or count>0
        self.guard(x,issue,count)
        self.last_t=t
        return x

    def guard(self,x,radar_issue,count):
        s=self.s;t=x['t'];h=x['h'];v=x['v'];w=x['wB']
        def fail(reason,**data): raise GuardError(reason,sim_s=t,height_m=h,**data)
        if x['damage']>self.damage0+1e-6: fail('damage_guard',damage_J=x['damage'])
        if x['read_end']-x['read_begin']>s['read_span_max_s']: fail('state_read_span_guard')
        if self.last_t is not None and t-self.last_t>s['sample_gap_max_s']: fail('sample_gap_guard',gap_s=t-self.last_t)
        floor=s['airborne_height_min_m'] if self.airborne else s['launch_height_min_m']
        ceiling=s['height_hard_max_m']
        if self.last_settled is not None:
            floor,ceiling=s['science_band_m']
        if h<floor or h>ceiling: fail('science_band_guard' if self.last_settled is not None else 'height_guard',floor_m=floor,ceiling_m=ceiling)
        if np.linalg.norm(x['horizontal'])>s['horizontal_max_m']: fail('horizontal_guard')
        if np.linalg.norm(v)>s['speed_max_m_s']: fail('speed_guard',speed_m_s=float(np.linalg.norm(v)))
        if np.linalg.norm(w)>s['rate_max_rad_s']: fail('angular_rate_guard')
        angle=float(np.linalg.norm(attitude_error(x['B'],self.B0)))
        if angle>s['attitude_max_rad']: fail('attitude_guard',angle_rad=angle)
        vu=float(v@self.up)
        approximate_a=x['B'].T@(self.last_command[:3]/s['native_step_prior_s'])+self.g
        adverse=max(0.,np.sign(vu)*float(approximate_a@self.up))
        adverse=min(adverse,s['linear_delay_accel_bound_m_s2'])
        d=braking_distance(abs(vu),s['brake_net_m_s2'],s['reaction_delay_s'],s['stopping_margin_m'],adverse)
        if vu>0 and h+d>ceiling: fail('upper_stopping_guard',required_m=d)
        if self.airborne and vu<0 and h-d<floor: fail('lower_stopping_guard',required_m=d)
        lateral=v-vu*self.up
        lateral_speed=float(np.linalg.norm(lateral))
        lateral_adverse=min(s['linear_delay_accel_bound_m_s2'],float(np.linalg.norm(approximate_a-self.up*float(approximate_a@self.up))))
        horizontal_stop=braking_distance(lateral_speed,s['brake_net_m_s2'],s['reaction_delay_s'],s['stopping_margin_m'],lateral_adverse)
        if np.linalg.norm(x['horizontal'])+horizontal_stop>s['horizontal_max_m']: fail('horizontal_stopping_guard')
        angular_adverse=0.
        if float(w@self.last_command[3:])>0:
            angular_adverse=min(s['angular_delay_accel_bound_rad_s2'],float(np.linalg.norm(self.last_command[3:]/s['native_step_prior_s'])))
        predicted_angle=angle+braking_distance(float(np.linalg.norm(w)),s['brake_angular_rad_s2'],s['reaction_delay_s'],0.,angular_adverse)
        if predicted_angle>s['attitude_max_rad']: fail('angular_stopping_guard',predicted_rad=float(predicted_angle))
        if radar_issue: fail(radar_issue,clearance_margin_m=self.nearest)
        if t-self.start>s['radar_startup_grace_s'] and count==0:
            # An empty radar may mean no detections, not necessarily a failed sensor.
            # Record incomplete coverage, but never label the path proven clear.
            if not self.valid_radar_seen:
                self.ctx.event('radar_no_valid_detections',sim_s=t,note='No clearance proof; own-state guards remain active')
                self.valid_radar_seen=True  # suppress repeated identical warnings
        if t-self.start>s['motion_check_s'] and not self.motion_test_done:
            self.motion_test_done=True
            if self.first_motion is None: fail('no_upward_motion')

    def issue(self,x,u,phase,kind):
        if self.last_phase!=phase:
            if self.phase_records: self.phase_records[-1]['end_sim_s']=x['t']
            self.phase_records.append({'name':phase,'kind':kind,'start_sim_s':x['t'],'end_sim_s':None,'complete':False})
            self.ctx.event('phase_begin',phase=phase,kind=kind,sim_s=x['t'])
            self.last_phase=phase
        before=self.ctx.clock();timing=self.ctx.command(u[:3],u[3:]);after=self.ctx.clock()
        if isinstance(timing,dict):
            before=timing['send_begin_sim_s'];after=timing['send_end_sim_s']
        self.last_command=np.asarray(u).copy()
        values=[x['t'],x['read_begin'],x['read_end'],before,after,phase,kind,x['h'],
                *x['p'],*x['v'],*x['q'],*x['wB'],*u,x['fuel'],x['damage'],self.nearest]
        with self.log.open('a',encoding='utf-8') as f:
            f.write('\t'.join('' if v is None else (f'{v:.12g}' if isinstance(v,(float,np.floating)) else str(v)) for v in values)+'\n')
        return before,after

    def delay(self,deadline):
        remaining=deadline-time.monotonic()
        self.ctx.sleep(max(0.,remaining))

    def recover(self,name,first=False):
        deadline=time.monotonic();start=None;settled=None;last=None
        timeout=self.s['takeoff_timeout_s'] if first else self.s['recovery_timeout_s']
        while True:
            x=self.sample();last=x
            if start is None: start=x['t']
            u=navigation(x,self.target,self.Bref,self.g,self.s)
            self.issue(x,u,name,'takeoff' if first else 'recovery')
            stable=(np.linalg.norm(x['p']-self.target)<self.s['settle_position_m'] and
                    np.linalg.norm(x['v'])<self.s['settle_speed_m_s'] and
                    np.linalg.norm(attitude_error(x['B'],self.Bref))<self.s['settle_attitude_rad'] and
                    np.linalg.norm(x['wB'])<self.s['settle_rate_rad_s'])
            if stable:
                if settled is None: settled=x['t']
                if x['t']-settled>=self.s['settle_dwell_s']:
                    self.phase_records[-1].update(end_sim_s=x['t'],complete=True)
                    self.last_settled=x
                    return x
            else: settled=None
            if x['t']-start>timeout: raise GuardError('staging_timeout' if first else 'recovery_timeout',sim_s=x['t'])
            deadline+=self.s['command_period_s']
            if deadline<time.monotonic(): deadline=time.monotonic()
            self.delay(deadline)

    def timed(self,phase):
        start=None;first_send=None;deadline=time.monotonic();samples=[]
        while True:
            x=self.sample()
            if start is not None and phase['kind']=='isolated_pulse' and x['t']-start>phase['duration']+.15:
                raise GuardError('pulse_timing_guard',phase=phase['name'],actual_s=x['t']-start)
            if start is not None and x['t']-start>=phase['duration']-1e-9:
                # Close the pulse with a timestamped zero request before analysis.
                last_send=self.issue(x,np.zeros(6),phase['name']+'_end','transition')
                if len(self.phase_records)>1:
                    exposure=.5*(last_send[0]+last_send[1]-first_send[0]-first_send[1])
                    phase['observed_request_exposure_s']=exposure
                    self.phase_records[-2].update(end_sim_s=x['t'],complete=True,
                        requested_duration_s=phase['duration'],observed_request_exposure_s=exposure,
                        exposure_bracket_s=[max(0.,last_send[0]-first_send[1]),max(0.,last_send[1]-first_send[0])])
                return samples,x
            u=navigation(x,self.target,self.Bref,self.g,self.s) if phase['u'] is None else phase['u']
            sent=self.issue(x,u,phase['name'],phase['kind']);samples.append(x)
            if start is None: first_send=sent;start=sent[1]
            deadline+=self.s['command_period_s']
            if deadline<time.monotonic(): deadline=time.monotonic()
            self.delay(deadline)

    def _pulse_check(self,name,begin,end,u,pulse_duration):
        dt=end['t']-begin['t']
        delta_v=begin['B']@(end['v']-begin['v']-self.g*dt)
        delta_w=end['wB']-begin['wB']
        # Post-pulse coasting lets delayed velocity/rate reports include the full
        # pulse. Divide by published request exposure, not pulse+coast duration.
        response=np.r_[delta_v,delta_w]/max(pulse_duration,1e-6)
        axis=int(np.argmax(np.abs(u)))
        value=float(np.sign(u[axis])*response[axis])
        self.pulse_checks.append({'phase':name,'duration_observed_s':pulse_duration,'readout_interval_s':dt,'axis':axis,
                                  'signed_response':value,'response_body':response.tolist()})
        if self.p['split']=='smoke' and value<(.005 if axis<3 else .001):
            raise GuardError('smoke_response_guard',phase=name,axis=axis,response=value)

    def execute(self):
        if self.p['split']!='smoke':
            missing=smoke_gate(self.ctx.root,self.signature)
            if missing: raise GuardError('smoke_tests_required',missing=missing)
        initial=self.sample()
        if self.damage0>1e-6: raise GuardError('initial_damage_guard',damage_J=self.damage0)
        mass=float(self.ctx.shuttle.getMass())
        inertia=np.asarray(self.ctx.shuttle.getInertiaTensor(self.ctx.shuttle_frame),float)
        write_json(self.ctx.output_dir/'metadata.json',{'schema_version':1,'profile':self.p,'settings':self.s,
            'signature':self.signature,'mass_kg':mass,'inertia_B_kg_m2':inertia.tolist(),
            'launch_position_P_m':self.launch.tolist(),'launch_up_P':self.up.tolist(),
            'reporting_point':'SDK reported origin; no fitted COM correction assumed',
            'quaternion':'scalar-last xyzw, passive Phobos-to-body candidate convention',
            'environment':'local net gravity estimated from public own-velocity coasts; no hidden ephemeris',
            'native_tick':'unobserved; native_step_prior_s is a control-scaling prior only'})
        self.recover('takeoff_and_settle',first=True)
        self.Bref=self.Btarget
        if np.linalg.norm(attitude_error(self.Bref,self.B0))>1e-8:
            self.recover('reference_attitude')
        pending_pulse=None
        for phase in self.phases:
            if phase.get('recover'):
                self.recover(phase['name']);continue
            samples,end=self.timed(phase)
            if pending_pulse is not None and phase['kind']=='coast':
                self._pulse_check(*pending_pulse[:2],end,*pending_pulse[2:])
                pending_pulse=None
            if phase['name']=='baseline_coast':
                g,rms=fit_coast(samples,self.s['coast_washout_s'])
                if not .001<np.linalg.norm(g)<.04: raise GuardError('coast_environment_guard',gravity=g.tolist())
                self.g=g;self.baseline_g=g
                self.ctx.event('local_net_gravity',gravity_P_m_s2=g.tolist(),residual_velocity_rms_m_s=rms)
            if phase['kind']=='isolated_pulse' and samples:
                pending_pulse=(phase['name'],samples[0],phase['u'],phase['observed_request_exposure_s'])
        self.recover('completion_hold')
        return {'duration_sim_s':float(self.last_settled['t']-self.start),
                'fuel_kg':float(self.last_settled['fuel']-self.fuel0),'damage_J':float(self.last_settled['damage']-self.damage0),
                'first_motion_s':self.first_motion,'pulse_checks':self.pulse_checks,
                'local_gravity_P_m_s2':None if self.baseline_g is None else self.baseline_g.tolist()}

    def write_quality(self,success,reason):
        last=self.last_t if self.last_t is not None else self.start
        cutoff=last
        if reason=='damage_guard' and cutoff is not None: cutoff-=self.s['pre_contact_margin_s']
        if self.phase_records and self.phase_records[-1]['end_sim_s'] is None:
            self.phase_records[-1]['end_sim_s']=last
        clean=[{'start_sim_s':r['start_sim_s'],'end_sim_s':min(r['end_sim_s'],cutoff),
                'phase':r['name'],'kind':r['kind']} for r in self.phase_records
               if r['complete'] and r['kind'] not in ('takeoff','transition') and cutoff is not None and r['start_sim_s']<cutoff]
        write_json(self.ctx.output_dir/'quality.json',{'schema_version':1,'acquisition_success':success,'reason':reason,
            'profile':self.p,'signature':self.signature,'phases':self.phase_records,'clean_intervals':clean,
            'last_clean_cutoff_sim_s':cutoff,'censored':not success,'guard_details':self.abort_details,
            'note':'Contact-free selection is conditional; raw observations and every failed attempt are retained.'})

def run(context):
    context.request_stop_after_task()
    flight=None
    try:
        pending_name=context.parameters.get('smoke_id')
        if pending_name in SMOKES:
            path=context.root/'system_identification'/'smoke_status.json'
            current=read_json(path) if path.is_file() else {}
            current[pending_name]={'signature':None,'attempt':context.output_dir.relative_to(context.root).as_posix(),
                                   'started_utc':utc()}
            write_json(path,current)
        flight=Flight(context)
        # Point to the latest attempted smoke even on failure; never retain an older
        # success as the current result after a failed repeat of the same smoke.
        name=flight.p.get('smoke_id')
        if name:
            path=context.root/'system_identification'/'smoke_status.json'
            current=read_json(path) if path.is_file() else {}
            current[name]={'signature':flight.signature,'attempt':context.output_dir.relative_to(context.root).as_posix(),
                           'started_utc':utc()}
            write_json(path,current)
        metrics=flight.execute()
    except GuardError as exc:
        if flight:
            flight.abort_details=exc.details;flight.write_quality(False,exc.reason)
        context.event('sysid_guard_stop',reason=exc.reason,details=exc.details)
        context.finish(outcome='failure',reason=exc.reason,stop_sim=True,guard_details=exc.details)
    except Exception:
        if flight: flight.write_quality(False,'campaign_error')
        # Let the shared runner preserve the traceback; stopping is needed even in manual mode.
        context.event('sysid_campaign_error')
        raise
    except BaseException:
        if flight: flight.write_quality(False,'interrupted')
        raise
    flight.write_quality(True,'completed')
    name=flight.p.get('smoke_id')
    if name:
        path=context.root/'system_identification'/'smoke_status.json'
        current=read_json(path) if path.is_file() else {}
        current[name]={'signature':flight.signature,'attempt':context.output_dir.relative_to(context.root).as_posix(),
                       'completed_utc':utc()}
        write_json(path,current)
    reason='sysid_'+name+'_complete' if name else 'sysid_acquisition_complete'
    context.finish(outcome='success',reason=reason,stop_sim=True,**metrics)
