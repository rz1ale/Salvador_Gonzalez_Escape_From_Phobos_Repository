"""Offline physical-fixture checks; they never certify a native STPro flight."""
from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from sim_core.common import read_json, write_json, validate
from run_simulation import parser, update_selection
from sim_core.runtime import Context, Finish
from system_identification.campaigns.identification import Flight, GuardError, make_phases, run, signature, smoke_gate, SMOKES
from system_identification.campaigns.flight_math import attitude_error, fit_coast, navigation, radar_threat, rotation
from system_identification.campaigns.make_plans import plans
from system_identification.analysis_code.summarize import audit, inspect_attempt, telemetry as analysis_telemetry, main as analysis_main

PACKAGE=Path(__file__).resolve().parents[2]

class Finished(BaseException):
    def __init__(self, result): self.result=result

class PlantContext:
    """Independent continuous rigid-body fixture with bounded shared actuation.

    Hardware limits approximate the historical source, not its exact LP. States
    use passive P->B orientation, delayed rate/velocity and a reporting lever arm.
    No native files, SDK allocator telemetry, or simulator engine are involved.
    """
    def __init__(self,root,case,delay=.045,native_step=.02072,rpc_latency=0.):
        self.root=Path(root);self.parameters=dict(settings_file='system_identification/campaigns/settings.json',**case['parameters'])
        self.assignment={'case_id':case['id'],'trial_id':case['id'],'attempt_id':'attempt_001'}
        self.output_dir=self.root/'system_identification/data_sets'/case['id']/'attempt_001'
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.session_dir=self.output_dir.parent
        self.t=0.;self.p=np.array([0.,0.,12000.]);self.v=np.zeros(3);self.B=np.eye(3);self.w=np.zeros(3)
        self.u=np.zeros(6);self.mass=3388.7;self.fuel=0.;self.damage=0.;self.delay=delay;self.h=native_step
        self.history=[(0.,self.v.copy(),self.w.copy())]
        self.commands=[];self.events=[];self.force_stop=False;self.thrust_enabled=True;self.rpc_latency=rpc_latency
        self.ell=np.array([-.285,0.,-1.494]);self.radar_override=None
        self.shuttle=types.SimpleNamespace(getMass=lambda:self.mass,getInertiaTensor=lambda f:np.diag([3025.8,5927.5,6756.]))
        self.shuttle_frame='body'
    def request_stop_after_task(self): self.force_stop=True
    def clock(self): return self.t
    def state(self):
        begin=self.t
        if self.rpc_latency: self.sleep(self.rpc_latency)
        past=max(0.,self.t-self.delay)
        item=min(self.history,key=lambda x:abs(x[0]-past))
        return dict(read_begin_sim_s=begin,read_end_sim_s=self.t,position_P_m=(self.p+self.B.T@self.ell).tolist(),
            velocity_P_m_s=item[1].tolist(),q_PB_xyzw_reported=Rotation.from_matrix(self.B).as_quat().tolist(),
            omega_relP_P_rad_s=(self.B.T@item[2]).tolist(),fuel_used_kg=self.fuel,damage_J=self.damage)
    def radar(self,n):
        if self.radar_override is not None: return self.radar_override
        return [np.array([[1000.,0,0],[999.,0,2.],[self.t,0,0]])]
    def event(self,name,**data): self.events.append((name,data))
    def command(self,a,b):
        before=self.t
        if self.rpc_latency: self.sleep(self.rpc_latency/2)
        self.u=np.r_[a,b].astype(float)
        self.commands.append((self.t,self.u.copy()))
        return dict(send_begin_sim_s=before,send_end_sim_s=self.t)
    def sleep(self,seconds):
        end=self.t+seconds
        if not seconds>0: return
        while self.t<end-1e-10:
            dt=min(.01,end-self.t)
            requested=self.u/self.h if self.thrust_enabled else np.zeros(6)
            positive=np.array([.175,.143,.096,.125,.076,.0805])
            negative=np.array([.0966,.0964,.117,.124,.0903,.0805])
            limits=np.where(requested>=0,positive,negative)
            # One shared feasible scaling; different from the exact native jet LP.
            load=np.sqrt(np.sum((requested/limits)**2))
            achieved=requested/max(1.,load)
            accel=self.B.T@achieved[:3]+np.array([0,0,-.01])
            self.p+=self.v*dt+.5*accel*dt*dt;self.v+=accel*dt
            self.w+=achieved[3:]*dt
            self.B=Rotation.from_rotvec(-self.w*dt).as_matrix()@self.B
            self.fuel+=float(np.linalg.norm(achieved[:3])*self.mass*dt/784.)
            self.t+=dt
            self.history.append((self.t,self.v.copy(),self.w.copy()))
        # Keep enough history for reporting delay without quadratic lookup cost.
        self.history=[x for x in self.history if x[0]>=self.t-.3]
    def finish(self,*,outcome='success',reason='done',stop_sim=False,**metrics):
        value=dict(execution_status='completed',task_outcome=outcome,reason=reason,metrics=metrics,
                   stop_sim_requested=stop_sim)
        write_json(self.output_dir/'result.json',value)
        raise Finished(value)

class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'STU4-Solution'
        self.root.mkdir()
        for name in ('run_simulation.py','sim_core','system_identification'):
            source,target=PACKAGE/name,self.root/name
            if source.is_dir():
                shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','data_sets','analysis_results','state'))
            else: shutil.copyfile(source,target)
        native=self.root.parent/'STU4-Sim/Sims'
        native.mkdir(parents=True)
        (native/'COMPETITION_SIM.simconf').write_text('{"offline_fixture":true}')
        self.definitions=plans()
    def tearDown(self): self.temp.cleanup()
    def execute(self,case,**kwargs):
        ctx=PlantContext(self.root,case,**kwargs)
        with patch('system_identification.campaigns.identification.time.monotonic',side_effect=lambda:ctx.t):
            try: run(ctx)
            except Finished as exc: return ctx,exc.result
        self.fail('Task did not produce a terminal result')
    def allow_science(self):
        sig=signature(self.root,'system_identification/campaigns/settings.json');status={}
        for name in SMOKES:
            d=self.root/'fixture_smokes'/name;d.mkdir(parents=True,exist_ok=True)
            write_json(d/'result.json',dict(execution_status='completed',task_outcome='success',reason='sysid_'+name+'_complete'))
            status[name]=dict(signature=sig,attempt=d.relative_to(self.root).as_posix())
        write_json(self.root/'system_identification/smoke_status.json',status)
    def test_plan_sizes_axis_sign_amplitude_coverage(self):
        sat=self.definitions['saturation.json']['schedule']['cases']
        self.assertEqual(len(sat),48)
        for axis in range(6):
            for sign in (-1,1):
                rows=[x for x in sat if x['parameters']['axis']==axis and x['parameters']['sign']==sign]
                self.assertEqual(len(rows),4)
                self.assertEqual(len({x['parameters']['amplitude'] for x in rows}),4)
        allcases=self.definitions['full_campaign.json']['schedule']['cases']
        self.assertEqual(len(allcases),72)
        self.assertEqual(sum(x['parameters']['split']=='testing' for x in allcases),4)
    def test_every_plan_and_phase_validates(self):
        s=read_json(self.root/'system_identification/campaigns/settings.json')
        for name,c in self.definitions.items():
            validate(self.root,c)
            for case in c['schedule']['cases']:
                p=dict(c['parameters'],**case['parameters']);make_phases(p,s)
    def test_coupled_profiles_are_balanced_and_six_input_rank(self):
        for name in ('coupled.json','validation.json'):
            for case in self.definitions[name]['schedule']['cases']:
                matrix=np.array(case['parameters']['patterns'])
                np.testing.assert_array_equal(matrix.sum(axis=0),np.zeros(6))
                self.assertEqual(np.linalg.matrix_rank(matrix),6)
    def test_config_case_selection_uses_shared_cli_and_preserves_source(self):
        cfg='system_identification/campaigns/smoke_tests.json'
        before=(self.root/cfg).read_bytes()
        args=parser().parse_args(['select','--config',cfg,'--case','SMOKE01_LIFT','--runs','1'])
        c=update_selection(self.root,args,'manual')
        self.assertEqual(c['schedule']['cases'][0]['id'],'SMOKE01_LIFT')
        self.assertIn(cfg,c['dependencies'])
        self.assertEqual((self.root/cfg).read_bytes(),before)
    def test_single_entry_selection_clears_previous_campaign_inputs(self):
        write_json(self.root/'active_run.json',self.definitions['full_campaign.json'])
        args=parser().parse_args(['select','--entry','sim_core/smoke_task.py','--runs','1'])
        c=update_selection(self.root,args,'manual')
        self.assertEqual(c['parameters'],{})
        self.assertEqual(c['output_root'],'run_logs')
        self.assertEqual(len(c['schedule']['cases']),1)
        self.assertEqual(c['dependencies'],[])
    def test_smoke_gate_rejects_missing_and_failed_receipts(self):
        sig=signature(self.root,'system_identification/campaigns/settings.json')
        self.assertEqual(set(smoke_gate(self.root,sig)),set(SMOKES))
        self.allow_science();self.assertEqual(smoke_gate(self.root,sig),[])
        d=self.root/'fixture_smokes'/SMOKES[0]
        write_json(d/'supervisor_result.json',dict(execution_status='failed',task_outcome='not_assessed',reason='shutdown_timeout'))
        self.assertIn(SMOKES[0],smoke_gate(self.root,sig))
    def test_settings_change_invalidates_smoke_gate(self):
        self.allow_science()
        p=self.root/'system_identification/campaigns/settings.json'
        value=read_json(p);value['settle_dwell_s']+=.01;write_json(p,value)
        self.assertEqual(len(smoke_gate(self.root,signature(self.root,'system_identification/campaigns/settings.json'))),3)
    def test_invalid_latest_smoke_also_supersedes_older_pass(self):
        self.allow_science()
        case=copy.deepcopy(self.definitions['smoke_tests.json']['schedule']['cases'][0])
        case['parameters']['family']='invalid_family'
        with self.assertRaises(ValueError): self.execute(case)
        sig=signature(self.root,'system_identification/campaigns/settings.json')
        self.assertIn(SMOKES[0],smoke_gate(self.root,sig))
    def test_official_source_change_invalidates_smoke_gate(self):
        self.allow_science()
        path=self.root.parent/'STU4-Sim/Sims/COMPETITION_SIM.simconf'
        path.write_text('{"offline_fixture":"updated"}')
        self.assertEqual(len(smoke_gate(self.root,signature(self.root,'system_identification/campaigns/settings.json'))),3)
    def test_manual_finish_stop_is_explicit_and_default_is_preserved(self):
        ctx=Context.__new__(Context)
        with self.assertRaises(Finish) as ex: ctx.finish()
        self.assertFalse(ex.exception.value['stop_sim_requested'])
        with self.assertRaises(Finish) as ex: ctx.finish(stop_sim=True)
        self.assertTrue(ex.exception.value['stop_sim_requested'])
        ctx.request_stop_after_task();self.assertTrue(ctx._stop_after_task)
    def test_navigation_restores_passive_attitude_error(self):
        B=Rotation.from_rotvec([0,0,-.1]).as_matrix()
        state=dict(p=np.zeros(3),v=np.zeros(3),B=B,wB=np.zeros(3))
        s=read_json(self.root/'system_identification/campaigns/settings.json')
        u=navigation(state,np.zeros(3),np.eye(3),np.zeros(3),s)
        self.assertLess(u[5],0)
    def test_coast_estimate_and_washout(self):
        g=np.array([.001,-.002,-.01])
        samples=[dict(t=i*.05,v=g*i*.05+np.array([.2,0,0])) for i in range(40)]
        actual,rms=fit_coast(samples,.25)
        np.testing.assert_allclose(actual,g,atol=1e-12);self.assertLess(rms,1e-12)
    def test_radar_uses_public_timestamps_and_own_tangent(self):
        s=read_json(self.root/'system_identification/campaigns/settings.json')
        x={'v':np.array([0,1.,0])}
        close=np.array([[8.,0,0],[6.,-.2,2.],[1.,0,0]])
        reason,margin,count=radar_threat([close],1.,x,s)
        self.assertIsNotNone(reason);self.assertEqual(count,1)
        reason,margin,count=radar_threat([close],10.,x,s)
        self.assertIsNone(reason);self.assertEqual(count,0)
    def test_three_smokes_lift_pulse_coast_and_recovery(self):
        for case in self.definitions['smoke_tests.json']['schedule']['cases']:
            ctx,value=self.execute(case)
            self.assertEqual(value['task_outcome'],'success',value)
            self.assertTrue(value['stop_sim_requested'])
            self.assertTrue(ctx.force_stop)
            self.assertLess(ctx.t,240.)
            q=read_json(ctx.output_dir/'quality.json')
            self.assertTrue(q['acquisition_success'])
        sig=signature(self.root,'system_identification/campaigns/settings.json')
        self.assertEqual(smoke_gate(self.root,sig),[])
    def test_science_without_smokes_stops_before_nonzero_command(self):
        case=self.definitions['saturation.json']['schedule']['cases'][0]
        ctx,value=self.execute(case)
        self.assertEqual(value['reason'],'smoke_tests_required')
        self.assertEqual(len(ctx.commands),0)
        self.assertTrue(value['stop_sim_requested'])
    def test_pulse_readout_with_reporting_and_rpc_delays(self):
        case=self.definitions['smoke_tests.json']['schedule']['cases'][1]
        ctx,value=self.execute(case,delay=.045,rpc_latency=.01)
        self.assertEqual(value['task_outcome'],'success',value)
        checks=value['metrics']['pulse_checks']
        self.assertEqual(len(checks),2)
        for point in checks:
            self.assertGreater(point['readout_interval_s'],point['duration_observed_s'])
    def test_zero_thrust_aborts_instead_of_waiting_for_staging_timeout(self):
        case=self.definitions['smoke_tests.json']['schedule']['cases'][0]
        ctx=PlantContext(self.root,case);ctx.thrust_enabled=False
        with patch('system_identification.campaigns.identification.time.monotonic',side_effect=lambda:ctx.t):
            with self.assertRaises(Finished) as ex: run(ctx)
        self.assertEqual(ex.exception.result['reason'],'no_upward_motion')
        self.assertLess(ctx.t,4.)
        # The failed newest attempt supersedes any old receipt.
        self.assertIn(SMOKES[0],smoke_gate(self.root,signature(self.root,'system_identification/campaigns/settings.json')))
    def test_detected_obstacle_aborts_before_takeoff(self):
        case=self.definitions['smoke_tests.json']['schedule']['cases'][0]
        ctx=PlantContext(self.root,case)
        ctx.t=1.;ctx.radar_override=[np.array([[5.,0,0],[3.,0,2.],[1.,0,0]])]
        with patch('system_identification.campaigns.identification.time.monotonic',side_effect=lambda:ctx.t):
            with self.assertRaises(Finished) as ex: run(ctx)
        self.assertEqual(ex.exception.result['reason'],'radar_clearance_guard')
        self.assertFalse(ctx.commands)
    def test_audit_parses_actual_campaign_telemetry(self):
        case=self.definitions['smoke_tests.json']['schedule']['cases'][0]
        ctx,value=self.execute(case)
        write_json(ctx.session_dir/'plan.json',{'config':{'entry':'fixture'}})
        write_json(ctx.session_dir/'ledger.json',dict(state='complete',attempts=[dict(trial_id=case['id'],
            attempt_id='attempt_001',directory='attempt_001',parameters=ctx.parameters,result=value)]))
        output=audit(self.root,ctx.session_dir,self.root/'audit_result',plots='none')
        rows=read_json(output/'audit.json')['attempts']
        self.assertGreater(rows[0]['samples'],100)
        self.assertLess(rows[0]['height_max_m'],30.)
        self.assertTrue((output/'metrics.txt').is_file())
        self.assertTrue((output/'report.txt').is_file())
        self.assertTrue((output/'assessment.json').is_file())
    def pulse_analysis_fixture(self,outcome='success',response=.2):
        profile=dict(family='saturation',split='training',axis=0,sign=1,amplitude=.5)
        settings=read_json(self.root/'system_identification/campaigns/settings.json')
        rows=[]
        for t in np.round(np.arange(0,2.01,.1),8):
            phase=('baseline_coast' if t<1 else 'axis_pulse' if t<1.5 else 'axis_pulse_end' if t==1.5
                   else 'post_pulse_coast' if t<2 else 'post_pulse_coast_end')
            r=dict(t_sim_s=t,read_begin_s=t,read_end_s=t,send_begin_s=t,send_end_s=t,phase=phase,
                kind='isolated_pulse' if phase=='axis_pulse' else 'coast' if 'coast' in phase else 'transition',
                h_launch_m=25.,vx_P_m_s=response*min(max(t-1,0),.5),vy_P_m_s=0.,vz_P_m_s=-.01*t,
                qPB_x=0.,qPB_y=0.,qPB_z=0.,qPB_w=1.,wx_B_rad_s=0.,wy_B_rad_s=0.,wz_B_rad_s=0.,
                fuel_kg=0.,damage_J=0.,u_ax=.5 if phase=='axis_pulse' else 0.,u_ay=0.,u_az=0.,u_alphax=0.,u_alphay=0.,u_alphaz=0.)
            rows.append({k:str(v) for k,v in r.items()})
        phases=[dict(name='axis_pulse',kind='isolated_pulse',complete=True,start_sim_s=1.,end_sim_s=1.5,
                     observed_request_exposure_s=.5,requested_duration_s=.5),
                dict(name='post_pulse_coast',kind='coast',complete=True,start_sim_s=1.6,end_sim_s=2.)]
        quality=dict(phases=phases,clean_intervals=[dict(phase=p['name'],start_sim_s=p['start_sim_s'],end_sim_s=p['end_sim_s']) for p in phases])
        summary=dict(trial_id='trial_A',attempt_id='attempt_001',outcome=outcome,
                     reason='done' if outcome=='success' else 'height_guard')
        return rows,profile,dict(settings=settings,profile=profile),quality,summary
    def test_local_analysis_recomputes_response_and_keeps_censored_data_separate(self):
        rows,p,m,q,s=self.pulse_analysis_fixture(outcome='failure')
        review=inspect_attempt(rows,self.root,p,m,q,s)
        self.assertFalse(any(f['level']=='error' for f in review['findings']))
        self.assertAlmostEqual(review['pulses'][0]['measured_signed_response'],.2,places=10)
        self.assertTrue(review['pulses'][0]['censored_flight'])
        self.assertAlmostEqual(review['pulses'][0]['duration_observed_s'],.5)
    def test_local_analysis_detects_cross_axis_commands_and_sign_reversal(self):
        rows,p,m,q,s=self.pulse_analysis_fixture(response=-.2)
        review=inspect_attempt(rows,self.root,p,m,q,s)
        self.assertIn('nonpositive_commanded_axis_response',[f['code'] for f in review['findings']])
        next(r for r in rows if r['phase']=='axis_pulse')['u_ay']='.1'
        review=inspect_attempt(rows,self.root,p,m,q,s)
        self.assertIn('pulse_not_isolated_and_constant',[f['code'] for f in review['findings']])
        self.assertEqual(review['pulses'],[])
    def test_local_analysis_nonfinite_input_is_reported(self):
        rows,p,m,q,s=self.pulse_analysis_fixture()
        rows[4]['vx_P_m_s']='nan'
        review=inspect_attempt(rows,self.root,p,m,q,s)
        self.assertIn('nonfinite_telemetry',[f['code'] for f in review['findings']])
    def test_local_analysis_damage_and_censored_boundaries_are_not_accepted(self):
        rows,p,m,q,s=self.pulse_analysis_fixture()
        rows[-1]['damage_J']='1.0'
        review=inspect_attempt(rows,self.root,p,m,q,s)
        self.assertIn('successful_attempt_has_damage_increment',[f['code'] for f in review['findings']])
        rows[-1]['damage_J']='0.0';q['clean_intervals'][0]['start_sim_s']=1.1
        review=inspect_attempt(rows,self.root,p,m,q,s)
        self.assertEqual(review['pulses'],[])
        self.assertIn('pulse_or_readout_censored',[f['code'] for f in review['findings']])
    def test_local_analysis_heldout_trajectory_remains_sealed(self):
        s=self.root/'system_identification/data_sets/heldout_fixture';d=s/'attempt_001';d.mkdir(parents=True)
        (d/'telemetry.txt').write_text('not numeric trajectory data; deliberately unparseable')
        profile=dict(family='validation',split='testing')
        write_json(d/'metadata.json',dict(profile=profile))
        write_json(s/'plan.json',dict(config=dict(entry='system_identification/campaigns/identification.py')))
        write_json(s/'ledger.json',dict(state='complete',attempts=[dict(trial_id='held',attempt_id='attempt_001',
            directory='attempt_001',parameters=profile,result=dict(task_outcome='success',reason='done'))]))
        output=audit(self.root,s,self.root/'heldout_audit')
        row=read_json(output/'audit.json')['attempts'][0]
        self.assertTrue(row['heldout_not_inspected']);self.assertNotIn('samples',row)
    def test_local_analysis_latest_pointer_and_readable_assessment(self):
        rows,profile,metadata,quality,summary=self.pulse_analysis_fixture()
        session=self.root/'system_identification/data_sets/pulse_fixture';d=session/'attempt_001';d.mkdir(parents=True)
        write_json(d/'metadata.json',metadata);write_json(d/'quality.json',quality)
        with (d/'telemetry.txt').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
        value=dict(execution_status='completed',task_outcome='success',reason='done')
        write_json(session/'plan.json',dict(config=dict(entry='system_identification/campaigns/identification.py')))
        write_json(session/'ledger.json',dict(state='complete',attempts=[dict(trial_id='trial_A',attempt_id='attempt_001',
            directory='attempt_001',parameters=profile,result=value)],trial_results=[dict(trial_id='trial_A',result=value)]))
        write_json(self.root/'sim_core/state/active_session.json',dict(session=session.relative_to(self.root).as_posix()))
        with patch('system_identification.analysis_code.summarize.ROOT',self.root):
            self.assertEqual(analysis_main(['--latest','--show','--out','analysis_fixture_output']),0)
        report=read_json(self.root/'analysis_fixture_output/assessment.json')
        self.assertEqual(report['verified_successful_axis_pulses'],1)
        self.assertEqual(report['next_step'],'COLLECT_MISSING_DIRECTIONS')
        self.assertFalse((self.root/'system_identification/nominal_model.json').exists())
    def test_signed_saturated_axes_and_coast_offset_complete(self):
        self.allow_science()
        sat=self.definitions['saturation.json']['schedule']['cases']
        cases=[x for x in sat if x['id'].endswith('A04')]
        cases.append(self.definitions['coasts.json']['schedule']['cases'][2])
        for case in cases:
            ctx,value=self.execute(case)
            self.assertEqual(value['task_outcome'],'success',value)
            self.assertLess(ctx.damage,1e-6)
            if 'axis' in case['parameters']:
                pulse=[u for _,u in ctx.commands if abs(u[case['parameters']['axis']]-case['parameters']['sign']*case['parameters']['amplitude'])<1e-12]
                self.assertTrue(pulse)
                for u in pulse: self.assertEqual(np.count_nonzero(u),1)
                scan=inspect_attempt(
                    analysis_telemetry(ctx.output_dir/'telemetry.txt'),
                    ctx.output_dir,ctx.parameters,read_json(ctx.output_dir/'metadata.json'),
                    read_json(ctx.output_dir/'quality.json'),dict(trial_id=case['id'],attempt_id='attempt_001',
                        outcome='success',reason=value['reason']))
                self.assertFalse(any(f['level']=='error' for f in scan['findings']),scan['findings'])
                self.assertGreater(scan['pulses'][0]['measured_signed_response'],0.)

if __name__=='__main__': unittest.main()
