"""V4 startup modes; public nominal model and mock simulator, never native SDK."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import Model, matrix
from policy.bangbang_coast_v4.core import TargetMotion,radar_lateral_body
from policy.bangbang_coast_v4.flight import Flight,load_settings
from policy.bangbang_coast_v4.startup import (
    select_startup,candidate,predict,rotation_between,slerp_direction,evaluate_remaining)
from test_integration_mock import MockContext,FlightEnded

class TestStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_settings(ROOT/'policy/bangbang_coast_v4/settings.json')
        cls.model=Model.load(ROOT/'nominal_model.json')
    def mock(self):
        ctx=MockContext(Path('/tmp'))
        ctx.x[:3]=[10000.,0.,0.]
        ctx.x[6:10]=[0.,0.,0.,1.]
        return ctx
    def target(self):return TargetMotion([10000.,0.,400.],np.zeros(3),'fixed')
    def test_startup_mode_matrix(self):
        ctx=self.mock();s=ctx.state();tar=self.target()
        for mode in ('no_orient','partial_turn','full_turn','rotate_and_thrust','auto_compare'):
            with self.subTest(mode=mode):
                cfg=copy.deepcopy(self.cfg)
                cfg['startup']['mode']=mode
                cfg['startup']['body_direction_B']=[0.,0.,1.]
                cfg['startup']['candidate_burn_samples']=9
                cfg['startup']['predict_step_s']=2.
                cfg['startup']['auto_partial_fractions']=[.5]
                z=select_startup(s,self.model,tar,dict(cfg,_start_sim=0.),0.)
                self.assertEqual(len(z['body_direction_B']),3)
                self.assertLessEqual(z['turn_angle_deg'],180.)
                self.assertGreater(z['planned_burn_s'],0.)
                if mode=='no_orient':
                    self.assertAlmostEqual(z['turn_angle_deg'],0.,places=7)
                    self.assertEqual(z['mode'],'no_orient')
                elif mode=='rotate_and_thrust':
                    self.assertEqual(z['mode'],mode)
                    self.assertEqual(z['overlap_translation'],z['turn_angle_deg']>1e-6)
    def test_partial_turn_is_actually_partial(self):
        C=np.eye(3);a=np.array([0.,0.,1.]);b=np.array([1.,0.,0.])
        q=candidate('partial_turn',a,.5,b,C,self.cfg)
        self.assertAlmostEqual(q['turn_angle_deg'],45.,places=6)
        self.assertAlmostEqual(float(np.dot(q['aim_P'],a)),math.cos(math.pi/4),places=6)
    def test_rotation_antiparallel(self):
        a=[1.,0.,0.];b=[-1.,0.,0.]
        R,theta=rotation_between(a,b)
        self.assertAlmostEqual(theta,math.pi)
        np.testing.assert_allclose(R@a,b,atol=1e-7)
        self.assertAlmostEqual(np.linalg.norm(slerp_direction(a,b,.5)),1.,places=8)
        self.assertAlmostEqual(float(slerp_direction(a,b,.5)@np.array(a)),0.,places=8)
    def test_free_terminal_region_and_velocity(self):
        ctx=self.mock();s=ctx.state();s['velocity_P_m_s']=[0.,0.,5.]
        cfg=copy.deepcopy(self.cfg);cfg['startup']['predict_horizon_s']=100.
        ans=predict(s,self.model,self.target(),dict(cfg,_start_sim=0.),0.,[0,0,1],[0,0,1],[0.])
        self.assertTrue(ans['feasible'][0])
        self.assertLess(ans['arrival_sim_s'][0],100.)
        self.assertEqual(cfg['objective']['quadratic']['Q_terminal_velocity'],0.)
        self.assertAlmostEqual(np.linalg.norm(ans['entry_P_m'][0]-[10000.,0.,400.]),42.,delta=2.)
    def test_axis_radar_avoidance_is_perpendicular_to_z(self):
        cfg=copy.deepcopy(self.cfg)
        # Threat at Z with lateral Y separation; closing radially.
        radar=[[ [0.,1.,14.], [5.,-4.,5.],[0.,0.,0.] ]]
        u,info=radar_lateral_body(radar,np.eye(3),0.,cfg,velocity_P=[0.,0.,0.],thrust_axis_B=[0.,0.,1.])
        self.assertAlmostEqual(float(u[2]),0.,places=10)
        self.assertLessEqual(np.linalg.norm(u),.5+1e-8)
    def test_all_presets_are_valid(self):
        folder=ROOT/'policy/bangbang_coast_v4/presets'
        paths=sorted(folder.glob('settings_*.json'))
        self.assertGreaterEqual(len(paths),6)
        for path in paths:
            with self.subTest(path=path.name):
                cfg=load_settings(path)
                self.assertEqual(cfg['schema_version'],4)

    def test_auto_no_orient_target_los_request(self):
        ctx=self.mock();s=ctx.state()
        d=np.asarray(s['target_P_m'],float)-np.asarray(s['position_P_m'],float)
        cfg=copy.deepcopy(self.cfg);cfg['startup']['mode']='no_orient'
        cfg['startup']['no_orient_body_selection']='target_los'
        plan=select_startup(s,self.model,self.target(),dict(cfg,_start_sim=0.),0.)
        self.assertAlmostEqual(plan['turn_angle_deg'],0.)
        self.assertGreater(np.linalg.norm(plan['body_direction_B']),.999999)
        self.assertGreater(plan['planned_burn_s'],0.)

    def test_validation_rejects_v3_joint_optimization(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['burn']['joint_optimizer']['enabled']=True
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'settings.json'
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'joint_optimizer'):
                load_settings(path)
    def test_mock_no_orient_emits_no_turn_at_start(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=MockContext(td)
            cfg=copy.deepcopy(self.cfg)
            cfg['startup']['mode']='no_orient';cfg['startup']['body_direction_B']=[0,0,1]
            cfg['startup']['no_orient_body_selection']='fixed'
            cfg['startup']['candidate_burn_samples']=8;cfg['startup']['predict_step_s']=2.
            flight=Flight(ctx,ctx.model,cfg)
            s=ctx.state()
            # Change target by overriding own-state so thrust points at +Z.
            old_state=ctx.state
            def on_z_target():
                r=old_state();r['target_P_m']=[10000.,0.,400.];return r
            ctx.state=on_z_target
            aim=flight.execute_interception(ctx.state())
            names=[name for name,payload in ctx.events]
            self.assertIn('v4_alignment_skipped',names)
            self.assertIn('v4_burn_started',names)
            self.assertNotIn('v4_alignment_started',names)
            event=next(v for n,v in ctx.events if n=='v4_burn_started')
            np.testing.assert_allclose(event['request_B'],[0.,0.,.5],atol=1e-10)
            self.assertLessEqual(ctx.max_linear,.5+1e-8)
    def test_all_strategy_modes_use_public_bounded_requests(self):
        for mode in ('partial_turn','full_turn','rotate_and_thrust'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                ctx=MockContext(td)
                cfg=copy.deepcopy(self.cfg)
                cfg['startup']['mode']=mode
                cfg['startup']['body_direction_B']=[0,0,1]
                cfg['startup']['required_prediction_feasible']=False
                cfg['startup']['partial_fraction']=.5
                cfg['startup']['combined_turn_fraction']=1.
                cfg['startup']['candidate_burn_samples']=7
                cfg['startup']['predict_step_s']=2.
                cfg['startup']['online_burn']['enabled']=False
                cfg['attitude']['max_align_s']=38.
                flight=Flight(ctx,ctx.model,cfg)
                try:
                    flight.execute_interception(ctx.state())
                except FlightEnded as e:
                    self.assertEqual(e.data.get('reason'),'v4_alignment_timeout')
                self.assertLessEqual(ctx.max_angular,.05+1e-8)
                self.assertLessEqual(ctx.max_linear,.5+1e-8)
                stages=[n for n,_ in ctx.events]
                self.assertIn('v4_strategy_selected',stages)
                self.assertIn('v4_alignment_started',stages)
                if mode=='rotate_and_thrust':
                    overlap=[p for n,p in ctx.events if n=='v4_alignment_started'][0]
                    self.assertTrue(overlap['combined'])
                    self.assertGreater(np.linalg.norm(overlap['translational_request_B']),0.)

if __name__=='__main__':unittest.main()
