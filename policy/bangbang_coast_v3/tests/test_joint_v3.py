"""V3-only tests. They do not start or modify the official simulator."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import Model
from policy.bangbang_coast_v3.joint import first_sphere_entry,search_joint_burn
from policy.bangbang_coast_v3.core import TargetMotion,run_candidates
from policy.bangbang_coast_v3.flight import Flight,load_settings
from policy.bangbang_coast_v3.tests.test_integration_mock import MockContext

class TestJointV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/'nominal_model.json')
        cls.cfg=load_settings(ROOT/'policy/bangbang_coast_v3/settings.json')

    def test_segment_entry_outside_to_outside(self):
        p=np.array([[-10.,0,0],[10.,0,0],[10.,4.,0],[0.,0.,0.]])
        q=np.array([[10.,0,0],[12.,0,0],[12.,4.,0],[0.,0.,0.]])
        x=first_sphere_entry(p,q,5.)
        self.assertAlmostEqual(x[0],.25,places=10)
        self.assertTrue(np.isnan(x[1]))
        self.assertTrue(np.isnan(x[2]))
        self.assertEqual(x[3],0.)

    def test_terminal_capture_is_region_not_center(self):
        # Positive x motion, stationary target: first entry should be on a
        # sphere, not a prescribed terminal position or zero-speed endpoint.
        state={'position_P_m':[10000.,0.,0.],
               'velocity_P_m_s':[0.,5.,0.],
               'q_PB_xyzw_reported':[0.,0.,0.,1.],
               'omega_relP_P_rad_s':[0.,0.,0.],
               'fuel_used_kg':0.,'damage_J':0.,
               'target_P_m':[10000.,180.,0.]}
        cfg=copy.deepcopy(self.cfg)
        cfg['burn']['joint_optimizer']['prediction_horizon_s']=80.
        cfg['burn']['joint_optimizer']['capture_radius_m']=42.
        cfg['burn']['joint_optimizer']['integration_step_s']=.4
        result=search_joint_burn(state,self.model,TargetMotion(state['target_P_m'],[0,0,0],'fixed'),cfg,10.,6.)
        best=result['best']
        self.assertTrue(best['feasible'],result)
        d=np.linalg.norm(np.asarray(best['predicted_entry_position_P_m'])-np.asarray(state['target_P_m']))
        self.assertAlmostEqual(d,42.,delta=.75)
        self.assertGreater(d,40.)
        self.assertGreater(best['predicted_capture_sim_s'],10.)

    def test_recorded_native_state_joint_search(self):
        # One own-state snapshot from public V2 Update 2 telemetry, no debris truth.
        state={
            'read_begin_sim_s':59.976958,
            'position_P_m':[-677.2435498597199,96.52750199803064,-8385.35638558876],
            'velocity_P_m_s':[-1.0871379089355468,0.10643956184387207,-6.494447021484375],
            'q_PB_xyzw_reported':[-0.6574985796191376,-0.161209310212864,
                                   0.7271067454430928,0.11411817046668199],
            'omega_relP_P_rad_s':[-0.0005290939006954432,-0.0001003007055260241,
                                   5.3831186960451305e-05],
            'fuel_used_kg':37.491074997758794,
            'damage_J':0.,
            'target_P_m':[-850.,120.,-9530.]
        }
        t=state['read_begin_sim_s']
        result=search_joint_burn(state,self.model,
            TargetMotion(state['target_P_m'],np.zeros(3),'fixed'),dict(self.cfg,_start_sim=0.),t,18.)
        self.assertGreater(result['n_candidates'],100)
        self.assertTrue(result['best']['feasible'])
        point=np.asarray(result['best']['predicted_entry_position_P_m'])
        self.assertAlmostEqual(np.linalg.norm(point-np.asarray(state['target_P_m'])),42.,delta=1.)
        self.assertLessEqual(np.linalg.norm(result['best']['lateral_request_B']),.5+1e-9)

    def test_no_terminal_velocity_penalty_in_any_objective(self):
        q=copy.deepcopy(self.cfg)
        q['objective']['quadratic']['Q_terminal_velocity']=10.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'settings.json'
            path.write_text(json.dumps(q))
            with self.assertRaisesRegex(ValueError,'terminal translational velocity'):
                load_settings(path)
        self.assertEqual(self.cfg['objective']['quadratic']['Q_terminal_velocity'],0.)

    def test_live_burn_can_issue_bounded_joint_pulse(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=MockContext(folder)
            cfg=copy.deepcopy(self.cfg)
            cfg['burn']['joint_optimizer']['min_burn_elapsed_s']=0.
            cfg['burn']['joint_optimizer']['max_lateral_episodes_per_burn']=1
            cfg['burn']['joint_optimizer']['check_period_s']=.3
            cfg['burn']['online_stop']['min_burn_elapsed_s']=0.
            cfg['burn']['online_stop']['max_extension_s']=0.
            cfg['burn']['online_stop']['enforce_max_burn_s']=3.
            p={'future_burn_s':2.,'lateral_request_B':[0.,.15,0.],
               'lateral_pulse_s':.8,'lateral_start_s':0.,
               'predicted_capture_sim_s':100.,'predicted_score':60.,
               'predicted_total_fuel_kg':1.,'feasible':True,'objective':-60.}
            prediction={'best':p,'best_pure':None,
                        'immediate_coast':{'feasible':False},'n_candidates':1,'n_feasible':1}
            flight=Flight(ctx,ctx.model,cfg)
            all_commands=[]
            original=ctx.command
            def recording(a,alpha):
                all_commands.append((np.asarray(a).copy(),np.asarray(alpha).copy()))
                original(a,alpha)
            ctx.command=recording
            with patch('policy.bangbang_coast_v3.flight.search_joint_burn',return_value=prediction):
                flight.burn(np.array([1.,0.,0.]),2.)
            nonzero=[c for c,_ in all_commands if c[1]>.01]
            self.assertTrue(nonzero,'Joint lateral pulse was never issued')
            self.assertTrue(all(np.linalg.norm(c)<=.5000001 for c,a in all_commands))
            starts=[name for name,payload in ctx.events if name=='joint_lateral_started']
            self.assertEqual(len(starts),1)

if __name__=='__main__':unittest.main()
