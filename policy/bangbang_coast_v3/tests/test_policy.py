"""Offline mathematics and package contract; does not launch any native simulation."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import Model, matrix
from policy.bangbang_coast_v3.core import (
    TargetMotion, aim_base, cap, coast_trim, estimate_target_velocity, fuel_per_second,
    latent_from_reported, linearized_qr_gain, point_command, radar_lateral_body, select_plan,
    axes_for_direction)
from policy.bangbang_coast_v3.flight import load_settings


def example_state():
    return {'read_begin_sim_s':3.,'read_end_sim_s':3.,
            'position_P_m':[10000.,0.,0.], 'velocity_P_m_s':[0.,0.,0.],
            'q_PB_xyzw_reported':[0.,0.,0.,1.],
            'omega_relP_P_rad_s':[0.,0.,0.],
            'target_P_m':[10400.,0.,0.],
            'fuel_used_kg':0.,'damage_J':0.}


class TestBangBang(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_settings(ROOT/'policy/bangbang_coast_v3/settings.json')
        cls.model=Model.load(ROOT/'nominal_model.json')

    def test_model_contract(self):
        self.assertEqual(self.model.n_dim,13)
        self.assertEqual(self.model.m_dim,6)
        self.assertEqual(self.model.allocator.A.shape,(38,6))

    def test_inverse_actuator_nominal(self):
        a=self.model.mean_acceleration([.5,0,0,0,0,0])
        self.assertTrue(.1<a[0]<.3)
        self.assertAlmostEqual(fuel_per_second(self.model,[0]*6),0.)
        self.assertGreater(fuel_per_second(self.model,[.5,0,0,0,0,0]),0)

    def test_target_motion_from_public_positions(self):
        v=estimate_target_velocity([(0,[5.,0.,0.]),(1,[6.,2.,0.]),(2,[7.,4.,0.])],30.)
        np.testing.assert_allclose(v,[1.,2.,0.],rtol=0,atol=1e-8)
        t=TargetMotion([1,2,3],v,'linear')
        np.testing.assert_allclose(t.at(2),[3,6,3],atol=1e-8)

    def test_orientation_frame(self):
        C=axes_for_direction([0,1,0])
        np.testing.assert_allclose(C.T@C,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(C[:,0],[0,1,0],atol=1e-12)
        command,theta,_,ok=point_command([0,0,0,1],[0,0,0],[0,1,0],self.model,self.cfg)
        self.assertFalse(ok)
        self.assertGreater(command[2],0)
        self.assertAlmostEqual(theta,np.pi/2)

    def test_linearized_quadratic_gain(self):
        K=linearized_qr_gain([1e4,0,0],[1,0,0],self.model,self.cfg)
        self.assertEqual(K.shape,(3,6))
        self.assertTrue(np.isfinite(K).all())

    def test_radar_body_transform(self):
        row=np.array([[50.,1.,0.],[8.,-3.,10.],[4.,0.,0.]])
        req,info=radar_lateral_body([row],np.eye(3),4.,self.cfg)
        self.assertGreater(info['hazards'],0)
        self.assertLess(req[1],0)
        self.assertLessEqual(np.linalg.norm(req),.5+1e-10)
        req2,info2=radar_lateral_body([row],np.eye(3),40.,self.cfg)
        self.assertEqual(info2['hazards'],0)

    def test_planner_score_and_quadratic(self):
        state=example_state();target=TargetMotion(state['target_P_m'],[0,0,0],'fixed')
        for objective in ('score','quadratic'):
            cfg=json.loads(json.dumps(self.cfg));cfg['objective']['mode']=objective
            result=select_plan(state,self.model,target,cfg,4.)
            self.assertTrue(result['feasible'],str(result))
            self.assertLessEqual(result['predicted_closest_m'],cfg['planning']['arrival_radius_m'])
            self.assertTrue(2<=result['burn_s']<=70)
            self.assertTrue(np.isfinite(result['estimated_score']))

    def test_refit(self):
        state=example_state();target=TargetMotion(state['target_P_m'],[0,0,0],'fixed')
        result=select_plan(state,self.model,target,self.cfg,4.,force_direction=[1.,0.,0.],known_aligned=True)
        self.assertTrue(result['feasible'])
        self.assertAlmostEqual(result['estimated_align_s'],0.)

    def test_settings_and_runner_plan(self):
        from sim_core.common import validate
        plan=json.loads((ROOT/'policy/bangbang_coast_v3/run_plan.json').read_text())
        checked=validate(ROOT,plan)
        self.assertEqual(checked['entry'],'policy/bangbang_coast_v3/entry.py')
        self.assertEqual(checked['model'],'nominal_model.json')
        self.assertEqual(checked['mode'],'manual')

if __name__=='__main__':
    unittest.main()
