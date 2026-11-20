"""Offline-only tests for V2 score-aware economy and event-triggered controls."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import Model
from policy.bangbang_coast_v3.adaptive import (
    BurnAttitudeHold, choose_attitude_profile, evaluate_burn_stop, pointing_geometry)
from policy.bangbang_coast_v3.core import TargetMotion
from policy.bangbang_coast_v3.flight import load_settings


def state(pos=(10000.,0.,0.),velocity=(0.,0.,0.),target=(10400.,0.,0.)):
    return {'read_begin_sim_s':5.,'read_end_sim_s':5.,'position_P_m':list(pos),
            'velocity_P_m_s':list(velocity),'q_PB_xyzw_reported':[0.,0.,0.,1.],
            'omega_relP_P_rad_s':[0.,0.,0.], 'target_P_m':list(target),
            'fuel_used_kg':2., 'damage_J':0.}


class TestAdaptive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/'nominal_model.json')
        cls.cfg=load_settings(ROOT/'policy/bangbang_coast_v3/settings.json')

    def test_v1_directory_not_needed_to_import_v2(self):
        from policy.bangbang_coast_v3 import adaptive,core,flight
        self.assertTrue(hasattr(flight,'run'))
        self.assertTrue(hasattr(adaptive,'evaluate_burn_stop'))
        self.assertTrue(hasattr(core,'select_plan'))

    def test_attitude_economy_no_false_gain_from_request_norm(self):
        s=state()
        p=choose_attitude_profile(s,[0.,1.,0.],self.model,self.cfg,220.)
        self.assertGreater(len(p['choices']),0)
        self.assertLessEqual(p['request_rad_s2'],.05)
        self.assertGreater(p['effective_alpha_rad_s2'],0)
        self.assertGreaterEqual(p['predicted_turn_fuel_kg'],0)
        self.assertLess(p['predicted_turn_s'],self.cfg['attitude']['max_align_s'])
        # All five candidate commands may collapse to the same physical response.
        if max(e['estimated_cost'] for e in p['choices'])-min(e['estimated_cost'] for e in p['choices'])<1e-3:
            self.assertAlmostEqual(p['request_rad_s2'],max(e['request_rad_s2'] for e in p['choices']))

    def test_burn_hold_zero_command_at_rest(self):
        controller=BurnAttitudeHold(self.cfg)
        s=state()
        for i in range(50):
            u,info=controller.command(s,[1.,0.,0.],self.model,self.cfg,i*.1)
            np.testing.assert_allclose(u,np.zeros(3))
            self.assertFalse(info['active'])
        self.assertEqual(controller.engagements,0)

    def test_burn_hold_triggers_without_roll(self):
        controller=BurnAttitudeHold(self.cfg)
        s=state();s['omega_relP_P_rad_s']=[0.,0.,.07]
        u,info=controller.command(s,[1.,0.,0.],self.model,self.cfg,1.)
        self.assertTrue(info['active'])
        self.assertGreater(np.linalg.norm(u),0)
        self.assertAlmostEqual(u[0],0.)
        self.assertLessEqual(np.linalg.norm(u),.04+1e-9)

    def test_online_forecast_causal_and_bounded(self):
        s=state(pos=(10000.,0.,0.),velocity=(3.,0.,0.),target=(10100.,0.,0.))
        goal=TargetMotion(s['target_P_m'],[0,0,0],'fixed')
        res=evaluate_burn_stop(s,self.model,goal,self.cfg,50.,12.)
        self.assertEqual(res['candidate_count'],self.cfg['burn']['online_stop']['candidate_samples'])
        self.assertTrue(res['coast_feasible_now'])
        self.assertIsNotNone(res['immediate_coast'])
        self.assertLess(res['immediate_coast']['predicted_closest_m'],38.)
        self.assertLessEqual(res['future_burn_window_s'],12.)

    def test_invalid_online_settings_rejected(self):
        s=json.loads(json.dumps(self.cfg));s['burn']['online_stop']['arrival_radius_m']=51
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'settings.json';p.write_text(json.dumps(s))
            with self.assertRaises(ValueError):load_settings(p)

if __name__=='__main__':unittest.main()
