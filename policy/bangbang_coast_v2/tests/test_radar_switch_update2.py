"""Regression tests for V2 Update 2 using the provided public impact-flight telemetry."""
from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import matrix
from policy.bangbang_coast_v2.core import radar_lateral_body
from policy.bangbang_coast_v2.adaptive import AlignmentSwitchGuard
from policy.bangbang_coast_v2.flight import load_settings

class TestUpdate2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_settings(ROOT/'policy/bangbang_coast_v2/settings.json')

    def test_recorded_rear_threat_is_not_discarded(self):
        # 2026-11-18 impact flight: radar packet 35.146436 s, raw public row.
        # This is a candidate threat, not independently identified as the hit object.
        row=np.array([[-9.50,-27.70,23.15],[22.60,-4.15,27.52],[34.204646,0,0]])
        q=np.array([-.6219206775172993,-.15484816968425555,.7576780213114591,.12312892123060198])
        v=np.array([-.6047449111938477,-.010329297780990601,-3.023448791503906])
        req,info=radar_lateral_body([row],matrix(q),35.146436,self.cfg,v)
        self.assertEqual(info['hazards'],1,info)
        self.assertEqual(info['rear_hazards'],1,info)
        self.assertTrue(info['emergency'])
        self.assertLess(info['min_margin_m'],0)
        self.assertLess(info['selected_body_x_m'],0)
        self.assertAlmostEqual(np.linalg.norm(req),self.cfg['avoidance']['emergency_lateral_request_m_s2'],places=5)
        self.assertAlmostEqual(req[0],0.)

    def test_outward_and_stale_objects_do_not_trigger(self):
        row=np.array([[-15.,2.,1.],[4.,+3.,10.],[4.,0,0]])
        req,info=radar_lateral_body([row],np.eye(3),4.,self.cfg,[0,0,0])
        self.assertEqual(info['hazards'],0)
        np.testing.assert_allclose(req,0)
        row[1,1]=-4.
        req,info=radar_lateral_body([row],np.eye(3),10.,self.cfg,[0,0,0])
        self.assertEqual(info['hazards'],0)
        self.assertEqual(info['stale'],1)

    def test_distant_crossing_object_does_not_force_dodge(self):
        row=np.array([[20.,120.,10.],[118.,-1.,10.],[8.,0,0]])
        req,info=radar_lateral_body([row],np.eye(3),8.,self.cfg,[-8,0,0])
        self.assertEqual(info['hazards'],0,info)
        np.testing.assert_allclose(req,0)

    def test_alignment_short_reversals_blocked_but_braking_allowed(self):
        g=AlignmentSwitchGuard(self.cfg)
        first=np.array([0.,0.,.05])
        reverse=-first
        np.testing.assert_allclose(g.filter(first,0.),first)
        np.testing.assert_allclose(g.filter(reverse,.10),0.)
        np.testing.assert_allclose(g.filter(reverse,.35),0.)
        np.testing.assert_allclose(g.filter(reverse,.50),reverse)
        self.assertEqual(g.inhibited,2)
        self.assertEqual(g.opposed_accepted,1)
        np.testing.assert_allclose(g.filter(first,.55),0.)
        np.testing.assert_allclose(g.filter(first,1.00),first)
        np.testing.assert_allclose(g.filter(first,1.10,settled=True),0.)

    def test_alignment_config_rejects_negative_dwell(self):
        import tempfile,json
        data=copy.deepcopy(self.cfg)
        data['attitude']['alignment_control']['minimum_reverse_dwell_s']=-1
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'settings.json';p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_settings(p)

if __name__=='__main__':unittest.main()
