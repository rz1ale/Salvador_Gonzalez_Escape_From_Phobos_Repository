"""Offline regression based on publicly logged state near the 19.9-point V2 replan.
No simulator is launched, and no privileged telemetry is accessed.
"""
import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import Model, matrix
from policy.bangbang_coast_v2.core import TargetMotion, fuel_per_second, latent_from_reported, point_command
from policy.bangbang_coast_v2.adaptive import alignment_command
from policy.bangbang_coast_v2.coast_monitor import predict_ballistic
from policy.bangbang_coast_v2.flight import load_settings
from policy.bangbang_coast_v2.recovery import search_body_correction


class TestRecordedV2Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/'nominal_model.json')
        cls.cfg=load_settings(ROOT/'policy/bangbang_coast_v2/settings.json')

    def test_measured_62m_miss_can_be_corrected_without_turn(self):
        # 2026-11-18 V2 public state at sim ~76.036 s.
        s={'position_P_m':[-685.875464591219,113.95964366226454,-8508.86566795942],
           'velocity_P_m_s':[-1.1891094970703124,.5240937423706055,-7.903321533203125],
           'q_PB_xyzw_reported':[-.586126547146431,-.25310360501102686,.7324345474475368,.23650342400469945],
           'omega_relP_P_rad_s':[-.0057232556864619255,-.0021254585590213537,.001685440307483077],
           'fuel_used_kg':46.49637954989689,'damage_J':0.,
           'target_P_m':[-850.,120.,-9530.]}
        target=TargetMotion(s['target_P_m'],[0.,0.,0.],'fixed')
        baseline=predict_ballistic(s,self.model,target,315.,.75,35.)
        self.assertGreater(baseline['predicted_closest_m'],60.)
        action=search_body_correction(s,self.model,target,self.cfg,76.036)
        self.assertIsNotNone(action)
        self.assertEqual(action['kind'],'body_correction')
        self.assertLess(action['predicted_closest_m'],35.)
        self.assertLess(action['estimated_fuel_delta_kg'],3.)
        self.assertLessEqual(np.linalg.norm(action['request_B']),.5+1e-8)

    def test_revised_alignment_does_not_correct_roll_only(self):
        s={'q_PB_xyzw_reported':[0.,0.,0.,1.], 'omega_relP_P_rad_s':[.1,0.,0.]}
        a,theta,rate,settled=alignment_command(s,[1.,0.,0.],self.model,self.cfg)
        self.assertTrue(settled)
        np.testing.assert_allclose(a,0.)

    def test_alignment_controller_reduces_nominal_cost_for_recorded_turn(self):
        # Reuse the public alignment starting state and nominal target heading.
        s={'position_P_m':[-651.0,95.,-8239.], 'velocity_P_m_s':[0.,0.,0.],
           'q_PB_xyzw_reported':[.9848614817870476,.16855626602590104,-.03977804037301114,.007372542282663704],
           'omega_relP_P_rad_s':[0.,0.,0.],'fuel_used_kg':0.,'damage_J':0.}
        aim=np.array([-.16411683917504824,.02893425927726854,-.9860164662617289])
        def simulate(new):
            p,v,q,w,C=latent_from_reported(s,self.model)
            x=np.r_[p,v,q,w];f=0.;stable=0
            cfg=copy.deepcopy(self.cfg)
            cfg['attitude']['estimated_effective_ang_accel_rad_s2']=.07976902013670914
            for _ in range(380):
                R=matrix(x[6:10]);meas={'q_PB_xyzw_reported':x[6:10],
                      'omega_relP_P_rad_s':R.T@x[10:13]}
                if new:
                    cmd,_,_,settled=alignment_command(meas,aim,self.model,cfg)
                else:
                    cmd,_,_,settled=point_command(meas['q_PB_xyzw_reported'],meas['omega_relP_P_rad_s'],aim,self.model,cfg)
                stable=stable+1 if settled else 0
                if stable>=3: return f
                u=np.r_[np.zeros(3),cmd]
                f+=fuel_per_second(self.model,u)*.1
                x=self.model.step(x,u,.1)
            self.fail('Alignment controller failed to settle within 38s')
        old=simulate(False);new=simulate(True)
        self.assertLess(new,old)


if __name__=='__main__':unittest.main()
