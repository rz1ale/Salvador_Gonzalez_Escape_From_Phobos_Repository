"""Mock integration through run(context) on the PUBLIC model; no STPro SDK."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import Model, matrix
from policy.bangbang_coast_v4.flight import run
from policy.bangbang_coast_v4.core import fuel_per_second

class FlightEnded(BaseException):
    def __init__(self,data):self.data=data

class MockContext:
    def __init__(self,folder):
        self.root=ROOT
        self.model_path=ROOT/'nominal_model.json'
        self.output_dir=Path(folder)
        self.parameters={}
        self.model=Model.load(self.model_path)
        self.t=0.
        self.x=np.r_[[10000.,0.,0.], np.zeros(3), [0.,0.,0.,1.], np.zeros(3)]
        self.u=np.zeros(6)
        self.fuel=0.
        self.max_linear=0.;self.max_angular=0.
        self.events=[]

    def clock(self):return self.t
    def state(self):
        C=matrix(self.x[6:10]);m=self.model
        return {'read_begin_sim_s':self.t,'read_end_sim_s':self.t,
                'position_P_m':(self.x[:3]+C.T@m.ell).tolist(),
                'velocity_P_m_s':(self.x[3:6]+m.vbias).tolist(),
                'q_PB_xyzw_reported':self.x[6:10].tolist(),
                'omega_relP_P_rad_s':(C.T@self.x[10:13]).tolist(),
                'target_P_m':[10000.,400.,0.],
                'fuel_used_kg':self.fuel,'damage_J':0.}
    def radar(self,count):return []
    def command(self,a,alpha):
        self.u=np.r_[a,alpha]
        self.max_linear=max(self.max_linear,float(np.linalg.norm(a)))
        self.max_angular=max(self.max_angular,float(np.linalg.norm(alpha)))
        if self.max_linear>.5000001 or self.max_angular>.0500001:
            raise AssertionError('Public request cap exceeded')
    def sleep(self,dt):
        self.x=self.model.step(self.x,self.u,dt)
        self.t+=dt
        self.fuel+=fuel_per_second(self.model,self.u)*dt   # same nominal allocator fuel model as policy
    def event(self,name,**details):self.events.append((name,details))
    def finish(self,**kwargs):raise FlightEnded(kwargs)

class TestMockFlight(unittest.TestCase):
    def test_end_to_end_quarter_turn_and_coast(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=MockContext(folder)
            try:
                run(ctx)
            except FlightEnded as exc:
                result=exc.data
            else:
                self.fail('No policy termination')
            self.assertEqual(result['reason'],'bangbang_observed_arrival',str(result))
            self.assertLess(result['distance_m'],50.)
            self.assertTrue(0<ctx.max_linear<=.5+1e-9)
            self.assertTrue(0<=ctx.max_angular<=.05+1e-9)
            stages=[name for name,_ in ctx.events]
            self.assertTrue('v4_alignment_skipped' in stages or 'v4_alignment_completed' in stages)
            self.assertIn('v4_burn_started',stages)
            self.assertIn('v4_burn_ended',stages)
            self.assertIn('coast_started',stages)
            burn=[details for name,details in ctx.events if name=='v4_burn_ended'][-1]
            self.assertGreater(burn['online_checks'],0)
            self.assertGreaterEqual(burn['hold_engagements'],0)
            self.assertLess(burn['hold_max_error_deg'],10.0)
            coast_index=stages.index('coast_started')
            self.assertTrue(all(name!='trim_update' for name,_ in ctx.events[coast_index:]))


if __name__=='__main__':unittest.main()
