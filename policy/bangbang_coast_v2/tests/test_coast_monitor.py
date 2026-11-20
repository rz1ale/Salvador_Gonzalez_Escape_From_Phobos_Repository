"""Miss-monitor tests: safe coast stays zero thrust, predicted miss can replan."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_identification.model import Model
from policy.bangbang_coast_v2.flight import Flight,load_settings
from policy.bangbang_coast_v2.coast_monitor import CoastMissMonitor,predict_ballistic
from policy.bangbang_coast_v2.core import TargetMotion
from test_integration_mock import MockContext,FlightEnded

class TracingContext(MockContext):
    def __init__(self,folder):
        super().__init__(folder)
        self.command_history=[]
    def command(self,a,alpha):
        super().command(a,alpha)
        self.command_history.append(self.u.copy())

class TestCoastMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/'nominal_model.json')
        cls.settings=load_settings(ROOT/'policy/bangbang_coast_v2/settings.json')

    def test_unpowered_interception(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=TracingContext(folder)
            ctx.x[:3]=[10000.,330.,0.]
            ctx.x[3:6]=[0.,1.9,0.]
            flight=Flight(ctx,ctx.model,copy.deepcopy(self.settings))
            target=TargetMotion([10000.,400.,0.],[0,0,0],'fixed')
            data=predict_ballistic(ctx.state(),ctx.model,target,120.,.75,43.)
            self.assertTrue(data['predicted_capture'],data)
            self.assertLess(data['predicted_closest_m'],43.)
            m=CoastMissMonitor(self.settings['coast']['miss_monitor'])
            for _ in range(10):
                result=m.update(data,True)
            self.assertFalse(result['trigger_replan'])
            with self.assertRaises(FlightEnded) as outcome:
                flight.coast(np.array([0.,1.,0.]))
            self.assertEqual(outcome.exception.data['reason'],'bangbang_observed_arrival')
            self.assertFalse(any(name=='coast_reinterception_trigger' for name,_ in ctx.events))
            self.assertTrue(ctx.command_history)
            self.assertTrue(all(np.linalg.norm(u)==0. for u in ctx.command_history))

    def test_predicted_miss_starts_replan_while_coasting_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=TracingContext(folder)
            cfg=copy.deepcopy(self.settings)
            cfg['coast']['miss_monitor']['entry_grace_s']=0.
            cfg['coast']['miss_monitor']['confirmations_required']=2
            cfg['coast']['miss_monitor']['check_period_s']=.5
            cfg['coast']['miss_monitor']['attempt_cooldown_s']=0.
            flight=Flight(ctx,ctx.model,cfg)
            plan=flight.coast(np.array([0.,1.,0.]))
            self.assertIsInstance(plan,dict)
            self.assertIn(plan['kind'], ('body_correction','turn_reintercept'))
            self.assertTrue(any(n in ('coast_reinterception_trigger','coast_direct_correction_selected')
                                for n,_ in ctx.events))
            self.assertTrue(all(np.linalg.norm(u)==0. for u in ctx.command_history))
            self.assertEqual(flight.reinterceptions,0) # incremented only on execution

    def test_full_reinterception_after_coast_velocity_perturbation(self):
        # Deliberately inject an external velocity perturbation at coast entry.
        # This tests the state machine; it is NOT native simulator validation.
        class PerturbedContext(TracingContext):
            def __init__(self,folder):
                super().__init__(folder)
                self.injected=False
                self.coast_entries=0
            def event(self,name,**kwargs):
                super().event(name,**kwargs)
                if name=='coast_started':
                    self.coast_entries+=1
                    if not self.injected:
                        self.x[3]+=2.5
                        self.injected=True
        from policy.bangbang_coast_v2.flight import run
        with tempfile.TemporaryDirectory() as folder:
            ctx=PerturbedContext(folder)
            with self.assertRaises(FlightEnded) as outcome:
                run(ctx)
            self.assertEqual(outcome.exception.data['reason'],'bangbang_observed_arrival')
            stages=[name for name,_ in ctx.events]
            self.assertTrue('coast_direct_correction_selected' in stages or
                            'coast_reinterception_trigger' in stages)
            self.assertTrue('body_correction_completed' in stages or
                            'reinterception_plan' in stages)
            self.assertGreaterEqual(ctx.coast_entries,2)
            self.assertLess(ctx.max_linear,0.5000001)
            self.assertLess(ctx.max_angular,0.0500001)

    def test_no_miss_trigger_when_prediction_unresolved(self):
        c={'trigger_closest_m':48.,'confirmations_required':2}
        monitor=CoastMissMonitor(c)
        prediction=dict(predicted_capture=False,predicted_closest_m=100.,
                        still_closing_at_horizon=True)
        for _ in range(5):
            self.assertFalse(monitor.update(prediction,False)['trigger_replan'])
        self.assertTrue(monitor.update(prediction,True)['unsafe'])
        self.assertTrue(monitor.update(prediction,True)['trigger_replan'])

if __name__=='__main__': unittest.main()
