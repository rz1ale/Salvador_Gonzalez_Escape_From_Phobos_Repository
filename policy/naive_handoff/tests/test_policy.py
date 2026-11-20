"""Offline physics/accounting and actual shared-runner loader checks."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
FOLDER=ROOT/"policy"/"naive_handoff"
from policy.naive_handoff.control import (Policy, matrix, increment, rotation_needed,
    direction_rotation, closest_point, load_settings, sha256)
from policy.naive_handoff.accounting import native_impulse_fuel, model_fuel_rate, score_estimate
from policy.naive_handoff.analyze import attempt_metrics, read_lines
from policy.naive_handoff.precompute import setup_problem, packet_from_latent
from system_identification.model import Model


class FakeContext:
    def __init__(self,output,*,fail_at=None):
        self.root=ROOT;self.model_path=ROOT/"nominal_model.json";self.output_dir=Path(output)
        self.parameters={};self.config={"entry":"policy/naive_handoff/entry.py"}
        self.assignment={"token":"offline_loader_test"}
        self.time=0.4;self.fail_at=fail_at;self.calls=[];self.events=[]
        model=Model.load(self.model_path)
        mission=json.loads((FOLDER/"mission.json").read_text())
        self.packet=packet_from_latent(model,np.asarray(mission["initial_state_latent"]),mission["target_center_P_m"])
    def clock(self):return self.time
    def checkpoint(self,**kwargs):
        if self.fail_at is not None and self.time>=self.fail_at:
            raise ConnectionError("synthetic backend disconnect")
    def state(self):
        p=copy.deepcopy(self.packet)
        p.update(read_begin_sim_s=self.time,read_end_sim_s=self.time,fuel_used_kg=self.time*.5)
        if self.time>=12.0:
            p["position_P_m"]=(np.asarray(p["target_P_m"])+[49.,0,0]).tolist()
        return p
    def command(self,a,alpha):
        u=np.r_[a,alpha];self.calls.append((self.time,u))
        row=dict(send_begin_sim_s=self.time,send_end_sim_s=self.time,
            linear_body_request_m_s2=np.asarray(a).tolist(),angular_body_request_rad_s2=np.asarray(alpha).tolist())
        with (self.output_dir/"commands.txt").open("a") as f:f.write(json.dumps(row)+"\n")
        return row
    def event(self,name,**kw):self.events.append((name,kw))
    def sleep(self,seconds):self.time+=max(seconds,.01)
    def finish(self,**value):
        from sim_core.runtime import Finish
        raise Finish(value)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/"nominal_model.json")
        cls.mission=json.loads((FOLDER/"mission.json").read_text())
    def test_quaternion_physical_sign_and_frame(self):
        q=np.array([0.,0.,0.,1.])
        rotated=increment(q,np.array([0.,0.,np.pi/2]))
        np.testing.assert_allclose(matrix(rotated).T@[1.,0,0],[0,1,0],atol=1e-12)
        np.testing.assert_allclose(rotation_needed(q,rotated),[0,0,np.pi/2],atol=1e-12)
        np.testing.assert_allclose(rotation_needed(q,-rotated),[0,0,np.pi/2],atol=1e-12)
    def test_direction_slew_points_main_axis(self):
        q=np.asarray(self.mission["initial_state_latent"])[6:10]
        d=np.array([-.2,.1,-1.]);d/=np.linalg.norm(d)
        axis,angle=direction_rotation(matrix(q)@d)
        out=increment(q,axis*angle)
        np.testing.assert_allclose(matrix(out).T@[1.,0,0],d,atol=1e-12)
    def test_closest_target_sphere(self):
        p=np.array([0.,0,0]);c=np.array([100.,0,0])
        np.testing.assert_allclose(closest_point(p,c,50),[50,0,0])
    def test_native_fuel_exact_reference_loop(self):
        forces=np.array([100.,140.,180.]);lam=np.array([.03,.07,0.])
        for dt in (.01,.07,.2):
            expected=sum(F*(max(t,0)/max(lam))*min(dt,max(lam))/(80*9.8) for F,t in zip(forces,lam))
            self.assertAlmostEqual(native_impulse_fuel(forces,lam,dt),expected,places=14)
        self.assertEqual(native_impulse_fuel(forces,np.zeros(3),.1),0)
    def test_pure_axis_knees_deliver_saturated_authority(self):
        manifest=json.loads((FOLDER/"assets/manifest.json").read_text())
        for j,key in ((0,"forward_request_at_saturation_knee"),(2,"boost_request_at_saturation_knee")):
            u=np.eye(6)[j]*manifest[key]
            high=np.eye(6)[j]*10.
            np.testing.assert_allclose(self.model.mean_acceleration(u),self.model.mean_acceleration(high),rtol=1e-4,atol=1e-7)
            self.assertAlmostEqual(model_fuel_rate(self.model,u),model_fuel_rate(self.model,high),places=5)
    def test_score_nominal_and_unclamped(self):
        self.assertEqual(score_estimate(120,75),50.)
        self.assertGreater(score_estimate(60,0),100.)
    def test_stage_timing_frames_and_angular_feedback(self):
        policy=Policy(FOLDER,ROOT/"nominal_model.json")
        x=np.asarray(self.mission["initial_state_latent"])
        p=packet_from_latent(self.model,x,self.mission["target_center_P_m"])
        a,d=policy.command(0,p);self.assertEqual(d["stage"],"boost");self.assertGreater(a[2],0)
        t=policy.settings["boost_duration_s"]
        a,d=policy.command(t,p);self.assertEqual(d["stage"],"slew")
        end=t+policy.settings["slew_duration_s"]
        a,d=policy.command(end,p);self.assertEqual(d["stage"],"forward");self.assertGreater(a[0],0)
        # A 90-degree orientation change must rotate WORLD angular-rate reports.
        p["q_PB_xyzw_reported"]=increment(np.array([0.,0,0,1.]),[0,0,np.pi/2]).tolist()
        p["omega_relP_P_rad_s"]=[1.,0,0]
        _,d=policy.command(end,p)
        expected=matrix(p["q_PB_xyzw_reported"])@np.array([1.,0,0])-policy.rate_bias
        np.testing.assert_allclose(d["omega_B_rad_s"],expected)
    def test_full_thrust_common_scaling_preserves_translation_rotation_ratio(self):
        policy=Policy(FOLDER,ROOT/"nominal_model.json")
        x=np.asarray(self.mission["initial_state_latent"])
        p=packet_from_latent(self.model,x,self.mission["target_center_P_m"])
        u,detail=policy.command(0,p)
        self.assertAlmostEqual(u[2],self.model.allocator.caps[0])
        t=policy.settings["boost_duration_s"]+policy.settings["slew_duration_s"]
        u,detail=policy.command(t,p)
        small=np.r_[policy.x_knee,0.,0.,policy.input_map@np.asarray(detail["feedback_alpha_B_rad_s2"])]
        nonzero=abs(small)>1e-10
        ratios=u[nonzero]/small[nonzero]
        np.testing.assert_allclose(ratios,np.full(len(ratios),ratios[0]),rtol=1e-12)
        np.testing.assert_allclose(self.model.mean_acceleration(u),self.model.mean_acceleration(small),rtol=1e-4,atol=1e-7)
    def test_stale_settings_require_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            from shutil import copytree
            folder=Path(temp)/"policy";copytree(FOLDER,folder,ignore=lambda d,n:[x for x in n if x in ("run_logs","analysis_results","__pycache__")])
            settings=load_settings(folder/"settings.json");settings["boost_duration_s"]=1.0
            (folder/"settings.json").write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError,"precompute"):
                Policy(folder,ROOT/"nominal_model.json")
    def test_convex_setup_stationarity_and_short_time_is_not_extended(self):
        s=load_settings(FOLDER/"settings.json")
        _,_,_,info=setup_problem(self.model,self.mission["initial_state_latent"],self.mission["target_center_P_m"],s)
        self.assertTrue(info["success"]);self.assertLess(info["projected_gradient_inf"],1e-4)
        s["slew_duration_s"]=5.;s["slew_steps"]=50
        t,x,_,info=setup_problem(self.model,self.mission["initial_state_latent"],self.mission["target_center_P_m"],s)
        self.assertEqual(t[-1],5.)
        self.assertGreater(abs(info["ideal_terminal_pointing_error_deg"]),5.)
    def test_entry_uses_actual_runner_loader_and_does_not_stop_native(self):
        from sim_core.runtime import execute_entry,Finish
        with tempfile.TemporaryDirectory() as temp:
            context=FakeContext(temp)
            with patch("builtins.print"):
                with self.assertRaises(Finish) as stop:
                    execute_entry(context,{})
            self.assertEqual(stop.exception.value["reason"],"observed_arrival")
            self.assertIs(stop.exception.value["stop_sim"],False)
            np.testing.assert_array_equal(context.calls[-1][1],np.zeros(6))
            stages=[d["stage"] for n,d in context.events if n=="naive_handoff_stage"]
            self.assertEqual(stages,["boost","slew","forward"])
            metrics,rows=attempt_metrics(temp)
            self.assertTrue(metrics["observed_arrival"])
            self.assertIsNotNone(metrics["score_formula_estimate"])
            self.assertEqual(len(metrics["stages"]),3)
            self.assertAlmostEqual(sum(s["public_fuel_increment_kg"] for s in metrics["stages"]),metrics["public_fuel_increment_kg"])
    def test_disconnect_has_partial_accounting_without_a_winning_score(self):
        from sim_core.runtime import execute_entry
        with tempfile.TemporaryDirectory() as temp:
            context=FakeContext(temp,fail_at=3.)
            with patch("builtins.print"),self.assertRaises(ConnectionError):
                execute_entry(context,{})
            metrics,_=attempt_metrics(temp)
            self.assertFalse(metrics["observed_arrival"])
            self.assertIsNone(metrics["score_formula_estimate"])
            self.assertGreater(metrics["public_fuel_increment_kg"],0)
            np.testing.assert_array_equal(context.calls[-1][1],np.zeros(6))
    def test_entry_import_has_no_scipy_or_simulator(self):
        code="import sys; import policy.naive_handoff.entry; assert 'scipy' not in sys.modules; assert 'spaceteams' not in sys.modules"
        result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_run_plan_is_valid_and_dependencies_cover_sources(self):
        from sim_core.common import validate
        c=validate(ROOT,json.loads((FOLDER/"run_plan.json").read_text()))
        self.assertEqual(c["entry"],"policy/naive_handoff/entry.py")
        self.assertIsNone(c["limits"]["task_sim_s"])
        self.assertIsNone(c["limits"]["attempt_wall_s"])


if __name__=="__main__":
    unittest.main()
