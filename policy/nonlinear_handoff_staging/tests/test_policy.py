"""Model/geometry/accounting and existing-runner loader checks; no native sim."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3];FOLDER=ROOT/"policy/nonlinear_handoff_staging"
from policy.nonlinear_handoff_staging.math_core import (matrix,increment,error,retract,settings,signature,mission_reference,terminal_limits,reported)
from policy.nonlinear_handoff_staging.flight import Policy,packet_row,causal_state
from policy.nonlinear_handoff_staging.precompute import tangent_maps,validate_replay
from policy.nonlinear_handoff_staging.accounting import native_impulse_fuel,model_fuel_rate,score_estimate
from policy.nonlinear_handoff_staging.analyze import metrics
from system_identification.model import Model


class EndFixture(Exception):pass


class NumericContext:
    """Public packet emulation using original model including report/input delays."""
    def __init__(self,output,*,perturb=False):
        self.root=ROOT;self.model_path=ROOT/"nominal_model.json";self.output_dir=Path(output)
        self.parameters={};self.assignment={"token":"numeric_loader_test"};self.config={"entry":"policy/nonlinear_handoff_staging/entry.py"}
        self.model=Model.load(self.model_path);mi=json.loads((FOLDER/"mission.json").read_text())
        self.target=np.asarray(mi["target_center_P_m"]);self.x=np.asarray(mi["initial_state_latent"],float)
        if perturb:self.x=retract(self.x,np.r_[[.1,-.1,.1],[0,0,0],[0,.025,.01],[0,.002,0]])
        self.time=.4;self.x=self.model.rollout(self.x,0,np.array([self.time]),np.array([]),np.zeros((0,6)))[-1]
        self.ts=[self.time];self.xs=[self.x.copy()];self.ct=[];self.us=[];self.calls=[];self.events=[];self.fuel=0.
        self.period=settings(FOLDER/"settings.json")["feedback"]["period_s"]
        self.end=self.time+settings(FOLDER/"settings.json")["design"]["duration_s"]+2.
    def clock(self):return self.time
    def checkpoint(self,**kwargs):
        if self.time>=self.end:raise EndFixture("numeric interval complete")
    def state(self):
        p=self.model.p;hist=np.asarray(self.xs);q=self.x[6:10]
        values=[]
        for sl,delay,bias in ((slice(3,6),p["velocity_report_delay_s"],self.model.vbias),
                (slice(10,13),p["rate_report_delay_s"],self.model.wbias)):
            values.append(np.array([np.interp(self.time-delay,self.ts,hist[:,j]) for j in range(sl.start,sl.stop)])+bias)
        return dict(read_begin_sim_s=self.time,read_end_sim_s=self.time,position_P_m=reported(self.x,self.model.definition).tolist(),
            velocity_P_m_s=values[0].tolist(),q_PB_xyzw_reported=q.tolist(),omega_relP_P_rad_s=(matrix(q).T@values[1]).tolist(),
            target_P_m=self.target.tolist(),fuel_used_kg=self.fuel,damage_J=0.)
    def command(self,a,alpha):
        u=np.r_[a,alpha];self.ct.append(self.time);self.us.append(u);self.calls.append(u)
        r=dict(send_begin_sim_s=self.time,send_end_sim_s=self.time,linear_body_request_m_s2=np.asarray(a).tolist(),angular_body_request_rad_s2=np.asarray(alpha).tolist())
        with (self.output_dir/"commands.txt").open("a") as f:f.write(json.dumps(r)+"\n")
        return r
    def sleep(self,seconds):
        if self.us:self.fuel+=model_fuel_rate(self.model,self.us[-1])*self.period
        self.x=self.model.rollout(self.x,self.time,np.array([self.time+self.period]),np.asarray(self.ct),np.asarray(self.us).reshape(-1,6))[-1]
        self.time+=self.period;self.ts.append(self.time);self.xs.append(self.x.copy())
    def event(self,name,**values):self.events.append((name,values))
    def finish(self,**values):
        from sim_core.runtime import Finish
        raise Finish(values)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/"nominal_model.json");cls.s=settings(FOLDER/"settings.json")
        cls.mi=json.loads((FOLDER/"mission.json").read_text());cls.x0=np.asarray(cls.mi["initial_state_latent"])
        cls.ref=mission_reference(cls.x0,cls.mi["target_center_P_m"],cls.model.definition,cls.s["design"])
    def test_quaternion_sign_and_retraction(self):
        d=np.r_[np.zeros(6),[.001,-.002,.003],np.zeros(3)]
        np.testing.assert_allclose(error(retract(self.x0,d),self.x0),d,atol=1e-11)
        x=self.x0.copy();x[6:10]*=-1;np.testing.assert_allclose(error(x,self.x0),0,atol=1e-12)
    def test_tangent_projection_is_inverse(self):
        E,G=tangent_maps(self.x0[6:10]);np.testing.assert_allclose(G@E,np.eye(12),atol=1e-12)
    def test_reference_and_pointing(self):
        ref=self.ref
        self.assertAlmostEqual(np.linalg.norm(ref["pd"]-ref["p0"]),self.s["design"]["distance_m"])
        self.assertAlmostEqual(np.linalg.norm(ref["xf"]-self.mi["target_center_P_m"]),50.)
        np.testing.assert_allclose(matrix(ref["qd"]).T@[1.,0,0],ref["n"],atol=1e-12)
    def test_two_percent_has_defined_scale_and_rate_is_absolute(self):
        limits=terminal_limits(self.s,self.ref)
        self.assertAlmostEqual(limits["pointing_deg"],.02*np.degrees(self.ref["angle"]))
        self.assertEqual(limits["rate_rad_s"],.005);self.assertIsNone(limits["position_m"])
    def test_runtime_fields_do_not_invalidate_plan(self):
        s=copy.deepcopy(self.s);s["execution"]["label"]="new label";s["execution"]["test_duration_s"]=20.
        self.assertEqual(signature(s),signature(self.s))
        s["design"]["duration_s"]+=1.;self.assertNotEqual(signature(s),signature(self.s))
    def test_no_implicit_zero_velocity_objective(self):
        with tempfile.TemporaryDirectory() as temp:
            s=copy.deepcopy(self.s);s["design"]["Q"]["velocity"]=[1,1,1]
            path=Path(temp)/"settings.json";path.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,"implicit zero"):
                settings(path)
    def test_official_fuel_formula(self):
        f=np.array([100.,150.,200.]);lam=np.array([.03,.07,0.])
        for dt in (.01,.07,.2):
            expected=sum(F*l/lam.max()*min(dt,lam.max())/(80*9.8) for F,l in zip(f,lam))
            self.assertAlmostEqual(native_impulse_fuel(f,lam,dt),expected,places=14)
        self.assertEqual(score_estimate(120,75),50.)
    def test_loaded_cache_and_terminal_input(self):
        p=Policy(FOLDER,ROOT/"nominal_model.json");self.assertTrue(p.manifest["export_verified"])
        np.testing.assert_allclose(self.model.mean_acceleration(p.commands[-1]),self.model.mean_acceleration([.5,0,0,0,0,0]),rtol=1e-4,atol=1e-7)
    def test_execution_edit_is_loadable_design_edit_is_rejected(self):
        from shutil import copytree
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"policy";copytree(FOLDER,folder,ignore=lambda d,n:[x for x in n if x in ("__pycache__","run_logs","analysis_results","_offline_packages")])
            s=copy.deepcopy(self.s);s["execution"]["label"]="new";(folder/"settings.json").write_text(json.dumps(s))
            Policy(folder,ROOT/"nominal_model.json")
            s["design"]["distance_m"]+=1.;(folder/"settings.json").write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,"--select"):Policy(folder,ROOT/"nominal_model.json")
    def test_fine_replay_and_soft_position(self):
        p=Policy(FOLDER,ROOT/"nominal_model.json")
        r=dict(times=np.r_[p.command_times,p.duration],commands=p.commands,scales=p.input_scales,reference=self.ref)
        r["states"]=self.model.rollout(self.x0,0,r["times"],r["times"][:-1],r["commands"])
        report=validate_replay(self.model,self.x0,r,self.s);self.assertTrue(report["passed"])
        r["reference"]=dict(self.ref,pd=self.ref["pd"]+100.)
        report=validate_replay(self.model,self.x0,r,self.s);self.assertTrue(report["checks"]["position"])
    def test_symbolic_physics_and_allocator_derivative(self):
        from policy.nonlinear_handoff_staging.offline import symbolic,Allocation,input_scales
        _,_,_,step,_=symbolic(self.model.definition,.5,.02)
        x=self.x0.copy();x[10:]=[.03,-.02,.01];b=np.array([.01,.02,-.01,.02,-.01,.03])
        np.testing.assert_allclose(np.array(step(x,b,.01)).ravel(),self.model._advance(x,.01,b),atol=1e-11)
        a=Allocation(self.model,input_scales(self.model));z=np.array([.7,-.3,.5,.6,-.8,.2]);eps=1e-5
        fd=np.column_stack([(a.values(z+np.eye(6)[j]*eps)-a.values(z-np.eye(6)[j]*eps))/(2*eps) for j in range(6)])
        np.testing.assert_allclose(a.derivative(z),fd,atol=1e-7,rtol=1e-5)
    def test_public_packet_world_to_body(self):
        q=increment(np.array([0.,0,0,1.]),[0,0,np.pi/2]);r=dict(position_P_m=[1,2,3],velocity_P_m_s=[0,0,0],q_PB_xyzw_reported=q.tolist(),omega_relP_P_rad_s=[1.,0,0])
        np.testing.assert_allclose(packet_row(r)[10:],matrix(q)@[1.,0,0],atol=1e-12)
    def test_no_scipy_or_casadi_import_in_flight_entry(self):
        code="import sys; import policy.nonlinear_handoff_staging.entry; assert 'casadi' not in sys.modules; assert 'scipy' not in sys.modules; assert 'spaceteams' not in sys.modules"
        r=subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
    def test_runner_plan_valid(self):
        from sim_core.common import validate
        c=validate(ROOT,json.loads((FOLDER/"run_plan.json").read_text()));self.assertEqual(c["entry"],"policy/nonlinear_handoff_staging/entry.py")
        self.assertIsNone(c["limits"]["task_sim_s"]);self.assertIsNone(c["limits"]["attempt_wall_s"])
    def test_actual_loader_fixed_period_model_feedback_nominal(self):
        self.run_numeric(False)
    def test_actual_loader_fixed_period_model_feedback_perturbed(self):
        self.run_numeric(True)
    def run_numeric(self,perturb):
        from sim_core.runtime import execute_entry
        with tempfile.TemporaryDirectory() as temp:
            c=NumericContext(temp,perturb=perturb)
            with patch("builtins.print"),self.assertRaises(EndFixture):execute_entry(c,{})
            np.testing.assert_array_equal(c.calls[-1],np.zeros(6))
            m,rows=metrics(temp);self.assertIsNotNone(m["handoff"]);self.assertIsNone(m["score_formula_estimate"])
            self.assertFalse(m["stop_sim_requested"])
            # A separate, full nonlinear closed-loop replay through the ACTUAL
            # runner entry. This is model validation, not native qualification.
            limits=terminal_limits(self.s,self.ref)
            self.assertLess(m["handoff_pointing_deg"],limits["pointing_deg"])
            self.assertLess(m["handoff_rate_rad_s"],limits["rate_rad_s"])
            self.assertEqual(rows[0]["stage"],"staging_trajectory")


if __name__=="__main__":unittest.main()
