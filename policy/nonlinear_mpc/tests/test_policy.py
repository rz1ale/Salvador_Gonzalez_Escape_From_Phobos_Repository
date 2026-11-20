"""Offline software/model checks. These do not run or certify native STPro."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import threading
import unittest
import numpy as np
from policy.nonlinear_mpc.optimizer import (Model,Predictor,settings,solve,score_win,
    future_schedule,reported,packet_row,swept_clearances,command_at,matrix)
from policy.nonlinear_mpc.entry import deviation,assess
from policy.nonlinear_mpc.worker import Worker
from policy.nonlinear_mpc.check import preflight
from sim_core.common import validate,read_json

ROOT=Path(__file__).resolve().parents[3]
C=dict(NominalTime_s=120.,NominalFuel_kg=75.,DamageThreshold_J=209200.,KeepOutRadius_m=50.)


def empty_frame(epoch):
    return dict(schema_version=1,frame='Phobos_body_fixed',units='SI',generated_sim_s=epoch,
        objects=[],coverage={'conditions':[]},acceleration_spectral_density_m2_s3=.01,
        default_query=dict(uncertainty_multiplier=3.,shape_padding_m=1.,forecast_horizon_s=10.))


class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/'nominal_model.json')
        cls.predictor=Predictor(cls.model,.15)

    def initial(self):
        return np.r_[[-650.,95.,-8240.],np.zeros(3),[0.,0.,0.,1.],np.zeros(3)]

    def job(self):
        x=self.initial();p,_=reported(self.model,x);target=p+np.array([0.,0.,-1200.])
        up=p/np.linalg.norm(p)
        from policy.nonlinear_mpc.optimizer import recovery_request
        u=recovery_request(self.predictor,x,target,'coast')
        return dict(job_id='test_job',generation=5,source_sim_s=1.,activation_sim_s=4.,
            state=x.tolist(),target=target.tolist(),fuel_kg=1.,damage_J=0.,
            score_constants=C,settings=settings({'solve_wall_s':.35}),
            committed_times=[0.],committed_commands=[u.tolist()],obstacles=empty_frame(1.),
            launch_position_P=p.tolist(),launch_up_P=up.tolist())

    def test_official_score_matches_recorded_formula_example(self):
        self.assertAlmostEqual(score_win(237.923093,64.61852438804436,0.,C),14.278433521656531,places=10)
        self.assertGreater(score_win(100,20,0,C),score_win(110,20,0,C))
        self.assertGreater(score_win(100,20,0,C),score_win(100,21,0,C))
        self.assertAlmostEqual(score_win(100,20,104600,C)-score_win(100,20,0,C),-100.)

    def test_settings_and_run_plans_use_shared_schema(self):
        for filename in ('run_plan.json','smoke_run.json'):
            plan=validate(ROOT,read_json(ROOT/'policy/nonlinear_mpc'/filename))
            self.assertEqual(plan['entry'],'policy/nonlinear_mpc/entry.py')
            self.assertEqual(plan['model'],'nominal_model.json')
            self.assertEqual(plan['retry']['max_retries_per_trial'],0)
            self.assertIn('system_identification/model.py',plan['dependencies'])
            self.assertIn('obstacle_characterization/runtime.py',plan['dependencies'])
        with self.assertRaises(ValueError):settings({'typographic_error':1})
        with self.assertRaises(ValueError):settings({'horizon_s':12.})

    def test_preflight_reads_unchanged_official_sources(self):
        model,constants,evidence=preflight(ROOT,ROOT/'nominal_model.json',settings())
        self.assertEqual(constants,C)
        self.assertEqual(set(evidence['sources']),{'Scoring','JetSelectionActuator'})
        self.assertEqual(model.definition['model_type'],'public_request_mean_allocation_6dof_v1')

    def test_prediction_matches_canonical_delayed_rollout(self):
        x=self.initial();x[10:]=[.12,-.09,.07]
        ct=[0.,1.,3.2,5.1]
        z=np.array([[.2,-.15,.1,.15,-.1,.2],[-.2,.3,.1,-.1,.15,0],
                    [.4,-.2,.3,.1,0,-.1],[0.,0.,0.,0.,0.,0.]])
        u=self.predictor.decode(z);times=np.linspace(.5,8.,16)
        canonical=self.model.rollout(x,.5,times,ct,u)
        coarse,_=self.predictor.rollout(x,.5,times,ct,u)
        self.assertLess(np.max(np.linalg.norm(canonical[:,:3]-coarse[:,:3],axis=1)),.02)
        self.assertLess(np.max(np.linalg.norm(canonical[:,3:6]-coarse[:,3:6],axis=1)),.005)
        self.assertLess(np.max(np.linalg.norm(canonical[:,6:10]-coarse[:,6:10],axis=1)),.002)

    def test_fuel_kernel_matches_independent_native_impulse_equation(self):
        # Independent one-tick impulse summation from supplied static source.
        for z in ([.03,-.02,.01,0,0,0],[1.,-.8,.7,.4,-.6,.9]):
            u=self.predictor.decode(np.array(z));b=self.model.allocator.clip(u)
            lam=self.model.allocator.solve(b);peak=max(lam);dt=self.model.h
            fired=lam/peak*min(dt,peak)
            impulse=self.model.definition['mass_kg']*fired[:,None]*self.model.allocator.A[:,:3]
            expected=float(np.linalg.norm(impulse,axis=1).sum()/(80*9.8)/dt)
            base,fuel=self.predictor.allocation(u)
            np.testing.assert_allclose(base,self.model.allocator.mean(u,dt),atol=1e-12)
            self.assertAlmostEqual(fuel,expected,places=7)

    def test_single_axis_knee_scaling_and_joint_saturation(self):
        for sign in (-1,1):
            for i in range(6):
                z=np.eye(6)[i]*sign;u=self.predictor.decode(z)
                self.assertAlmostEqual(self.model.allocator.solve(u).max(),self.model.h,places=8)
        z=np.ones(6);u=self.predictor.decode(z)
        self.assertLess(np.linalg.norm(self.model.allocator.mean(u,self.model.h)),np.linalg.norm(u/self.model.h))

    def test_public_rate_frame_and_reporting_origin(self):
        q=np.array([0.,0.,np.sin(.3),np.cos(.3)])
        row=packet_row(dict(position_P_m=[1.,2.,3.],velocity_P_m_s=[0.,0.,0.],
            q_PB_xyzw_reported=q.tolist(),omega_relP_P_rad_s=[1.,0.,0.]))
        np.testing.assert_allclose(row[10:],[np.cos(.6),np.sin(.6),0.],atol=1e-12)
        _,latent,_=self.model.causal_initial(np.array([1.]),row[None,:],1.)
        p,_=reported(self.model,latent);np.testing.assert_allclose(p,row[:3],atol=1e-12)

    def test_completed_send_after_state_snapshot_is_not_lost(self):
        history=[(.8,np.zeros(6)),(1.01,np.ones(6)*.001)]
        t,u=future_schedule(history,None,1.,2.)
        self.assertIn(1.01,t)
        np.testing.assert_allclose(u[-1],history[-1][1])

    def test_swept_check_catches_collision_between_knots(self):
        path=np.array([[-10.,0.,0.],[10.,0.,0.]])
        centers=np.zeros((1,2,3));radii=np.ones((1,2))*2
        self.assertLess(swept_clearances(path,centers,radii)[0],0.)
        path[:,1]=3.;self.assertGreater(swept_clearances(path,centers,radii)[0],0.)

    def test_solver_returns_feasible_bounded_coupled_plan_under_budget(self):
        result=solve(self.predictor,self.job())
        self.assertEqual(result['status'],'feasible')
        z=self.predictor.encode(result['commands'])
        self.assertLessEqual(np.max(abs(z)),1.+1e-8)
        self.assertEqual(z.shape,(3,6))
        self.assertLess(result['solve_wall_s'],1.5)
        self.assertFalse(result['official_score_captured'])
        self.assertGreater(result['tail_time_s'],0.)

    def test_unavoidable_obstacle_never_returns_feasible_plan(self):
        job=self.job();p=np.array(job['launch_position_P'])
        job['obstacles']['objects']=[dict(id='covering_hazard',kind='track',status='tentative',
            center_P_m=p.tolist(),velocity_P_m_s=[0.,0.,0.],covariance_P=np.zeros((6,6)).tolist(),
            state_epoch_sim_s=1.,radius_m=50.,radius_sigma_m=0.,extra_position_margin_m=0.)]
        result=solve(self.predictor,job)
        self.assertEqual(result['status'],'no_feasible_plan')
        self.assertNotIn('commands',result)

    def test_expired_commands_and_state_deviation_rejected(self):
        result=solve(self.predictor,self.job());result['kind']='nmpc'
        self.assertIsNone(command_at(result,result['expires_sim_s']))
        x=np.array(result['predicted_states'][0]);x[0]+=20.
        self.assertTrue(deviation(self.predictor,x,result['activation_sim_s'],result,settings()))

    def test_numeric_subprocess_handshake_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=Worker(ROOT,ROOT/'nominal_model.json',Path(directory)/'worker')
            try:
                deadline=time.monotonic()+15.
                while not worker.ready() and time.monotonic()<deadline:
                    self.assertTrue(worker.alive());time.sleep(.05)
                self.assertTrue(worker.ready())
                job=self.job();worker.submit(job)
                result=None
                while result is None and time.monotonic()<deadline:
                    result=worker.poll();time.sleep(.03)
                self.assertIsNotNone(result)
                self.assertEqual(result['job_id'],job['job_id'])
                self.assertEqual(result['status'],'feasible')
            finally:worker.close()
            self.assertIsNotNone(worker.process.poll())

    def test_numeric_worker_exits_after_owner_pulse_expires(self):
        with tempfile.TemporaryDirectory() as folder:
            worker=Worker(ROOT,ROOT/'nominal_model.json',Path(folder)/'worker')
            try:
                deadline=time.monotonic()+15.
                while not worker.ready() and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue(worker.ready())
                old=time.time()-11.
                os.utime(worker.directory/'owner_heartbeat',(old,old))
                self.assertEqual(worker.process.wait(timeout=2.),0)
            finally:worker.close()

    def test_complete_entry_with_public_interface_fixture(self):
        """Real numeric subprocess + mathematical own-state fixture, not STPro."""
        from policy.nonlinear_mpc.entry import run
        class Finished(BaseException):pass
        class PublicContext:
            def __init__(self,folder,model,x):
                self.root=ROOT;self.model_path=ROOT/'nominal_model.json';self.output_dir=Path(folder)
                self.parameters=dict(task='smoke',smoke_duration_s=10.,overrides=dict(
                    command_bound_scale=.35,solve_wall_s=.5,activation_lead_s=2.,update_period_s=2.))
                self.model=model;self.x=x.copy();self.wall=time.monotonic();self.t=1.
                self.commands=[];self.main_thread=threading.get_ident();self.stopped=False;self.value=None
                self.goal=reported(model,x)[0]+np.array([0.,0.,-1200.])
                self.obstacles=reported(model,x)[0]+np.array([[800.,800.,-100.],[-800.,700.,-200.]])
            def clock(self):
                assert threading.get_ident()==self.main_thread,'SDK fixture called off controller thread'
                now=1.+time.monotonic()-self.wall
                if now>self.t:
                    ct=[t for t,_ in self.commands] or [0.]
                    cu=[u for _,u in self.commands] or [np.zeros(6).tolist()]
                    self.x=self.model.rollout(self.x,self.t,[now],ct,cu)[0];self.t=now
                return now
            def state(self):
                begin=self.clock();p,_=reported(self.model,self.x)
                return dict(read_begin_sim_s=begin,read_end_sim_s=self.clock(),position_P_m=p.tolist(),
                    velocity_P_m_s=(self.x[3:6]+self.model.vbias).tolist(),q_PB_xyzw_reported=self.x[6:10].tolist(),
                    omega_relP_P_rad_s=(matrix(self.x[6:10]).T@(self.x[10:13]+self.model.wbias)).tolist(),
                    target_P_m=self.goal.tolist(),fuel_used_kg=0.,damage_J=0.)
            def radar(self,count):
                now=self.clock();p,_=reported(self.model,self.x)
                return [np.array([o-p,[np.linalg.norm(o-p)-2.,0.,4.],[now,0.,0.]]) for o in self.obstacles]
            def command(self,a,alpha):
                now=self.clock();self.commands.append((now,np.r_[a,alpha].tolist()))
                return dict(send_begin_sim_s=now,send_end_sim_s=now)
            def event(self,*args,**kwargs):pass
            def checkpoint(self,**kwargs):self.clock()
            def request_stop_after_task(self):self.stopped=True
            def sleep(self,seconds):time.sleep(seconds)
            def finish(self,**kwargs):self.value=kwargs;raise Finished()
        with tempfile.TemporaryDirectory() as folder:
            ctx=PublicContext(folder,self.model,self.initial())
            with self.assertRaises(Finished):run(ctx)
            self.assertEqual(ctx.value['reason'],'nmpc_smoke_complete',ctx.value)
            self.assertTrue(ctx.value['smoke_passed'])
            self.assertTrue(ctx.stopped)
            self.assertGreaterEqual(ctx.value['activated_plans'],1)
            self.assertFalse(ctx.value['official_score_captured'])
            self.assertIsNone(ctx.value['score_formula_estimate'])


if __name__=='__main__':unittest.main()
