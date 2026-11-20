"""Offline verification only. Does not connect to or start a native simulation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize
from system_identification.model import Model,matrix,Allocator
from policy.trajectory_handoff.reference import Reference,error,retract,packet_row,latent_state,interpolate_state
from policy.trajectory_handoff.core import Plant,score,CertifiedAllocationCache
from policy.trajectory_handoff.check import settings,preflight
from policy.trajectory_handoff.online import qp,solve,readiness
from policy.trajectory_handoff.entry import response_usable,send_readiness,plan_control,renewal_budget
from policy.trajectory_handoff.worker import Tail,PublicPerception,run_job,PerceptionDeadline,Worker

ROOT=Path(__file__).resolve().parents[3]


def frame(center,epoch=40.,radius=2.):
    return dict(schema_version=1,frame='Phobos_body_fixed',units='SI',generated_sim_s=epoch,sequence=1,diagnostics={},
        acceleration_spectral_density_m2_s3=0.,default_query=dict(uncertainty_multiplier=0.,shape_padding_m=0.,forecast_horizon_s=45.),
        coverage=dict(conditions=[],guarantees_free_space=False),objects=[dict(id='synthetic',kind='track',status='confirmed',
            state_epoch_sim_s=epoch,center_P_m=list(center),velocity_P_m_s=[0.,0.,0.],covariance_P=np.zeros((6,6)).tolist(),
            radius_m=radius,radius_sigma_m=0.,extra_position_margin_m=0.)])


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Model.load(ROOT/'nominal_model.json');cls.plant=Plant(cls.model,.1)
        cls.ref=Reference(ROOT/'policy/trajectory_handoff/assets',ROOT/'nominal_model.json');cls.s=settings()

    def job(self,t=40.):
        x,u,*_=self.ref.at(t);launch=self.ref.at(0)[0][:3]
        return dict(job_id='unit',generation=0,source_sim_s=t,phase=t,state=x.tolist(),last_command=u.tolist(),active_plan=None,
            launch_position_P=launch.tolist(),launch_up_P=(launch/np.linalg.norm(launch)).tolist(),fuel_kg=5.,damage_J=0.,score_constants=self.ref.manifest['score_constants'])

    def test_quaternion_error_inverse_sign_and_cover(self):
        rng=np.random.default_rng(20)
        for _ in range(20):
            x=self.ref.at(float(rng.uniform(0,self.ref.duration)))[0];d=rng.normal(0,.03,12)
            y=retract(x,d);self.assertLess(np.max(abs(error(y,x)-d)),1e-10)
            y[6:10]*=-1;self.assertLess(np.max(abs(error(y,x)-d)),1e-10)

    def test_reporting_adapter_matches_canonical_with_delayed_slopes(self):
        ts=np.linspace(1.,1.3,5);ys=np.tile(self.ref.at(0)[0],(5,1))
        ys[:,3:6]+=ts[:,None]*np.array([.1,.2,-.3]);ys[:,10:13]+=ts[:,None]*[.02,-.01,.03]
        actual=latent_state(ts,ys,self.model.p);_,expected,_=self.model.causal_initial(ts,ys,ts[-1])
        np.testing.assert_allclose(actual,expected,atol=1e-13)
        p=dict(position_P_m=ys[-1,:3],velocity_P_m_s=ys[-1,3:6],q_PB_xyzw_reported=ys[-1,6:10],
            omega_relP_P_rad_s=matrix(ys[-1,6:10]).T@ys[-1,10:13])
        np.testing.assert_allclose(packet_row(p),ys[-1],atol=1e-13)

    def test_delay_and_coupled_allocator_match_canonical_rollout(self):
        x=self.ref.at(20.)[0];old=np.array([.001,-.001,.002,.0002,0.,.0001]);u=np.array([-.001,.002,.001,-.001,.0005,.0002])
        fast=self.plant.transition(x,old,u,.7)
        exact=self.model.rollout(x,0.,[.7],[-1.,0.],np.array([old,u]))[-1]
        self.assertLess(np.linalg.norm(error(fast,exact)),.002)
        requested=np.array([.2,.1,.2,.02,.03,.01]);achieved=self.model.mean_acceleration(requested)
        self.assertLess(np.linalg.norm(achieved[:3]),np.linalg.norm(requested[:3]))
        self.assertGreater(self.plant.fuel_rate(requested),0.)

    def test_canonical_request_preserves_saturated_mean_and_fuel(self):
        rng=np.random.default_rng(8)
        for _ in range(10):
            raw=rng.normal(size=6)*np.r_[[.006]*3,[.002]*3];canonical=self.plant.canonical_request(raw)
            np.testing.assert_allclose(self.plant.base(raw),self.plant.base(canonical),atol=1e-8)
            self.assertAlmostEqual(self.plant.fuel_rate(raw),self.plant.fuel_rate(canonical),places=7)
            self.assertLessEqual(self.model.allocator.solve(canonical).max(),self.model.h+1e-8)

    def test_canonical_linearization_retains_thrust_reduction_authority(self):
        x=self.ref.at(0.)[0];raw=np.array([0.,0.,.01,0.,0.,0.]);canonical=self.plant.canonical_request(raw)
        _,_,B,D=self.plant.linearize(x,raw,raw,.3)
        _,_,Bc,Dc=self.plant.linearize(x,canonical,canonical,.3)
        self.assertLess(np.linalg.norm((B+D)[3:6,2]),1e-6)
        self.assertGreater(np.linalg.norm((Bc+Dc)[3:6,2]),.005)

    def test_qp_agrees_with_independent_slsqp_solution(self):
        H=np.array([[4.,1.],[1.,2.]]);q=np.array([-3.,-4.]);C=np.array([[1.,0.],[0.,1.],[1.,1.]])
        lo=np.array([0.,0.,-.1]);hi=np.array([1.,2.,1.4])
        x,info=qp(H,q,C,lo,hi,iterations=3000,tolerance=1e-7)
        fit=minimize(lambda z:.5*z@H@z+q@z,[.4,.4],jac=lambda z:H@z+q,method='SLSQP',
            constraints=[{'type':'ineq','fun':lambda z:C@z-lo},{'type':'ineq','fun':lambda z:hi-C@z}],options={'ftol':1e-12})
        self.assertTrue(fit.success);np.testing.assert_allclose(x,fit.x,atol=1e-5)
        bad,_=qp(np.eye(1),np.zeros(1),np.array([[1.],[1.]]),np.array([1.,-np.inf]),np.array([np.inf,0.]),iterations=500)
        self.assertIsNone(bad)

    def test_missing_empty_future_and_lost_capacity_radar_never_clear(self):
        s=self.s;self.assertFalse(readiness(None,40,s)[0])
        f=frame([1.,2.,3.]);f['objects']=[];self.assertFalse(readiness(f,40,s)[0])
        f=frame([1.,2.,3.],41.);self.assertFalse(readiness(f,40,s)[0])
        for condition in ('empty_radar_packet','unresolved_capacity_loss','clock_unregistered'):
            f=frame([1.,2.,3.]);f['coverage']['conditions']=[condition];self.assertFalse(readiness(f,40,s)[0])

    def test_nominal_clear_uses_future_forecast_without_false_staleness(self):
        j=self.job();x=np.asarray(j['state']);f=frame(x[:3]+[300.,300.,0.])
        r=solve(self.plant,self.ref,j,f,self.s)
        self.assertEqual(r['status'],'nominal_clear',r)
        self.assertFalse(r['certified_safe']);self.assertGreater(r['valid_until_sim_s'],j['source_sim_s'])

    def test_verified_incumbent_survives_optimizer_budget_exhaustion(self):
        from policy.trajectory_handoff import online
        j=self.job();j['force_mpc']=True;x=np.asarray(j['state']);f=frame(x[:3]+[300.,300.,0.])
        with patch.object(online,'mpc',side_effect=online.Deadline()):r=online.solve(self.plant,self.ref,j,f,self.s)
        self.assertEqual(r['status'],'mpc_feasible',r);self.assertTrue(r['optimization_budget_exhausted'])
        self.assertEqual(r['selected_candidate'],'verified_feasible_start');self.assertTrue(r['plan_assessment']['admissible'])

    def test_planning_budget_is_independent_of_expired_monitor_assessment(self):
        for now in (42.8,44.,100.):
            self.assertEqual(renewal_budget(self.s,dict(valid_until_sim_s=44.),now),self.s['solve_wall_s'])
        j=self.job();j['solve_wall_s']=1e-6;x=np.asarray(j['state'])
        r=solve(self.plant,self.ref,j,frame(x[:3]+300.),self.s)
        self.assertEqual(r['reason'],'numeric_budget_exhausted')
        with self.assertRaises(ValueError):solve(self.plant,self.ref,dict(j,solve_wall_s=4.),None,self.s)

    def test_mission_grid_has_fine_near_field_and_coarse_target_tail(self):
        from policy.trajectory_handoff.online import planning_grid
        ts=planning_grid(self.ref,40.,44.,self.s)
        np.testing.assert_allclose(np.diff(ts)[:10],1.5)
        self.assertAlmostEqual(ts[-1],44.+self.ref.duration-40.)
        self.assertGreater(np.max(np.diff(ts)),1.5)
        self.assertGreater(ts[-1]-ts[0],self.s['advisory_horizon_s'])

    def test_affine_error_jacobian_matches_short_nonlinear_transitions(self):
        for phase in (0.,40.,100.):
            x,u,*_=self.ref.at(phase);u=self.plant.canonical_request(u)
            A,B,D=self.plant.affine_linearize(x,u,u,.1)
            _,Ae,Be,De=self.plant.linearize(x,u,u,.1)
            np.testing.assert_allclose(A,Ae,atol=1.1e-5,rtol=0)
            np.testing.assert_allclose(B,Be,atol=4e-6,rtol=0)
            np.testing.assert_allclose(D,De,atol=3e-7,rtol=0)

    def test_monitor_does_not_need_a_successful_mission_extension(self):
        from policy.trajectory_handoff import guard
        from policy.trajectory_handoff.reference import coast_plan
        j=self.job();j['active_plan']=coast_plan(np.asarray(j['state']),40.,40.);j['solve_wall_s']=.8
        j['do_advisory']=False
        r=guard.monitor(self.plant,self.ref,j,frame(np.asarray(j['state'])[:3]+300.),self.s)
        self.assertEqual(r['status'],'monitor_clear',r.get('reason'))
        self.assertTrue(r['clearance_verified']);self.assertNotIn('plan',r)

    def test_expired_plan_selects_explicit_coast_not_nominal_thrust(self):
        from policy.trajectory_handoff.reference import schedule_control
        x,u,*_=self.ref.at(40.)
        plan=dict(times=[39.,40.],states=[x.tolist(),x.tolist()],commands=[u.tolist()],gains=[np.zeros((6,12)).tolist()],phase=39.)
        actual=schedule_control(plan,x,41.,self.ref,41.,self.s)
        np.testing.assert_array_equal(actual[:3],np.zeros(3))

    def test_unassessed_radar_never_becomes_monitor_clear(self):
        from policy.trajectory_handoff.guard import monitor
        j=dict(self.job(),solve_wall_s=.8)
        r=monitor(self.plant,self.ref,j,None,self.s)
        self.assertEqual(r['status'],'monitor_unassessed');self.assertFalse(r['observation_available'])
        self.assertFalse(r['clearance_verified'])

    def test_monitor_overlap_is_risk_not_a_clearance_certificate(self):
        from policy.trajectory_handoff.guard import monitor
        j=dict(self.job(),solve_wall_s=.8);f=frame(np.asarray(j['state'])[:3],radius=1000.)
        r=monitor(self.plant,self.ref,j,f,self.s)
        self.assertIn(r['status'],('monitor_hazard','guard_risk_fallback'))
        self.assertFalse(r['clearance_verified']);self.assertTrue(r['replan_requested'])
        if 'plan' in r:self.assertTrue(r['risk_accepted']);self.assertFalse(r['plan']['clearance_verified'])

    def test_pending_switch_is_included_in_the_predicted_command_schedule(self):
        from policy.trajectory_handoff.entry import scheduled_snapshot
        from policy.trajectory_handoff.reference import coast_plan,schedule_control
        x,u,*_=self.ref.at(40.);active=coast_plan(x,40.,40.)
        plan=dict(times=[41.,43.],states=[x.tolist(),x.tolist()],commands=[u.tolist()],gains=[np.zeros((6,12)).tolist()],phase=41.)
        pending=dict(status='mpc_feasible',activation_sim_s=41.,plan=plan)
        schedule=scheduled_snapshot(active,pending,40.)
        np.testing.assert_array_equal(schedule_control(schedule,x,40.5,self.ref,40.5,self.s)[:3],np.zeros(3))
        np.testing.assert_allclose(schedule_control(schedule,x,41.5,self.ref,41.5,self.s),u,atol=1e-15)

    def test_coarse_tail_attitude_does_not_change_the_explicit_coast_continuation(self):
        from policy.trajectory_handoff.reference import coast_plan,schedule_control
        x=self.ref.at(55.)[0].copy();x[10:13]=0.
        far=retract(x,np.r_[np.zeros(6),[1.,0.,0.],np.zeros(3)])
        plan=dict(times=[40.,55.,100.],execute_until_sim_s=55.,states=[x.tolist(),x.tolist(),far.tolist()],phase=40.)
        predicted=schedule_control(plan,x,55.5,self.ref,55.5,self.s)
        actual=schedule_control(coast_plan(x,55.,55.),x,55.5,self.ref,55.5,self.s)
        np.testing.assert_allclose(predicted,actual,atol=1e-15)
        np.testing.assert_array_equal(predicted,np.zeros(6))

    def test_whole_mission_candidate_has_only_a_verified_near_execution_window(self):
        j=self.job();j['force_mpc']=True;x=np.asarray(j['state'])
        r=solve(self.plant,self.ref,j,frame(x[:3]+300.),self.s)
        self.assertEqual(r['status'],'mpc_feasible',r.get('reason'))
        self.assertEqual(r['selected_candidate'],'linear_mpc')
        plan=r['plan'];self.assertGreater(plan['times'][-1],plan['execute_until_sim_s'])
        self.assertAlmostEqual(plan['execute_until_sim_s']-plan['times'][0],15.)
        self.assertFalse(plan['full_tail_clearance_verified'])
        self.assertLessEqual(plan['affine_terminal_distance_m'],50.)
        self.assertTrue(r['plan_assessment']['admissible'])

    def test_worker_interim_response_keeps_single_job_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=object.__new__(Worker);w.directory=Path(tmp);w.job_id='job';w.submitted=1.;w.response_sequence=-1
            path=w.directory/'response.json'
            path.write_text(json.dumps(dict(job_id='job',is_final=False,response_sequence=1,status='active_clear')))
            self.assertFalse(w.poll()['is_final']);self.assertEqual(w.job_id,'job')
            self.assertEqual(w.submitted,1.);self.assertIsNone(w.poll())
            with self.assertRaises(RuntimeError):w.submit({'job_id':'overlap'})
            path.write_text(json.dumps(dict(job_id='job',is_final=True,response_sequence=2,status='active_clear')))
            self.assertTrue(w.poll()['is_final']);self.assertIsNone(w.job_id);self.assertIsNone(w.submitted)
            self.assertIsNone(w.poll())

    def test_run_job_relays_early_revalidation_before_extension_finishes(self):
        class Clock:
            value=.1
            def __call__(self):return self.value
        clock=Clock();plant=types.SimpleNamespace(model=types.SimpleNamespace(allocator=None));partials=[]
        class Perception:
            def update(self,until,**kw):clock.value+=.2;return {}
        def solver(p,r,j,f,s,*,on_current):
            clock.value+=.1
            on_current(dict(job_id=j['job_id'],status='active_clear',wall_s=.1))
            clock.value+=.6
            return dict(job_id=j['job_id'],status='active_clear',planning_status='blocked',wall_s=.7)
        job=dict(job_id='staged',generation=0,source_sim_s=10.,solve_wall_s=2.,submitted_monotonic_s=0.)
        final=run_job(plant,None,job,Perception(),self.s,clock=clock,solver=solver,on_assessment=partials.append)
        self.assertAlmostEqual(partials[0]['timing']['end_to_end_wall_s'],.4)
        self.assertAlmostEqual(final['timing']['end_to_end_wall_s'],1.)
        self.assertEqual(final['planning_status'],'blocked')

    def test_certified_cache_preserves_allocation_mean_and_fuel(self):
        canonical=Model.load(ROOT/'nominal_model.json');fast=Model.load(ROOT/'nominal_model.json');plant=Plant(fast)
        origin=np.array([.004,.00002,-.0003,-.00002,-.0001,-.00004])
        for k in range(24):
            u=origin+plant.scales*np.array([1.,-.3,.2,.1,-.2,.4])*k*1e-5
            b=canonical.allocator.clip(u);expected=canonical.allocator.solve(b);actual=fast.allocator.solve(b)
            np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=1e-10)
            np.testing.assert_allclose(plant.base(u),canonical.allocator.mean(u,canonical.h),rtol=2e-8,atol=1e-10)
            fuel=float(canonical.allocator.forces@expected/max(canonical.h,float(expected.max()))/(80.*9.8))
            self.assertAlmostEqual(plant.fuel_rate(u),fuel,places=8)
        self.assertGreater(fast.allocator.stats['certified_hits'],10)
        for u in [np.zeros(6),np.full(6,1e-13),-origin,origin*1e-5]:
            b=canonical.allocator.clip(u)
            np.testing.assert_allclose(fast.allocator.solve(b),canonical.allocator.solve(b),rtol=2e-8,atol=1e-10)

    def test_nonunique_allocation_keeps_the_declared_solver_tie_break(self):
        # Duplicate +axis jets give multiple minimum-sum optima with different
        # per-jet peaks and fuel weights. A merely feasible optimum is insufficient.
        A=np.vstack((np.eye(6),-np.eye(6),np.eye(6)));forces=np.arange(1.,19.)
        canonical=Allocator(A,forces);cached=CertifiedAllocationCache(Allocator(A,forces))
        for k in range(1,5):
            b=np.arange(1.,7.)*k*.001
            expected=canonical.solve(b);actual=cached.solve(b)
            np.testing.assert_allclose(actual,expected,rtol=0.,atol=1e-12)
            self.assertEqual(float(actual.max()),float(expected.max()))
            self.assertEqual(float(forces@actual),float(forces@expected))
        self.assertEqual(cached.stats['certified_hits'],0)
        self.assertEqual(cached.stats['full_solves'],4)

    def test_cached_allocation_preserves_the_full_nonlinear_forecast(self):
        from policy.trajectory_handoff.online import forecast
        ordinary=Plant(Model.load(ROOT/'nominal_model.json'),self.s['integration_step_s'],certified_allocation=False)
        cached=Plant(Model.load(ROOT/'nominal_model.json'),self.s['integration_step_s'])
        x=self.ref.at(40.)[0];x=retract(x,np.r_[[.2,-.1,.1],np.zeros(9)])
        old=self.ref.control(40.,x,self.s['feedback_limit'])
        a=forecast(ordinary,self.ref,x,40.,57.8,40.,None,old,self.s)
        b=forecast(cached,self.ref,x,40.,57.8,40.,None,old,self.s)
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_allclose(a[1],b[1],rtol=0.,atol=1e-7)
        np.testing.assert_allclose(a[2],b[2],rtol=0.,atol=1e-9)
        self.assertGreater(cached.model.allocator.stats['certified_hits'],0)

    def test_job_budget_includes_queue_and_perception_before_numeric_work(self):
        class Clock:
            value=.2
            def __call__(self):return self.value
        clock=Clock();plant=types.SimpleNamespace(model=types.SimpleNamespace(allocator=None));seen=[]
        class Perception:
            def update(self,until,**kw):clock.value+=.6;return {'public':True}
        def solver(p,r,j,f,s):
            seen.append(j['solve_wall_s']);clock.value+=1.
            return dict(job_id=j['job_id'],status='nominal_clear',wall_s=1.)
        j=dict(job_id='deadline',generation=0,source_sim_s=10.,solve_wall_s=2.,submitted_monotonic_s=0.,deadline_monotonic_s=2.)
        value=run_job(plant,None,j,Perception(),self.s,clock=clock,solver=solver)
        self.assertAlmostEqual(seen[0],1.2)
        self.assertAlmostEqual(value['timing']['queue_wall_s'],.2)
        self.assertAlmostEqual(value['timing']['perception_wall_s'],.6)
        self.assertAlmostEqual(value['timing']['end_to_end_wall_s'],1.8)

    def test_queued_job_that_has_expired_never_processes_observations(self):
        plant=types.SimpleNamespace(model=types.SimpleNamespace(allocator=None));perception=types.SimpleNamespace(update=lambda *a,**kw:self.fail('Expired job consumed observations'))
        j=dict(job_id='expired',generation=0,source_sim_s=10.,solve_wall_s=2.,submitted_monotonic_s=0.,deadline_monotonic_s=2.)
        value=run_job(plant,None,j,perception,self.s,clock=lambda:2.1,solver=lambda *a:self.fail('Expired job ran solver'))
        self.assertEqual(value['reason'],'job_budget_exhausted');self.assertEqual(value['budget_stage'],'queue')

    def test_perception_spending_the_budget_never_starts_the_solver(self):
        class Clock:
            value=0.
            def __call__(self):return self.value
        clock=Clock();plant=types.SimpleNamespace(model=types.SimpleNamespace(allocator=None))
        class Perception:
            def update(self,*a,**kw):clock.value=2.1;return {}
        j=dict(job_id='perception',generation=0,source_sim_s=10.,solve_wall_s=2.,submitted_monotonic_s=0.,deadline_monotonic_s=2.)
        value=run_job(plant,None,j,Perception(),self.s,clock=clock,solver=lambda *a:self.fail('No solver budget remains'))
        self.assertEqual(value['reason'],'job_budget_exhausted');self.assertEqual(value['budget_stage'],'perception')

    def test_perception_pause_preserves_packets_and_matches_uninterrupted_tracking(self):
        class Clock:
            value=0.
            def __call__(self):return self.value
        clock=Clock()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            states=[dict(read_begin_sim_s=float(t),read_end_sim_s=float(t),position_P_m=[0.,0.,0.],
                velocity_P_m_s=[0.,0.,0.],q_PB_xyzw_reported=[0.,0.,0.,1.]) for t in np.arange(0.,3.2,.1)]
            packets=[dict(read_sim_s=float(t),observations=[[[100.,0.,0.],[99.,0.,2.],[float(t),0.,0.]]]) for t in [1.,2.,3.]]
            (folder/'states.txt').write_text(''.join(json.dumps(p)+'\n' for p in states))
            (folder/'radar.txt').write_text(''.join(json.dumps(p)+'\n' for p in packets))
            interrupted=PublicPerception(folder,self.s,log=False);baseline=PublicPerception(folder,self.s,log=False)
            original=interrupted.tracker.update
            def slow(*a,**kw):
                f=original(*a,**kw);clock.value+=.6;return f
            interrupted.tracker.update=slow
            with self.assertRaises(PerceptionDeadline):interrupted.update(3.,deadline_wall_s=1.,clock=clock)
            self.assertEqual(interrupted.tracker.sequence,2);self.assertEqual(len(interrupted.radar.pending),1)
            actual=interrupted.update(3.);expected=baseline.update(3.)
            self.assertEqual(actual['sequence'],3);self.assertEqual(actual['coverage'],expected['coverage'])
            self.assertEqual([o['id'] for o in actual['objects']],[o['id'] for o in expected['objects']])
            np.testing.assert_allclose(actual['objects'][0]['covariance_P'],expected['objects'][0]['covariance_P'],atol=1e-12)

    def test_coverage_warmup_and_blocked_responses_keep_their_actual_reason(self):
        j=self.job();x=np.asarray(j['state']);f=frame(x[:3]+[300.,300.,0.]);f['coverage']['conditions']=['initial_unstamped']
        r=solve(self.plant,self.ref,j,f,self.s)
        self.assertEqual(r['status'],'data_unavailable');self.assertEqual(r['reason'],'radar_coverage_incomplete')
        usable,reason=response_usable(r,j['generation'],j['source_sim_s']+.1,x,self.s)
        self.assertFalse(usable);self.assertEqual(reason,'radar_coverage_incomplete')

    def test_staggered_track_dropouts_do_not_expire_the_policy_assessment(self):
        # One live object plus five staggered dropouts. Exercise the actual
        # public-log reader -> tracker -> planner -> receipt/freshness contract
        # for 38.4 simulation seconds. This does not start a native simulator.
        def position(t):
            x=self.ref.at(t)[0];return x[:3]+matrix(x[6:10]).T@self.model.ell
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);packets=[]
            for t in np.arange(0.,40.1,.1):
                x=self.ref.at(t)[0]
                packets.append(dict(read_begin_sim_s=float(t),read_end_sim_s=float(t),position_P_m=position(t).tolist(),
                    velocity_P_m_s=(x[3:6]+self.model.vbias).tolist(),q_PB_xyzw_reported=x[6:10].tolist()))
            (folder/'states.txt').write_text(''.join(json.dumps(p)+'\n' for p in packets))
            centers=position(0.)+np.array([[1000.,1000.+200*i,50.*i] for i in range(6)])
            radar=[];cached={};epochs=np.arange(1.2,40.,1.2)
            for t in epochs:
                p=position(t);v=(position(t+.001)-position(t-.001))/.002
                for i,c in enumerate(centers):
                    if i==0 or t<=1.2*(i+1)+1e-6:
                        r=c-p;distance=np.linalg.norm(r)
                        cached[i]=[r.tolist(),[distance-1.,float(-v@(r/distance)),2.],[float(t),0.,0.]]
                radar.append(dict(read_sim_s=float(t),observations=list(cached.values())))
            (folder/'radar.txt').write_text(''.join(json.dumps(p)+'\n' for p in radar))
            fixed=PublicPerception(folder,self.s,log=False)
            legacy=PublicPerception(folder,dict(self.s,radar_track_expiry_mode='drop_unassessed'),log=False)
            valid_until=-np.inf;legacy_valid=-np.inf;legacy_stopped=False
            for t in epochs:
                f=fixed.update(float(t));old=legacy.update(float(t));j=self.job(float(t))
                launch=position(0.);j.update(launch_position_P=launch.tolist(),launch_up_P=(launch/np.linalg.norm(launch)).tolist())
                r=solve(self.plant,self.ref,j,f,self.s);before=solve(self.plant,self.ref,j,old,self.s)
                self.assertEqual(r['status'],'nominal_clear',r)
                actual=interpolate_state(np.array(r['prefix_times']),np.array(r['prefix_states']),t+.05)
                accepted,reason=response_usable(r,j['generation'],t+.05,actual,self.s)
                self.assertTrue(accepted,reason)
                if t>8.:self.assertLess(t,valid_until)
                valid_until=r['valid_until_sim_s']
                if before['status']=='nominal_clear':legacy_valid=before['valid_until_sim_s']
                elif t>8. and t>=legacy_valid:legacy_stopped=True
            self.assertTrue(legacy_stopped,'Fixture must reproduce the legacy freshness-gate stop')
            self.assertEqual(f['coverage']['retained_expired_tracks'],5)
            self.assertFalse(f['coverage']['guarantees_free_space'])

    def test_online_mpc_produces_nonlinearly_screened_avoidance(self):
        j=self.job();x=np.asarray(j['state']);launch=np.asarray(j['launch_position_P']);up=launch/np.linalg.norm(launch)
        direction=x[3:6]/np.linalg.norm(x[3:6]);side=np.cross(direction,up);side/=np.linalg.norm(side)
        obstacle=self.ref.at(52.)[0][:3]+side*10.
        s=dict(self.s,solve_wall_s=5.,mpc_candidate_count=5)
        f=frame(obstacle)
        r=solve(self.plant,self.ref,j,f,s)
        self.assertEqual(r['status'],'mpc_feasible',r)
        self.assertTrue(r['plan_assessment']['admissible']);self.assertIn('qp',r)

    def test_stale_result_generation_state_and_deadlines_are_rejected(self):
        state=self.ref.at(40)[0];r=dict(generation=1,status='mpc_feasible',source_sim_s=40.,valid_until_sim_s=44.,
            prefix_times=[40.,43.],prefix_states=[state.tolist(),state.tolist()])
        self.assertFalse(response_usable(r,0,41.,state,self.s)[0])
        self.assertFalse(response_usable(r,1,43.,state,self.s)[0])
        y=state.copy();y[0]+=10;self.assertFalse(response_usable(r,1,41.,y,self.s)[0])
        self.assertTrue(response_usable(r,1,41.,state,self.s)[0])

    def test_stage_boundary_never_sends_previous_screened_command(self):
        s=self.s;p=dict(times=[0.,1.,2.])
        self.assertEqual(send_readiness(p,.99,1.01,s),'stage_changed_before_send')
        self.assertIsNone(send_readiness(p,1.,1.01,s))
        self.assertEqual(send_readiness(p,1.,1.4,s),'send_state_too_old')
        self.assertEqual(send_readiness(None,.99,1.01,s,self.ref),'stage_changed_before_send')

    def test_cache_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'model mismatch'):
                Reference(ROOT/'policy/trajectory_handoff/assets',path)

    def test_cached_coupled_feedback_recovers_a_state_perturbation(self):
        phase=40.;x=self.ref.at(phase)[0];x=retract(x,np.r_[[.3,-.2,.1],[.05,-.03,.02],[.01,-.01,.005],[.005,-.004,.002]])
        prev=self.ref.at(phase)[1];maximum=0.
        for t in np.arange(phase,phase+90.,.1):
            u=self.ref.control(t,x);x=self.plant.transition(x,prev,u,.1);prev=u
            d=error(x,self.ref.at(t+.1)[0]);maximum=max(maximum,float(np.linalg.norm(d[:3])))
        self.assertLess(maximum,1.5);self.assertLess(np.linalg.norm(d[:3]),.3)
        self.assertLess(np.linalg.norm(d[6:9]),.03)

    def test_tailer_does_not_consume_partial_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'public.txt';p.write_text('{"v":1}\n{"v":')
            t=Tail(p);t.read();self.assertEqual(t.pending,[{'v':1}])
            with p.open('a') as f:f.write('2}\n')
            t.read();self.assertEqual(t.pending,[{'v':1},{'v':2}])

    def test_score_equation_and_static_provenance(self):
        c=self.ref.manifest['score_constants'];self.assertAlmostEqual(score(120,75,0,c),50.)
        d,x,target,cc,e=preflight(ROOT,ROOT/'nominal_model.json',ROOT/'policy/trajectory_handoff/assets')
        self.assertEqual(e['radar_num_debris'],200);self.assertEqual(e['spawn_num_debris'],500)
        self.assertEqual(cc,c)

    def test_analysis_retains_failure_before_any_state_and_partial_final_line(self):
        from policy.trajectory_handoff import analyze as mod
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);session=base/'session';attempt=session/'trials'/'trial'/'attempt';attempt.mkdir(parents=True)
            (attempt/'assignment.json').write_text('{}')
            (attempt/'supervisor_result.json').write_text('{"reason":"startup_timeout"}')
            (attempt/'states.txt').write_text('{"incomplete":')
            with patch.object(mod,'__file__',str(base/'analyze.py')):
                report=mod.analyze(session,False)
            self.assertEqual(len(report['attempts']),1)
            self.assertEqual(report['attempts'][0]['result']['reason'],'startup_timeout')
            self.assertFalse(report['attempts'][0]['native_crash_inferred'])

    def test_analysis_distinguishes_unreceived_late_response_and_receipt_latency(self):
        from policy.trajectory_handoff import analyze as mod
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);(attempt/'numeric_worker').mkdir()
            received=dict(job_id='received',source_sim_s=10.,wall_s=.5,status='nominal_clear')
            late=dict(job_id='late',source_sim_s=12.,wall_s=1.4,status='blocked',reason='numeric_budget_exhausted')
            (attempt/'solver_results.txt').write_text(json.dumps(received)+'\n')
            (attempt/'numeric_worker/response.json').write_text(json.dumps(late))
            rows=mod.response_records(attempt)
            self.assertEqual([r['recorded_response_received'] for r in rows],[True,False])
            telemetry=[dict(sim_s=10.8,worker_results=0),dict(sim_s=10.9,worker_results=1)]
            summary=mod.recorded_latency([received],telemetry)
            self.assertAlmostEqual(summary['maximum_source_to_receipt_s'],.8)
            self.assertAlmostEqual(summary['median_outside_numeric_s'],.3)
            self.assertEqual(summary['received_jobs'],1)

    def test_analysis_preserves_unreceived_final_after_received_interim(self):
        from policy.trajectory_handoff import analyze as mod
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);(attempt/'numeric_worker').mkdir()
            early=dict(job_id='same',source_sim_s=10.,response_sequence=1,is_final=False,status='active_clear')
            final=dict(early,response_sequence=2,is_final=True)
            (attempt/'solver_results.txt').write_text(json.dumps(early)+'\n')
            (attempt/'numeric_worker/response.json').write_text(json.dumps(final))
            rows=mod.response_records(attempt)
            self.assertEqual(len(rows),2)
            self.assertEqual([r['recorded_response_received'] for r in rows],[True,False])

    def test_analysis_reconstructs_public_activation_and_keeps_final_per_job(self):
        from policy.trajectory_handoff import analyze as mod
        plan={'times':[20.,35.]}
        response=dict(job_id='plan',source_sim_s=17.2,generation=0,status='mpc_feasible',
            activation_sim_s=20.,plan=plan,recorded_response_received=True)
        events=[dict(event='handoff_activated',mode='linear_mpc',generation=1,sim_s=20.08)]
        active,known=mod.active_plan_at(20.079,1,[response],events,{})
        self.assertTrue(known);self.assertEqual(active,plan)
        self.assertFalse(mod.active_plan_at(22.,2,[response],events,{})[1])
        early=dict(response,job_id='renew',response_sequence=1,is_final=False,status='active_clear')
        final=dict(early,response_sequence=2,is_final=True,recorded_response_received=False)
        rows=mod.planning_responses([early,final])
        self.assertEqual(len(rows),1);self.assertTrue(rows[0]['interim_received'])
        self.assertFalse(rows[0]['recorded_response_received']);self.assertTrue(rows[0]['is_final'])

    def test_fast_path_first_command_precedes_numeric_worker(self):
        from policy.trajectory_handoff import entry
        class Done(BaseException):pass
        order=[];ref=self.ref;state=ref.at(0)[0];m=self.model
        packet=dict(read_begin_sim_s=0.,read_end_sim_s=0.,position_P_m=(state[:3]+matrix(state[6:10]).T@m.ell).tolist(),
            velocity_P_m_s=(state[3:6]+m.vbias).tolist(),q_PB_xyzw_reported=state[6:10].tolist(),
            omega_relP_P_rad_s=(matrix(state[6:10]).T@(state[10:13]+m.wbias)).tolist(),target_P_m=ref.manifest['target_P_m'],fuel_used_kg=0.,damage_J=0.)
        class Stub:
            def __init__(self,*args):order.append('worker');self.started=time.monotonic();self.job_id=None
            def ready(self):return False
            def close(self):order.append('closed')
        with tempfile.TemporaryDirectory() as tmp:
            ctx=types.SimpleNamespace(parameters={'task':'smoke'},model_path=ROOT/'nominal_model.json',root=ROOT,output_dir=Path(tmp),
                checkpoint=lambda **kw:None,state=lambda:packet,clock=lambda:0.,event=lambda *a,**kw:None,
                request_stop_after_task=lambda:None,radar=lambda n:[],sleep=lambda s:(_ for _ in ()).throw(Done()))
            def command(a,b):order.append('command');return dict(send_end_sim_s=0.)
            ctx.command=command
            with patch.object(entry,'Worker',Stub):
                with self.assertRaises(Done):entry.run(ctx)
        self.assertLess(order.index('command'),order.index('worker'));self.assertEqual(order[-1],'closed')

    def test_shared_loader_preflight_in_fresh_interpreter_without_numeric_imports(self):
        code=('from pathlib import Path; import json,sys; '
              'from policy.trajectory_handoff.check import check_entry_loader; '
              'r=check_entry_loader(Path.cwd()); '
              'print(json.dumps(dict(check=r,scipy_loaded="scipy" in sys.modules,casadi_loaded="casadi" in sys.modules)))')
        completed=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30.)
        self.assertEqual(completed.returncode,0,completed.stderr)
        data=json.loads(completed.stdout)
        self.assertTrue(data['check']['run_reached']);self.assertFalse(data['check']['native_started'])
        self.assertFalse(data['scipy_loaded']);self.assertFalse(data['casadi_loaded'])


if __name__=='__main__':unittest.main(verbosity=2)
