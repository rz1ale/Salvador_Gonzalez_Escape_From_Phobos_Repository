"""Functional tests use synthetic public packets, never a native simulator."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
from obstacle_characterization import (
    RadarTracker, RadarStream, load_settings, predict_objects, check_trajectory, rank_threats)
from obstacle_characterization.replay import replay_attempt
from obstacle_characterization.radar import cv_predict, cv_predict_many


def own(t, p=None, v=None, q=None):
    return dict(read_begin_sim_s=float(t), read_end_sim_s=float(t),
        position_P_m=np.zeros(3).tolist() if p is None else np.asarray(p).tolist(),
        velocity_P_m_s=np.zeros(3).tolist() if v is None else np.asarray(v).tolist(),
        q_PB_xyzw_reported=[0.,0.,0.,1.] if q is None else list(q))


def measurement(center, velocity, stamp, ego_p=None, ego_v=None, diameter=2., rng=None):
    ego_p = np.zeros(3) if ego_p is None else np.asarray(ego_p)
    ego_v = np.zeros(3) if ego_v is None else np.asarray(ego_v)
    r = np.asarray(center)-ego_p
    distance = np.linalg.norm(r)
    radial = float((np.asarray(velocity)-ego_v)@(r/distance))
    location = r.copy()
    closest = distance-diameter/2
    if rng is not None:
        location += rng.normal(0, 2., 3)
        closest += rng.normal(0, 2.)
        radial += rng.normal(0, .02)
        diameter += rng.normal(0, .05*diameter)
    return np.array([location, [closest,radial,diameter], [float(stamp),0.,0.]])


def fill_history(tracker, end, moving=False):
    first = tracker.history.samples[-1][0]+.1 if tracker.history.samples else 0.
    for t in np.arange(first, end+1e-7, .1):
        p = [.05*t*t,.1*t,0.] if moving else np.zeros(3)
        v = [.1*t,.1,0.] if moving else np.zeros(3)
        tracker.add_own_state(own(t,p,v))


def query_frame(center=(10.,0.,0.), velocity=(0.,0.,0.), epoch=1.):
    return dict(schema_version=1, frame='Phobos_body_fixed', units='SI', generated_sim_s=epoch,
        acceleration_spectral_density_m2_s3=0.,
        default_query=dict(uncertainty_multiplier=0.,shape_padding_m=0.,forecast_horizon_s=10.),
        coverage=dict(conditions=[],guarantees_free_space=False),
        objects=[dict(id='test_object',kind='track',status='confirmed',state_epoch_sim_s=epoch,
            center_P_m=list(center),velocity_P_m_s=list(velocity),covariance_P=np.zeros((6,6)).tolist(),
            radius_m=1.,radius_sigma_m=0.,extra_position_margin_m=0.)])


class RadarTests(unittest.TestCase):
    def test_cv_covariance_propagation_matches_linear_state_space_solution(self):
        rng=np.random.default_rng(3);A=rng.normal(size=(6,6));P=A@A.T;x=rng.normal(size=6)
        h=.7;q=.02;F=np.eye(6);F[:3,3:]=np.eye(3)*h
        Q=np.kron(np.array([[h**3/3,h**2/2],[h**2/2,h]]),np.eye(3))*q
        xp,Pp=cv_predict(x,P,h,q)
        self.assertTrue(np.allclose(xp,F@x));self.assertTrue(np.allclose(Pp,F@P@F.T+Q))
        xb,Pb=cv_predict_many(np.array([x,x]),np.array([P,P]),np.array([h,h]),q)
        self.assertTrue(np.allclose(xb[0],xp));self.assertTrue(np.allclose(Pb[1],Pp))

    def test_same_epoch_changed_payload_is_not_an_independent_confirmation(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.));fill_history(tr,1.2)
        row=measurement([30.,0.,0.],[0.,0.,0.],1.)
        tr.update([row],1.2);changed=row.copy();changed[0,0]+=.01
        f=tr.update([changed],1.2)
        self.assertEqual(tr.tracks[0].hits,1)
        self.assertEqual(f['diagnostics']['same_epoch_alternative'],1)
        self.assertTrue(any(o['kind']=='unassociated_observation' for o in f['objects']))

    def test_capacity_loss_stays_visible_instead_of_dropping_hazards_silently(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,max_unresolved_observations=1));fill_history(tr,1.1)
        tr.update([measurement([20.,1.,0.],[0.,0.,0.],1.),
                   measurement([20.,-1.,0.],[0.,0.,0.],1.)],1.1)
        fill_history(tr,2.1)
        f=tr.update([measurement([20.,0.,0.],[0.,0.,0.],2.),
                     measurement([20.,.01,0.],[0.,0.,0.],2.)],2.1)
        self.assertIn('unresolved_capacity_loss',f['coverage']['conditions'])
        self.assertEqual(f['coverage']['unresolved_observations'],1)

    def test_no_new_information_from_repeated_packets_or_row_sorting(self):
        tr = RadarTracker(dict(timestamp_offset_s=0.))
        fill_history(tr,1.1)
        rows=[measurement([20.,0.,0.],[0.,0.,0.],1.),
              measurement([0.,40.,0.],[0.,0.,0.],1.,diameter=4.)]
        first=tr.update(rows,1.1);ids=[o['id'] for o in first['objects']]
        P=[t.P.copy() for t in tr.tracks]
        fill_history(tr,1.4)
        second=tr.update(rows[::-1],1.4)
        self.assertEqual(second['diagnostics']['duplicates'],2)
        self.assertEqual([o['id'] for o in second['objects']],ids)
        self.assertEqual([o['independent_observations'] for o in second['objects']],[1,1])
        for a,b in zip(P,tr.tracks):self.assertTrue(np.array_equal(a,b.P))
        json.dumps(second,allow_nan=False)

    def test_ego_motion_compensated_and_unknown_tangential_velocity_is_not_known_zero(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,position_sigma_m=.01))
        center=np.array([30.,10.,5.]);velocity=np.array([.2,-.1,.05])
        frame=None
        for t in np.arange(1.,12.,.5):
            fill_history(tr,t+.05,moving=True)
            ep=np.array([.05*t*t,.1*t,0.]);ev=np.array([.1*t,.1,0.])
            frame=tr.update([measurement(center+velocity*t,velocity,t,ep,ev)],t+.05)
            if t==1.:
                P=np.array(frame['objects'][0]['covariance_P'])
                self.assertGreater(np.linalg.eigvalsh(P[3:,3:]).max(),3.)
        self.assertEqual(len(frame['objects']),1)
        obj=frame['objects'][0]
        self.assertEqual(obj['status'],'confirmed')
        self.assertLess(np.linalg.norm(np.array(obj['center_P_m'])-(center+velocity*frame['generated_sim_s'])),.1)
        self.assertLess(np.linalg.norm(np.array(obj['velocity_P_m_s'])-velocity),.08)

    def test_world_conversion_uses_configured_sensor_origin_once(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,radar_origin_offset_B_m=[1.,0.,0.]))
        # Passive P->B +90deg means body +X is world -Y.
        q=[0.,0.,np.sin(np.pi/4),np.cos(np.pi/4)]
        for t in np.arange(0.,1.11,.1):tr.add_own_state(own(t,[10.,0.,0.],q=q))
        origin=np.array([10.,-1.,0.]);target=np.array([30.,5.,0.])
        frame=tr.update([measurement(target,[0.,0.,0.],1.,origin)],1.1)
        self.assertTrue(np.allclose(frame['objects'][0]['center_P_m'],target))

    def test_auto_clock_is_a_bound_and_never_claims_exact_registration(self):
        tr=RadarTracker();fill_history(tr,2.)
        f=tr.update([measurement([30.,0.,0.],[0.,0.,0.],1.)],2.)
        self.assertEqual(f['clock']['offset_bounds_s'],[0.,1.])
        self.assertFalse(f['clock']['exact'])
        self.assertGreaterEqual(f['objects'][0]['extra_position_margin_m'],10.)
        fill_history(tr,2.8)
        f=tr.update([measurement([30.,0.,0.],[0.,0.,0.],2.)],2.8)
        self.assertAlmostEqual(f['clock']['offset_bounds_s'][1],.8)
        self.assertEqual(f['objects'][0]['last_measurement_sim_s'],2.)
        json.dumps(f,allow_nan=False)

    def test_unstamped_malformed_future_and_missing_history_remain_unassessed(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.));tr.add_own_state(own(1.))
        rows=[measurement([30.,0.,0.],[0.,0.,0.],0.),
              measurement([40.,0.,0.],[0.,0.,0.],3.),
              measurement([50.,0.,0.],[0.,0.,0.],.5),np.zeros((2,2))]
        f=tr.update(rows,1.)
        self.assertEqual(len(f['objects']),0)
        self.assertEqual(f['diagnostics']['initial_unstamped'],1)
        self.assertEqual(f['diagnostics']['future_measurement'],1)
        self.assertEqual(f['diagnostics']['malformed'],1)
        self.assertEqual(f['diagnostics']['own_history_missing'],1)
        risk=check_trajectory(f,[1.,2.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.)
        self.assertEqual(risk['assessment'],'unassessed')
        self.assertFalse(risk['certified_safe'])

    def test_ambiguous_association_preserves_alternative_hazards(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.));fill_history(tr,1.1)
        tr.update([measurement([20.,1.,0.],[0.,0.,0.],1.),
                   measurement([20.,-1.,0.],[0.,0.,0.],1.)],1.1)
        fill_history(tr,2.1)
        f=tr.update([measurement([20.,0.,0.],[0.,0.,0.],2.)],2.1)
        self.assertEqual(f['diagnostics']['ambiguous'],1)
        self.assertEqual(len(f['objects']),3)
        self.assertTrue(any(o['kind']=='unassociated_observation' for o in f['objects']))
        self.assertTrue(all(t.hits==1 for t in tr.tracks))

    def test_old_tracks_expire_without_implying_free_space(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.));fill_history(tr,1.1)
        tr.update([measurement([20.,0.,0.],[0.,0.,0.],1.)],1.1)
        fill_history(tr,14.)
        f=tr.update([],14.)
        self.assertEqual(f['diagnostics']['expired_tracks'],1)
        self.assertFalse(f['coverage']['guarantees_free_space'])
        self.assertEqual(rank_threats(f,vehicle_radius_m=1.)['assessment'],'unassessed')

    def lost_frame(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,track_expiry_mode='retain_occupancy'))
        fill_history(tr,1.1);tr.update([measurement([20.,0.,0.],[0.,0.,0.],1.)],1.1)
        original=copy.deepcopy(tr.tracks[0]);fill_history(tr,14.)
        f=tr.update([measurement([0.,500.,0.],[0.,0.,0.],13.9)],14.)
        return tr,f,original

    def test_retired_occupancy_keeps_identity_and_full_elapsed_covariance(self):
        tr,f,original=self.lost_frame();lost=next(o for o in f['objects'] if o['id']==original.id)
        x,P=cv_predict(original.x,original.P,13.,tr.settings['acceleration_spectral_density_m2_s3'])
        np.testing.assert_allclose(lost['center_P_m'],x[:3],atol=1e-12)
        np.testing.assert_allclose(lost['covariance_P'],P,atol=1e-12)
        self.assertEqual(lost['last_measurement_sim_s'],1.);self.assertEqual(lost['status'],'lost')
        self.assertTrue(lost['occupancy_retained']);self.assertFalse(lost['association_active'])
        self.assertEqual(f['diagnostics']['expired_tracks'],1)
        self.assertEqual(f['coverage']['retained_expired_tracks'],1)
        self.assertNotIn('expired_tracks',f['coverage']['conditions'])
        self.assertFalse(f['coverage']['guarantees_free_space'])
        fill_history(tr,25.);later=tr.update([measurement([0.,500.,0.],[0.,0.,0.],24.9)],25.)
        self.assertTrue(any(o['id']==original.id for o in later['objects']))

    def test_retired_hazard_still_blocks_an_intersecting_path(self):
        tr,f,original=self.lost_frame()
        hit=check_trajectory(f,[14.,15.],[[20.,0.,0.],[20.,0.,0.]],vehicle_radius_m=1.,uncertainty_geometry='ellipsoid')
        self.assertEqual(hit['assessment'],'potential_collision');self.assertEqual(hit['critical_object_id'],original.id)
        far=check_trajectory(f,[14.,15.],[[1000.,1000.,0.],[1000.,1000.,0.]],vehicle_radius_m=1.,uncertainty_geometry='ellipsoid')
        self.assertEqual(far['assessment'],'no_predicted_intersection');self.assertFalse(far['certified_safe'])

    def test_expired_unresolved_alternative_is_retained_with_its_original_epoch(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,track_expiry_mode='retain_occupancy'))
        fill_history(tr,1.1);tr.update([measurement([20.,0.,0.],[0.,0.,0.],1.)],1.1)
        before=tr.update([measurement([20.,.01,0.],[0.,0.,0.],1.)],1.1)
        uid=next(o['id'] for o in before['objects'] if o['kind']=='unassociated_observation')
        fill_history(tr,14.);f=tr.update([measurement([0.,500.,0.],[0.,0.,0.],13.9)],14.)
        obj=next(o for o in f['objects'] if o['id']==uid)
        self.assertEqual(obj['status'],'lost_unresolved');self.assertEqual(obj['last_measurement_sim_s'],1.)
        self.assertEqual(f['coverage']['retained_expired_unresolved_observations'],1)

    def test_retaining_hazards_does_not_clear_invalid_observation_conditions(self):
        tr,_,_=self.lost_frame();fill_history(tr,14.2)
        f=tr.update([measurement([0.,500.,0.],[0.,0.,0.],14.1),[[1.]]],14.2)
        self.assertIn('malformed',f['coverage']['conditions'])
        q=check_trajectory(f,[14.2,15.],[[1000.,1000.,0.],[1000.,1000.,0.]],vehicle_radius_m=1.)
        self.assertEqual(q['assessment'],'unassessed')

    def test_retained_occupancy_capacity_loss_stays_flagged_after_legacy_window(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,track_expiry_mode='retain_occupancy',max_unresolved_observations=1))
        fill_history(tr,1.1);tr.update([measurement([20.,0.,0.],[0.,0.,0.],1.)],1.1)
        f=tr.update([measurement([20.,.01,0.],[0.,0.,0.],1.),measurement([20.,-.01,0.],[0.,0.,0.],1.)],1.1)
        self.assertIn('retained_occupancy_capacity_loss',f['coverage']['conditions'])
        fill_history(tr,20.);f=tr.update([measurement([0.,500.,0.],[0.,0.,0.],19.9)],20.)
        self.assertIn('retained_occupancy_capacity_loss',f['coverage']['conditions'])
        self.assertLessEqual(len(tr.unresolved)+len(tr.retired_unresolved),1)

    def test_retired_and_active_tracks_share_the_declared_capacity(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,track_expiry_mode='retain_occupancy',max_tracks=500))
        fill_history(tr,1.1);tr.update([measurement([20.,0.,0.],[0.,0.,0.],1.)],1.1)
        for i in range(499):
            old=copy.deepcopy(tr.tracks[0]);old.id='retired_fixture_'+str(i);tr.retired_tracks[old.id]=old
        fill_history(tr,2.1);f=tr.update([measurement([0.,500.,0.],[0.,0.,0.],2.)],2.1)
        self.assertEqual(len(tr.tracks)+len(tr.retired_tracks),500)
        self.assertEqual(f['diagnostics']['capacity_rejected'],1)
        self.assertEqual(len(f['objects']),501)

    def test_unknown_expiry_mode_is_rejected(self):
        with self.assertRaises(ValueError):load_settings(dict(track_expiry_mode='ignore'))
        with self.assertRaises(ValueError):load_settings(dict(birth_initialization='ignore'))

    def test_two_position_birth_resolves_a_unique_radial_rejection_with_cross_covariance(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,birth_initialization='two_position'))
        fill_history(tr,1.1)
        tr.update([measurement([100.,0.,0.],[3.,0.,0.],1.)],1.1)
        identity=tr.tracks[0].id;first_variance=tr.tracks[0].P[0,0]
        fill_history(tr,3.1);f=tr.update([measurement([96.,6.,0.],[-2.,3.,0.],3.)],3.1)
        self.assertEqual(len(tr.tracks),1);self.assertEqual(tr.tracks[0].id,identity)
        self.assertEqual(f['diagnostics']['birth_reinitializations'],1)
        self.assertEqual(tr.tracks[0].hits,2);self.assertEqual(tr.tracks[0].velocity_initialization,'two_position')
        np.testing.assert_allclose(tr.tracks[0].x[3:],[-2.,3.,0.],atol=1e-10)
        tangent=np.array([-6.,96.,0.]);tangent/=np.linalg.norm(tangent)
        new_variance=tr.settings['position_sigma_m']**2+tr.settings['ego_position_sigma_m']**2
        expected=(first_variance+new_variance)/4+tr.settings['acceleration_spectral_density_m2_s3']*2/3
        self.assertAlmostEqual(float(tangent@tr.tracks[0].P[3:,3:]@tangent),expected)
        self.assertAlmostEqual(float(tangent@tr.tracks[0].P[:3,3:]@tangent),new_variance/2)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(tr.tracks[0].P).min()),-1e-10)

    def test_two_position_birth_does_not_force_an_ambiguous_association(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,birth_initialization='two_position'))
        fill_history(tr,1.1)
        tr.update([measurement([100.,.5,0.],[3.,0.,0.],1.),measurement([100.,-.5,0.],[3.,0.,0.],1.)],1.1)
        fill_history(tr,3.1);f=tr.update([measurement([96.,0.,0.],[-2.,0.,0.],3.)],3.1)
        self.assertEqual(f['diagnostics']['ambiguous'],1)
        self.assertEqual(f['diagnostics']['birth_reinitializations'],0)
        self.assertTrue(all(t.hits==1 for t in tr.tracks));self.assertEqual(len(f['objects']),3)

    def test_two_position_birth_does_not_reset_a_mature_track(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.,birth_initialization='two_position'))
        for t in [1.,2.,3.]:
            fill_history(tr,t+.1);tr.update([measurement([100.,0.,0.],[0.,0.,0.],t)],t+.1)
        identity=tr.tracks[0].id;self.assertEqual(tr.tracks[0].hits,3)
        fill_history(tr,4.1);f=tr.update([measurement([100.,0.,0.],[6.,0.,0.],4.)],4.1)
        self.assertEqual(f['diagnostics']['birth_reinitializations'],0)
        self.assertEqual(next(t for t in tr.tracks if t.id==identity).hits,3)
        self.assertEqual(len(tr.tracks),2)

    def test_actual_multitarget_noise_and_shuffle_produce_finite_estimates(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.));rng=np.random.default_rng(42)
        centers=np.array([[40+30*i,10*i+5,3*i+8] for i in range(6)],float)
        velocities=np.array([[.2+.02*i,-.1,.03] for i in range(6)])
        for t in np.arange(1.,31.,.5):
            fill_history(tr,t+.05,moving=True)
            ep=np.array([.05*t*t,.1*t,0.]);ev=np.array([.1*t,.1,0.])
            rows=[measurement(c+v*t,v,t,ep,ev,diameter=2+i,rng=rng) for i,(c,v) in enumerate(zip(centers,velocities))]
            rng.shuffle(rows);frame=tr.update(rows,t+.05)
        for c,v in zip(centers,velocities):
            truth=c+v*frame['generated_sim_s']
            obj=min(frame['objects'],key=lambda o:np.linalg.norm(np.array(o['center_P_m'])-truth))
            self.assertLess(np.linalg.norm(np.array(obj['center_P_m'])-truth),3.)
            self.assertLess(np.linalg.norm(np.array(obj['velocity_P_m_s'])-v),.6)
            self.assertGreaterEqual(np.linalg.eigvalsh(np.array(obj['covariance_P'])).min(),-1e-8)
        json.dumps(frame,allow_nan=False)

    def test_clock_reversal_and_invalid_settings_fail(self):
        with self.assertRaises(ValueError):load_settings(dict(unknown_option=True))
        with self.assertRaises(ValueError):load_settings(dict(position_sigma_m=0.))
        tr=RadarTracker();tr.add_own_state(own(2.))
        with self.assertRaisesRegex(ValueError,'future observations'):tr.update([],1.)
        with self.assertRaisesRegex(ValueError,'backwards'):tr.add_own_state(own(1.))

    def test_slightly_future_sensor_time_is_rejected_without_negative_prediction(self):
        tr=RadarTracker(dict(timestamp_offset_s=0.));fill_history(tr,1.)
        f=tr.update([measurement([30.,0.,0.],[0.,0.,0.],1.01)],1.)
        self.assertEqual(f['diagnostics']['future_measurement'],1)
        self.assertEqual(f['objects'],[])

    def test_optional_runner_smoke_entry_reads_only_and_requests_normal_stop(self):
        from obstacle_characterization.smoke import run
        class Finished(BaseException):
            pass
        with tempfile.TemporaryDirectory() as temp:
            clock=[1.]
            stopped=[]
            captured=[]
            def finish(**value):
                captured.append(value)
                raise Finished()
            ctx=types.SimpleNamespace(root=temp,output_dir=Path(temp)/'logs',parameters={'duration_s':1.},
                clock=lambda:clock[0],state=lambda:own(clock[0]),
                radar=lambda n:[measurement([30.,0.,0.],[0.,0.,0.],clock[0]-.1)],
                request_stop_after_task=lambda:stopped.append(True),
                sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds),finish=finish)
            with self.assertRaises(Finished):run(ctx)
            self.assertEqual(stopped,[True])
            self.assertEqual(captured[0]['outcome'],'success')
            self.assertEqual(captured[0]['reason'],'obstacle_stream_smoke_complete')

    def test_stream_adapter_uses_only_existing_public_reads_and_poll_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            calls=dict(state=0,radar=0);clock=[1.]
            def state():calls['state']+=1;return own(clock[0])
            def radar(count):
                calls['radar']+=1;self.assertEqual(count,500)
                return [measurement([30.,0.,0.],[0.,0.,0.],clock[0]-.1)]
            context=types.SimpleNamespace(root=temp,output_dir=Path(temp)/'logs',state=state,radar=radar,clock=lambda:clock[0])
            stream=RadarStream(context,dict(timestamp_offset_s=0.))
            first=stream.poll()
            clock[0]=1.1;cached=stream.poll()
            self.assertIs(first,cached)
            clock[0]=1.25;frame=stream.poll(own_state=own(1.25))
            self.assertEqual(calls,dict(state=2,radar=2))
            self.assertTrue(frame['objects'])
            self.assertTrue((context.output_dir/'obstacle_diagnostics.txt').exists())
            self.assertFalse((context.output_dir/'obstacle_stream.txt').exists())

    def test_replay_is_causal_and_emits_standard_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);states=[own(t) for t in np.arange(0.,5.11,.1)]
            packets=[dict(read_sim_s=float(t+.05),observations=[measurement([30.,0.,0.],[0.,0.,0.],t).tolist()])
                     for t in np.arange(1.,5.01,.5)]
            for name,values in [('states.txt',states),('radar.txt',packets)]:
                (path/name).write_text('\n'.join(json.dumps(v) for v in values))
            result=replay_attempt(path,dict(timestamp_offset_s=0.),path/'stream.txt')
            frames=[json.loads(s) for s in (path/'stream.txt').read_text().splitlines()]
            self.assertEqual(result['radar_frames'],9)
            self.assertEqual(result['maximum_confirmed_tracks'],1)
            self.assertEqual(frames[-1]['schema_version'],1)
            self.assertFalse(frames[-1]['coverage']['guarantees_free_space'])


class CollisionTests(unittest.TestCase):
    def test_solver_array_forecast_matches_standard_json_forecast(self):
        f=query_frame(velocity=(.2,0.,0.));f['objects'][0]['center_P_m']=np.array([10.,0.,0.])
        standard=predict_objects(f,[1.,2.,3.])
        array=predict_objects(f,[1.,2.,3.],array_output=True)
        self.assertEqual(array['numeric_representation'],'numpy')
        self.assertIsInstance(array['objects'][0]['center_P_m'],np.ndarray)
        for key in ('center_P_m','position_covariance_P','inflated_obstacle_radius_m'):
            self.assertTrue(np.allclose(standard['objects'][0][key],array['objects'][0][key]))
        json.dumps(standard,allow_nan=False)

    def test_swept_check_detects_collision_between_clear_endpoints(self):
        f=query_frame(center=(-10.,0.,0.),velocity=(20.,0.,0.))
        result=check_trajectory(f,[1.,2.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.)
        self.assertEqual(result['assessment'],'potential_collision')
        self.assertAlmostEqual(result['objects'][0]['critical_time_sim_s'],1.5)
        self.assertAlmostEqual(result['minimum_inflated_clearance_m'],-2.)

    def test_clearance_knob_is_monotone_and_all_track_types_included(self):
        f=query_frame(center=(10.,0.,0.))
        f['objects'][0].update(kind='unassociated_observation',status='unresolved')
        a=check_trajectory(f,[1.,2.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.)
        b=check_trajectory(f,[1.,2.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.,clearance_m=3.)
        self.assertAlmostEqual(a['minimum_inflated_clearance_m']-b['minimum_inflated_clearance_m'],3.)
        self.assertEqual(a['objects'][0]['kind'],'unassociated_observation')

    def test_uncertainty_growth_and_interpolation_padding_affect_collision(self):
        f=query_frame(center=(0.,3.,0.));P=np.eye(6)*.01
        f['objects'][0]['covariance_P']=P.tolist()
        f['default_query']['uncertainty_multiplier']=3.
        prediction=predict_objects(f,[1.,2.,3.])
        radii=prediction['objects'][0]['inflated_obstacle_radius_m']
        self.assertTrue(np.all(np.diff(radii)>0))
        f['default_query']['uncertainty_multiplier']=0.
        a=check_trajectory(f,[1.,3.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.)
        b=check_trajectory(f,[1.,3.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.,vehicle_acceleration_bound_m_s2=4.)
        self.assertEqual(a['assessment'],'no_predicted_intersection')
        self.assertEqual(b['assessment'],'potential_collision')

    def test_unknown_coverage_or_horizon_never_returns_clear(self):
        f=query_frame();f['coverage']['conditions']=['unresolved_capacity_loss']
        result=check_trajectory(f,[1.,2.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.)
        self.assertEqual(result['assessment'],'unassessed')
        result=check_trajectory(f,[1.,20.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=1.)
        self.assertIn('forecast_horizon_exceeded',result['conditions'])
        with self.assertRaises(ValueError):predict_objects(f,[1.,20.])

    def test_wrong_frames_negative_radius_and_invalid_covariance_fail(self):
        f=query_frame();f['frame']='body'
        with self.assertRaises(ValueError):predict_objects(f,[1.,2.])
        f=query_frame()
        with self.assertRaises(ValueError):check_trajectory(f,[1.,2.],[[0.,0.,0.],[0.,0.,0.]],vehicle_radius_m=-1.)
        f['objects'][0]['covariance_P'][0][0]=-1.
        with self.assertRaisesRegex(ValueError,'positive semidefinite'):predict_objects(f,[1.,2.])


if __name__=='__main__':
    unittest.main()
