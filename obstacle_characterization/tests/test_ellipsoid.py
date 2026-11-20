"""Directional ellipsoid support: analytic, continuous and frame-invariant checks."""
import copy
import unittest
import numpy as np
from scipy.optimize import minimize
from obstacle_characterization.collision import (check_trajectory,ellipsoid_separating_normal,
    position_support_sigma,predict_objects)


def frame(center=(100.,0.,0.),covariance=None):
    P=np.zeros((6,6)) if covariance is None else np.asarray(covariance,float)
    return dict(schema_version=1,frame='Phobos_body_fixed',units='SI',generated_sim_s=1.,
        acceleration_spectral_density_m2_s3=0.,default_query=dict(uncertainty_multiplier=3.,shape_padding_m=0.,forecast_horizon_s=20.),
        coverage=dict(conditions=[],guarantees_free_space=False),objects=[dict(id='synthetic',kind='track',status='tentative',
            state_epoch_sim_s=1.,center_P_m=list(center),velocity_P_m_s=[0.,0.,0.],covariance_P=P.tolist(),
            radius_m=2.,radius_sigma_m=0.,extra_position_margin_m=0.)])


class EllipsoidTests(unittest.TestCase):
    def query(self,f,positions,**kw):
        return check_trajectory(f,[1.,2.],positions,vehicle_radius_m=2.,uncertainty_geometry='ellipsoid',**kw)

    def test_tangential_uncertainty_is_not_replaced_by_radial_uncertainty(self):
        P=np.zeros((6,6));P[:3,:3]=np.diag([4.,10000.,10000.]);f=frame(covariance=P);original=copy.deepcopy(f)
        sphere=check_trajectory(f,[1.,2.],[[0.,0.,0.]]*2,vehicle_radius_m=2.)
        ell=self.query(f,[[0.,0.,0.]]*2)
        self.assertEqual(sphere['assessment'],'potential_collision');self.assertEqual(ell['assessment'],'no_predicted_intersection')
        self.assertAlmostEqual(ell['minimum_inflated_clearance_m'],90.)
        self.assertEqual(f,original);self.assertFalse(ell['certified_safe'])

    def test_radial_uncertainty_that_reaches_vehicle_remains_a_hazard(self):
        P=np.zeros((6,6));P[:3,:3]=np.diag([1600.,4.,4.])
        r=self.query(frame(covariance=P),[[0.,0.,0.]]*2)
        self.assertEqual(r['assessment'],'potential_collision')

    def test_between_knot_contact_is_detected(self):
        f=frame(center=[10.,0.,0.]);r=self.query(f,[[0.,0.,0.],[20.,0.,0.]])
        self.assertEqual(r['assessment'],'potential_collision');self.assertLess(r['minimum_inflated_clearance_m'],0.)

    def test_closest_normal_matches_independent_ellipsoid_projection(self):
        rng=np.random.default_rng(42);Q,_=np.linalg.qr(rng.normal(size=(3,3)))
        axes=np.array([1.,5.,12.]);P=Q@np.diag(axes**2)@Q.T;k=2.;r=np.array([25.,10.,30.]);local=Q.T@r
        n=ellipsoid_separating_normal(r,P,k);gap=float(n@r-k*position_support_sigma(P,n))
        fun=lambda z:.5*np.sum((z-local)**2)
        fit=minimize(fun,np.zeros(3),jac=lambda z:z-local,method='SLSQP',
            constraints=[dict(type='ineq',fun=lambda z:1.-np.sum((z/(k*axes))**2),jac=lambda z:-2*z/(k*axes)**2)],
            options={'maxiter':200,'ftol':1e-11})
        self.assertTrue(fit.success,fit.message);self.assertAlmostEqual(gap,np.linalg.norm(fit.x-local),places=5)
        self.assertAlmostEqual(float(np.linalg.norm(n)),1.)

    def test_continuous_support_bound_with_negative_cross_covariance(self):
        P=np.block([[4*np.eye(3),-1.5*np.eye(3)],[-1.5*np.eye(3),np.eye(3)]])
        f=frame([70.,5.,-10.],P);f['acceleration_spectral_density_m2_s3']=.01
        positions=np.array([[0.,0.,0.],[5.,0.,0.]])
        r=check_trajectory(f,[1.,11.],positions,vehicle_radius_m=2.,clearance_m=3.,uncertainty_geometry='ellipsoid')
        o=r['objects'][0];n=np.array(o['separating_normal_P']);samples=[]
        for t in np.linspace(0,10,501):
            pp=positions[0]+(positions[1]-positions[0])*t/10
            C=P[:3,:3]+t*(P[:3,3:]+P[3:,:3])+t*t*P[3:,3:]+.01*t**3/3*np.eye(3)
            samples.append(n@(pp-np.array(f['objects'][0]['center_P_m']))-7.-3*position_support_sigma(C,n))
        self.assertLessEqual(r['minimum_inflated_clearance_m'],min(samples)+1e-8)
        self.assertGreater(r['minimum_inflated_clearance_m'],0.)

    def test_rigid_frame_rotation_preserves_the_query(self):
        rng=np.random.default_rng(4);R,_=np.linalg.qr(rng.normal(size=(3,3)));A=rng.normal(size=(6,6));P=A@A.T
        f=frame([30.,5.,-10.],P);f['objects'][0]['velocity_P_m_s']=[-1.,.2,.1]
        pos=np.array([[0.,1.,0.],[2.,2.,1.]]);a=self.query(f,pos)
        g=copy.deepcopy(f);T=np.block([[R,np.zeros((3,3))],[np.zeros((3,3)),R]])
        o=g['objects'][0];o['center_P_m']=(R@np.array(o['center_P_m'])).tolist();o['velocity_P_m_s']=(R@np.array(o['velocity_P_m_s'])).tolist();o['covariance_P']=(T@P@T.T).tolist()
        b=self.query(g,pos@R.T);self.assertAlmostEqual(a['minimum_inflated_clearance_m'],b['minimum_inflated_clearance_m'],places=7)

    def test_every_retained_object_and_geometry_margin_remain_present(self):
        f=frame();o=copy.deepcopy(f['objects'][0]);o.update(id='near',center_P_m=[1.,0.,0.],status='unresolved');f['objects'].append(o)
        r=self.query(f,[[0.,0.,0.]]*2);self.assertEqual(len(r['objects']),2);self.assertEqual(r['assessment'],'potential_collision')
        predicted=predict_objects(f,[1.,2.]);self.assertTrue(all('extent_radius_with_margins_m' in o for o in predicted['objects']))

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):position_support_sigma(np.eye(3),[2.,0.,0.])
        with self.assertRaises(ValueError):ellipsoid_separating_normal([1.,0.,0.],-np.eye(3))
        with self.assertRaises(ValueError):self.query(frame(),[[0.,0.,0.]]*2,uncertainty_multiplier=-1.)


if __name__=='__main__':unittest.main(verbosity=2)
