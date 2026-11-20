"""Identification/replay tests against independently integrated synthetic flights.

These establish software behavior, not agreement with native STPro data.
"""
from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from sim_core.common import read_json, write_json, digest
from system_identification.campaigns.identification import signature
from system_identification.model import Model, MODEL_TYPE, Allocator, load_jet_matrix, matrix
from system_identification.analysis_code.fit_model import Flight, fit_parameters, fit_session, evaluate, resolve_jet_table

PACKAGE=Path(__file__).resolve().parents[2]


def jet_table(path):
    rows=[]
    for axis in range(3):
        for arm in (-1,1):
            for sign in (-1,1):
                position=np.zeros(3);position[(axis+1)%3]=arm
                normal=np.zeros(3);normal[axis]=-sign
                rows.append(np.r_[position,normal,340.])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(path,rows,delimiter=',',header='x,y,z,nx,ny,nz,thrust',comments='')


def definition(A,forces):
    return dict(schema_version=1,model_type=MODEL_TYPE,inertia_B_kg_m2=np.diag([3000.,6000.,7000.]).tolist(),
        integration_max_step_s=.01,actuator=dict(jet_matrix=A.tolist(),jet_force_N=forces.tolist(),
            lp_solver='revised simplex',request_norm_caps=[.5,.05]),
        parameters=dict(linear_gain=np.diag([1.03,.98,1.01]).tolist(),angular_gain=np.diag([.99,1.02,1.01]).tolist(),
            reporting_offset_B_m=[-.2,.03,-1.2],velocity_report_delay_s=.04,rate_report_delay_s=.04,
            velocity_report_bias_P_m_s=[0.,0.,0.],rate_report_bias_B_rad_s=[0.,0.,0.],
            uniform_external_acceleration_P_m_s2=[.002,-.001,.001],angular_bias_B_rad_s2=[0.,0.,0.],
            mean_native_period_s=.02,linear_effect_delay_s=.025,angular_effect_delay_s=.025,
            gyro_coefficient=0.,phobos_mu_prior_m3_s2=0.))


def independent_truth(d,allocator,t,ct,commands):
    """DOP853 continuous integration; does not call model stepping/rotation code."""
    p=d['parameters'];Kl=np.array(p['linear_gain']);Ka=np.array(p['angular_gain'])
    g=np.array(p['uniform_external_acceleration_P_m_s2']);base=[allocator.mean(u,p['mean_native_period_s']) for u in commands]
    delay=p['linear_effect_delay_s']
    assert delay==p['angular_effect_delay_s']
    initial=np.r_[0.,0.,12000.,np.zeros(3),0.,0.,0.,1.,np.zeros(3)]
    edges=np.unique(np.r_[0.,ct+delay,t[-1]])
    edges=edges[(edges>=0)&(edges<=t[-1])];values=np.zeros((len(t),13));current=initial
    values[0]=initial
    for left,right in zip(edges[:-1],edges[1:]):
        idx=np.searchsorted(ct,(left+right)/2-delay,side='right')-1
        command=base[idx] if idx>=0 else np.zeros(6)
        def rhs(clock,x):
            q=x[6:10];w=x[10:13]
            accel=Rotation.from_quat(q).inv().apply(Kl@command[:3])+g
            qv=-.5*(q[3]*w+np.cross(w,q[:3]));qw=.5*w@q[:3]
            return np.r_[x[3:6],accel,qv,qw,Ka@command[3:]]
        result=solve_ivp(rhs,(left,right),current,rtol=1e-10,atol=1e-11,dense_output=True,method='DOP853')
        use=(t>left+1e-10)&(t<=right+1e-10)
        if np.any(use):values[use]=result.sol(t[use]).T
        current=result.y[:,-1]
    reported=values.copy()
    reported[:,:3]+=Rotation.from_quat(values[:,6:10]).inv().apply(np.array(p['reporting_offset_B_m']))
    for sl,key in ((slice(3,6),'velocity_report_delay_s'),(slice(10,13),'rate_report_delay_s')):
        reported[:,sl]=np.column_stack([np.interp(t-p[key],t,values[:,j]) for j in range(sl.start,sl.stop)])
    return reported


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'STU4-Solution';self.root.mkdir()
        for name in ('run_simulation.py','sim_core','system_identification'):
            src,dst=PACKAGE/name,self.root/name
            if src.is_dir():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','data_sets','analysis_results','state'))
            else:shutil.copyfile(src,dst)
        official=self.root.parent/'STU4-Sim/Sims';official.mkdir(parents=True)
        self.config=official/'COMPETITION_SIM.simconf'
        write_json(self.config,dict(synthetic_fixture_only=True,SystemManager=dict(Systems=[dict(
            Active=True,Nametag='JetSelectionActuator',Source='Local/Repos/STU4-Sim/Systems/JetSelectionActuator.py',
            Inst_Parameters=dict(JetConfigName='Local/Repos/STU4-Sim/Data/SpaceBuggyJetConfigNoMains.csv'))])))
        self.table=self.root.parent/'STU4-Sim/Data/SpaceBuggyJetConfigNoMains.csv';jet_table(self.table)
        self.J=np.diag([3000.,6000.,7000.]);self.A,self.forces=load_jet_matrix(self.table,3388.7,self.J)
        self.d=definition(self.A,self.forces);self.allocator=Allocator(self.A,self.forces)
    def tearDown(self):self.temp.cleanup()
    def test_jet_table_follows_official_asset_reference_without_guessing_export_name(self):
        before=digest(self.config)
        (self.table.parent/'jet_config.csv').write_text('old export; never read')
        self.assertEqual(resolve_jet_table(self.root),self.table.resolve())
        renamed=self.table.with_name('CurrentJetGeometry.csv');self.table.rename(renamed)
        config=read_json(self.config)
        config['SystemManager']['Systems'][0]['Inst_Parameters']['JetConfigName']='Local/Repos/STU4-Sim/Data/CurrentJetGeometry.csv'
        write_json(self.config,config);current=digest(self.config)
        self.assertEqual(resolve_jet_table(self.root),renamed.resolve())
        self.assertEqual(digest(self.config),current)
        self.assertNotEqual(before,current)  # Only this test intentionally edits its synthetic fixture.
    def test_jet_table_missing_ambiguous_or_old_clone_reference_is_rejected(self):
        self.table.unlink();(self.table.parent/'jet_config.csv').write_text('decoy')
        with self.assertRaisesRegex(ValueError,'Referenced current jet table missing'):resolve_jet_table(self.root)
        config=read_json(self.config);first=config['SystemManager']['Systems'][0]
        second=copy.deepcopy(first);second['Inst_Parameters']['JetConfigName']='Local/Repos/STU4-Sim/Data/Other.csv'
        config['SystemManager']['Systems'].append(second);write_json(self.config,config)
        with self.assertRaisesRegex(ValueError,'ambiguous'):resolve_jet_table(self.root)
        config['SystemManager']['Systems']=[first]
        first['Inst_Parameters']['JetConfigName']='Local/Repos/STU4-Solution-old/jet_config.csv';write_json(self.config,config)
        with self.assertRaisesRegex(ValueError,'outside current official Data'):resolve_jet_table(self.root)
        with self.assertRaisesRegex(ValueError,'this solution clone'):resolve_jet_table(self.root,self.root.parent/'STU4-Solution-old/jet_config.csv')
    def test_jet_table_explicit_relative_copy_and_inactive_system_handling(self):
        config=read_json(self.config);inactive=copy.deepcopy(config['SystemManager']['Systems'][0])
        inactive.update(Active=False,Inst_Parameters=dict(JetConfigName='Local/Repos/STU4-Sim/Data/Inactive.csv'))
        config['SystemManager']['Systems'].append(inactive);write_json(self.config,config)
        self.assertEqual(resolve_jet_table(self.root),self.table.resolve())
        copy_path=self.root/'system_identification/current_geometry.csv';shutil.copyfile(self.table,copy_path)
        self.assertEqual(resolve_jet_table(self.root,'system_identification/current_geometry.csv'),copy_path.resolve())
    def test_public_request_scale_shared_saturation_and_inverse_residual(self):
        m=Model(self.d)
        low=np.array([.0002,-.0001,0.,0.,0.,0.])
        self.assertTrue(np.allclose(m.mean_acceleration(low)[:3],np.array(self.d['parameters']['linear_gain'])@low[:3]/.02))
        high=np.array([.5,0,0,0,0,0]);mean=m.mean_acceleration(high)
        self.assertLess(mean[0],.5/.02);self.assertGreater(mean[0],.01)
        mixed=high.copy();mixed[5]=.05
        self.assertLess(m.mean_acceleration(mixed)[0],mean[0])
        desired=np.array([.004,-.003,.002,.001,-.001,.002]);result=m.request_for_mean_acceleration(desired)
        self.assertLess(np.linalg.norm(result['residual']),1e-10)
        self.assertGreater(np.linalg.norm(m.request_for_mean_acceleration(np.ones(6)*10)['residual']),1.)
    def test_delayed_command_edges_and_passive_rotation(self):
        m=Model(self.d);x=np.r_[0.,0.,12000.,np.zeros(3),0.,0.,0.,1.,np.zeros(3)]
        commands=np.array([[0.]*6,[.0004,0,0,0,0,0],[0.]*6]);ct=np.array([0.,1.,2.])
        out=m.rollout(x,0.,np.array([1.02,1.025,2.025,3.]),ct,commands)
        gravity=np.array(self.d['parameters']['uniform_external_acceleration_P_m_s2'])
        self.assertAlmostEqual(out[0,3],gravity[0]*1.02,places=9)
        self.assertAlmostEqual(out[-1,3],gravity[0]*3.+.0004/.02*1.03,places=9)
        x[10:13]=[0,0,.1]
        rotated=m.step(x,np.zeros(6),2.)
        expected=Rotation.from_rotvec([0,0,-.2]).as_matrix()
        self.assertTrue(np.allclose(matrix(rotated[6:10]),expected,atol=1e-10))
        self.assertAlmostEqual(np.linalg.norm(rotated[6:10]),1.)
    def test_causal_initialization_and_embedded_json_portability(self):
        m=Model(self.d);t=np.arange(0.,4.001,.04);ct=np.array([0.,1.,2.]);u=np.zeros((3,6));u[1]=[.001,0,0,0,.001,0]
        truth=independent_truth(self.d,self.allocator,t,ct,u)
        pred=m.predict_reported(t,truth,.4,np.array([1.,2.,3.]),ct,u)
        changed=truth.copy();changed[t>.4,:3]+=1e5;changed[t>.4,3:6]*=30;changed[t>.4,6:10]=[0,0,1,0]
        self.assertTrue(np.array_equal(pred,m.predict_reported(t,changed,.4,np.array([1.,2.,3.]),ct,u)))
        err=pred-truth[[25,50,75]]
        self.assertLess(np.max(np.linalg.norm(err[:,:3],axis=1)),2e-4)
        candidate=self.root/'candidate.json';write_json(candidate,self.d);self.table.unlink()
        self.assertTrue(np.allclose(Model.load(candidate).mean_acceleration(u[1]),m.mean_acceleration(u[1])))
    def test_invalid_models_fail_instead_of_emitting_predictions(self):
        d=copy.deepcopy(self.d);d['parameters']['mean_native_period_s']=0.
        with self.assertRaisesRegex(ValueError,'period'):Model(d)
        d=copy.deepcopy(self.d);d['parameters']['linear_gain'][0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'linear_gain'):Model(d)
        with self.assertRaisesRegex(ValueError,'quaternion'):matrix(np.zeros(4))
    def make_flight(self,session,index,profile,ct,u,signature_value):
        directory=session/f'attempt_{index:03d}';directory.mkdir(parents=True)
        t=np.round(np.arange(0.,4.001,.04),9);x=independent_truth(self.d,self.allocator,t,ct,u)
        keys=('x_P_m','y_P_m','z_P_m','vx_P_m_s','vy_P_m_s','vz_P_m_s','qPB_x','qPB_y','qPB_z','qPB_w','wx_B_rad_s','wy_B_rad_s','wz_B_rad_s')
        rows=[]
        for ti,xi in zip(t,x):
            r=dict(t_sim_s=ti,read_begin_s=ti,read_end_s=ti,send_begin_s=ti,send_end_s=ti,phase='synthetic',kind='science',
                h_launch_m=25.,fuel_kg=0.,damage_J=0.,**dict(zip(keys,xi)))
            rows.append(r)
        with (directory/'telemetry.txt').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
        command_records=[dict(send_begin_sim_s=float(ti),send_end_sim_s=float(ti),linear_body_request_m_s2=ui[:3].tolist(),
                              angular_body_request_rad_s2=ui[3:].tolist()) for ti,ui in zip(ct,u)]
        (directory/'commands.txt').write_text('\n'.join(json.dumps(r) for r in command_records))
        write_json(directory/'metadata.json',dict(profile=profile,signature=signature_value,mass_kg=3388.7,inertia_B_kg_m2=self.J.tolist()))
        write_json(directory/'quality.json',dict(censored=False,last_clean_cutoff_sim_s=4.,clean_intervals=[dict(start_sim_s=0.,end_sim_s=4.)]))
        return dict(trial_id=f'T{index:03d}',attempt_id=f'A{index:03d}',directory=directory.name)
    def synthetic_session(self):
        session=self.root/'system_identification/data_sets/model_fixture';session.mkdir(parents=True)
        source_signature=signature(self.root,'system_identification/campaigns/settings.json');records=[]
        ct=np.array([0.,1.123,2.123])
        for axis in range(6):
            for sign in (-1,1):
                for amp in ((.0002,.5) if axis<3 else (.0001,.05)):
                    u=np.zeros((3,6));u[1,axis]=sign*amp
                    profile=dict(split='training',family='saturation',axis=axis,sign=sign,amplitude=amp)
                    records.append(self.make_flight(session,len(records),profile,ct,u,source_signature))
        u=np.zeros((3,6));u[1]=[.001,-.0008,.0003,.0005,-.0004,.0003]
        records.append(self.make_flight(session,len(records),dict(split='validation',family='validation'),ct,u,source_signature))
        sealed=session/'sealed';sealed.mkdir();write_json(sealed/'metadata.json',dict(profile=dict(split='testing')))
        (sealed/'telemetry.txt').write_text('deliberately unreadable sealed testing trajectory')
        records.append(dict(trial_id='TEST',attempt_id='SEALED',directory='sealed'))
        write_json(session/'plan.json',dict(config=dict(entry='system_identification/campaigns/identification.py')))
        write_json(session/'ledger.json',dict(state='complete',attempts=records))
        return session,records
    def test_fit_recovers_mean_scale_and_exports_real_metrics_and_plots(self):
        session,records=self.synthetic_session()
        # A deliberately wrong held-out trajectory must affect validation
        # metrics, never selected training parameters or the validation label.
        validation=session/records[-2]['directory']/'telemetry.txt'
        with validation.open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
        for r in rows:
            if float(r['t_sim_s'])>2.:r['x_P_m']=str(float(r['x_P_m'])+1000.)
        with validation.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
        output=self.root/'system_identification/analysis_results/fit_fixture'
        # Real fit/replay for every flight. Render one figure pair to keep the
        # test bounded; the CLI renders every selected flight without this patch.
        from system_identification.analysis_code import fit_model as module
        real_plot=module.plot_flight;rendered=[]
        def once(*args):
            if not rendered:real_plot(*args);rendered.append(True)
        native_before=digest(self.root.parent/'STU4-Sim/Sims/COMPETITION_SIM.simconf')
        with patch.object(module,'plot_flight',side_effect=once):
            fit_session(self.root,session,output,mu=0.,publish=True)
        d=read_json(output/'model.json');p=d['parameters'];metrics=read_json(output/'metrics.json')['flights']
        self.assertAlmostEqual(p['mean_native_period_s'],.02,delta=.002)
        self.assertTrue(np.allclose(np.diag(p['linear_gain']),[1.03,.98,1.01],atol=.04),p)
        self.assertTrue(np.allclose(np.diag(p['angular_gain']),[.99,1.02,1.01],atol=.04),p)
        self.assertAlmostEqual(p['velocity_report_delay_s'],.04,delta=.015)
        self.assertAlmostEqual(p['rate_report_delay_s'],.04,delta=.015)
        self.assertEqual(d['validation']['independent_flights'],1)
        self.assertEqual(len(d['provenance']['selection']['training']),24)
        self.assertEqual(d['provenance']['selection']['testing'],[])
        self.assertEqual(len(metrics),25)
        self.assertLess(max(r['metrics']['position_m']['rmse'] for r in metrics if r['cohort']=='training_replay'),.01)
        self.assertGreater(next(r['metrics']['position_m']['rmse'] for r in metrics if r['cohort']=='validation'),100.)
        self.assertLess(max(r['metrics']['attitude_rad']['rmse'] for r in metrics),.001)
        self.assertEqual(len(list(output.rglob('*.png'))),2)
        self.assertTrue((output/'metrics.txt').stat().st_size>300)
        self.assertEqual(digest(self.root.parent/'STU4-Sim/Sims/COMPETITION_SIM.simconf'),native_before)
        self.assertEqual(read_json(self.root/'system_identification/nominal_model.json'),d)
        self.assertEqual(d['frozen_parameter_sha256'],read_json(output/'metrics.json')['frozen_parameter_sha256'])
    def test_clean_prefix_excludes_corrupt_tail_and_sign_flips_are_equivalent(self):
        session=self.root/'system_identification/data_sets/prefix';session.mkdir(parents=True)
        source_signature=signature(self.root,'system_identification/campaigns/settings.json')
        u=np.zeros((3,6));u[1,0]=.5;ct=np.array([0.,1.,2.])
        record=self.make_flight(session,0,dict(split='training',family='saturation',axis=0,sign=1,amplitude=.5),ct,u,source_signature)
        directory=session/record['directory'];q=read_json(directory/'quality.json');q.update(last_clean_cutoff_sim_s=3.,censored=True)
        write_json(directory/'quality.json',q)
        path=directory/'telemetry.txt'
        with path.open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
        for i,r in enumerate(rows):
            if i%2:
                for key in ('qPB_x','qPB_y','qPB_z','qPB_w'):r[key]=str(-float(r[key]))
            if float(r['t_sim_s'])>3.:r['vx_P_m_s']='nan';r['damage_J']='10.'
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
        f=Flight(session,record)
        self.assertEqual(f.t[-1],3.);self.assertTrue(f.observed_level)
        self.assertTrue(np.all(np.sum(f.x[1:,6:10]*f.x[:-1,6:10],axis=1)>0))
        self.assertTrue(np.isfinite(f.values(np.array([1.02]))).all())

if __name__=='__main__':unittest.main()
