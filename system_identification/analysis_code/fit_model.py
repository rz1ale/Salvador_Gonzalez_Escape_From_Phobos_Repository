"""Local gray-box fit, causal free-run replay, model export and scientific plots.

Training only determines parameters. Declared validation flights are evaluated
after freezing. Testing remains sealed unless --include-testing is explicit.
With a training-only scout the outputs say TRAINING REPLAY, not validation.
"""
from __future__ import annotations
import os
for _thread_name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_name,'1')
import argparse
import csv
import datetime as dt
import json
from pathlib import Path, PurePosixPath
import sys
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sim_core.common import digest, json_hash, local_path, read_json, write_json
from system_identification.model import Allocator, Model, MODEL_TYPE, matrix, gravity, load_jet_matrix
from system_identification.analysis_code.summarize import telemetry
from system_identification.campaigns.identification import signature

def resolve_jet_table(root,explicit=None):
    """Read the active official actuator's static asset reference; never scan CSVs."""
    root=Path(root).resolve();data=root.parent/'STU4-Sim/Data'
    if explicit is not None:
        table=Path(explicit)
        if not table.is_absolute():table=root/table
        table=table.resolve()
    else:
        config_path=root.parent/'STU4-Sim/Sims/COMPETITION_SIM.simconf'
        config=read_json(config_path)
        systems=config.get('SystemManager',{}).get('Systems',[])
        if not isinstance(systems,list):raise ValueError('Official SystemManager.Systems must be a list')
        references=set()
        for system in systems:
            if not isinstance(system,dict) or system.get('Active') is False:continue
            source=PurePosixPath(str(system.get('Source','')).replace('\\','/')).name
            if system.get('Nametag')!='JetSelectionActuator' and source!='JetSelectionActuator.py':continue
            parameters=system.get('Inst_Parameters',{})
            reference=parameters.get('JetConfigName') if isinstance(parameters,dict) else None
            if isinstance(reference,str) and reference.strip():references.add(reference.strip().replace('\\','/'))
        if len(references)!=1:
            raise ValueError('Official active JetSelectionActuator.JetConfigName is missing or ambiguous in '+str(config_path)+
                '; supply --jet-table PATH to a confirmed current static table')
        reference=references.pop();parts=PurePosixPath(reference).parts
        if parts[:4]!=('Local','Repos','STU4-Sim','Data') or '..' in parts:
            raise ValueError('Unsupported JetConfigName outside current official Data: '+reference)
        table=root.parent.joinpath(*parts[2:]).resolve()
    if not (table.is_relative_to(root) or table.is_relative_to(data.resolve())):
        raise ValueError('Jet table must be in this solution clone or its current official sibling Data directory')
    if not table.is_file():
        raise ValueError('Referenced current jet table missing: '+str(table)+'; check official JetConfigName; do not substitute an old table')
    return table

def cumulative(t,values):
    shape=(len(t)-1,)+(1,)*(values.ndim-1)
    return np.concatenate((np.zeros_like(values[:1]),np.cumsum(.5*(values[1:]+values[:-1])*np.diff(t).reshape(shape),axis=0)))

def interpolate(t,values,query):
    if values.ndim==1:return np.interp(query,t,values)
    return np.column_stack([np.interp(query,t,values[:,j]) for j in range(values.shape[1])])

class Flight:
    def __init__(self,session,record):
        self.directory=local_path(session,record['directory']);self.record=record
        self.metadata=read_json(self.directory/'metadata.json');self.profile=self.metadata['profile']
        self.quality=read_json(self.directory/'quality.json');rows=telemetry(self.directory/'telemetry.txt')
        self.id=record['trial_id']+'__'+record['attempt_id'];self.split=self.profile['split']
        self.t=np.array([float(r['t_sim_s']) for r in rows])
        keys=('x_P_m','y_P_m','z_P_m','vx_P_m_s','vy_P_m_s','vz_P_m_s',
              'qPB_x','qPB_y','qPB_z','qPB_w','wx_B_rad_s','wy_B_rad_s','wz_B_rad_s')
        self.x=np.array([[float(r[k]) for k in keys] for r in rows])
        self.fuel=np.array([float(r['fuel_kg']) for r in rows]);damage=np.array([float(r['damage_J']) for r in rows])
        cutoff=self.quality.get('last_clean_cutoff_sim_s',self.t[-1] if len(self.t) else 0.)
        keep=self.t<=cutoff;self.t=self.t[keep];self.x=self.x[keep];self.fuel=self.fuel[keep];damage=damage[keep]
        if len(self.t)<5 or not np.isfinite(self.t).all() or not np.isfinite(self.x).all() or np.any(np.diff(self.t)<0):raise ValueError('Invalid telemetry: '+self.id)
        if np.any(np.linalg.norm(self.x[:,6:10],axis=1)<1e-8):raise ValueError('Zero reported quaternion: '+self.id)
        unique=np.r_[np.diff(self.t)>0,True];self.t=self.t[unique];self.x=self.x[unique];self.fuel=self.fuel[unique]
        self.x[:,6:10]/=np.linalg.norm(self.x[:,6:10],axis=1)[:,None]
        for i in range(1,len(self.x)):
            if self.x[i,6:10]@self.x[i-1,6:10]<0:self.x[i,6:10]*=-1
        commands=[json.loads(l) for l in (self.directory/'commands.txt').read_text().splitlines() if l.strip()]
        self.ct=np.array([.5*(r['send_begin_sim_s']+r['send_end_sim_s']) for r in commands])
        self.u=np.array([r['linear_body_request_m_s2']+r['angular_body_request_rad_s2'] for r in commands],float)
        if not len(self.ct) or self.u.shape!=(len(self.ct),6) or not np.isfinite(self.u).all() or not np.isfinite(self.ct).all() or np.any(np.diff(self.ct)<0):raise ValueError('Invalid request history: '+self.id)
        self.segments=[]
        for interval in sorted(self.quality.get('clean_intervals',[]),key=lambda p:p['start_sim_s']):
            a=max(interval['start_sim_s'],self.t[0]);b=min(interval['end_sim_s'],self.t[-1],self.quality.get('last_clean_cutoff_sim_s',self.t[-1]))
            if b<=a:continue
            if self.segments and a-self.segments[-1][1]<=.15:self.segments[-1][1]=max(b,self.segments[-1][1])
            else:self.segments.append([a,b])
        if not self.segments:raise ValueError('No completed clean interval: '+self.id)
        # Recheck only retained clean data; raw post-cutoff tails are not fitted.
        settings=self.metadata.get('settings',{})
        for a,b in self.segments:
            use=(self.t>=a)&(self.t<=b);indices=np.flatnonzero(use)
            if not len(indices):continue
            if not np.isfinite(damage).all() or np.any(damage>1e-6):raise ValueError('Positive pre-cutoff damage: '+self.id)
            if len(indices)>1 and np.max(np.diff(self.t[indices]))>settings.get('sample_gap_max_s',.6):raise ValueError('Clean interval has a telemetry gap: '+self.id)
        self.observed_level=False
        if self.profile.get('family')=='saturation':
            axis=self.profile['axis'];target=np.zeros(6);target[axis]=self.profile['sign']*self.profile['amplitude']
            exposure=0.
            for ci in range(len(self.ct)-1):
                if np.allclose(self.u[ci],target,rtol=1e-8,atol=1e-12):
                    exposure+=sum(max(0.,min(b,self.ct[ci+1])-max(a,self.ct[ci])) for a,b in self.segments)
            self.observed_level=exposure>=.05
        self.nodes=[]
        for a,b in self.segments:
            ix=np.flatnonzero((self.t>=a)&(self.t<=b))
            if len(ix)<12:continue
            # Public sample cadence retained; windows use integral balances.
            tx=self.t[ix];xx=self.x[ix];windows=[]
            for i in range(0,len(ix)-6,max(1,int(.25/max(np.median(np.diff(tx)),.005)))):
                j=int(np.searchsorted(tx,tx[i]+.75))
                if j<len(tx):windows.append((i,j))
            if windows:self.nodes.append(dict(t=tx,x=xx,windows=windows))
    def values(self,t):
        x=interpolate(self.t,self.x,t)
        x[:,6:10]/=np.linalg.norm(x[:,6:10],axis=1)[:,None]
        return x

def fit_reporting(flights):
    xp=[];yp=[];xw=[];yw=[]
    for f in flights:
        for node in f.nodes:
            t,x=node['t'],node['x'];R=np.array([matrix(q).T for q in x[:,6:10]])
            iv=cumulative(t,x[:,3:6]);iw=cumulative(t,x[:,10:13])
            for i,j in node['windows']:
                d=t[j]-t[i]
                xp.extend(np.c_[R[j]-R[i],x[j,3:6]-x[i,3:6],-np.eye(3)*d])
                yp.extend(x[j,:3]-x[i,:3]-(iv[j]-iv[i]))
                turn=-Rotation.from_matrix(R[j].T@R[i]).as_rotvec()
                xw.extend(np.c_[x[j,10:13]-x[i,10:13],-np.eye(3)*d])
                yw.extend(turn-(iw[j]-iw[i]))
    if len(yp)<30:raise ValueError('Too few independent free-flight integral windows')
    X=np.array(xp);y=np.array(yp);W=np.array(xw);z=np.array(yw)
    fp=lsq_linear(X,y,bounds=(np.r_[[-5.]*3,0.,[-.05]*3],np.r_[[5.]*3,.15,[.05]*3]),tol=1e-10)
    fw=lsq_linear(W,z,bounds=(np.r_[0.,[-.01]*3],np.r_[.15,[.01]*3]),tol=1e-10)
    return dict(reporting_offset_B_m=fp.x[:3].tolist(),velocity_report_delay_s=float(fp.x[3]),
        velocity_report_bias_P_m_s=fp.x[4:].tolist(),rate_report_delay_s=float(fw.x[0]),
        rate_report_bias_B_rad_s=fw.x[1:].tolist()),dict(position_balance_rank=int(np.linalg.matrix_rank(X)),
        position_balance_columns=7,position_balance_rmse_m=float(np.sqrt(np.mean((X@fp.x-y)**2))),
        rate_balance_rank=int(np.linalg.matrix_rank(W)),rate_balance_columns=4,
        rate_balance_rmse_rad=float(np.sqrt(np.mean((W@fw.x-z)**2))),
        note='Effective reporting point and delays, not independently established COM or SDK constants.')

def prepare_nodes(flights,reporting,allocator,mu,minimum_period=.005):
    nodes=[]
    for f in flights:
        print('Preparing command map:',f.id,flush=True)
        # Skip ground/staging commands; retain a causal input-history prefix.
        begin=min(a for a,b in f.segments)-.3;end=max(b for a,b in f.segments)+.2
        ci=max(0,np.searchsorted(f.ct,begin,side='right')-1);cj=np.searchsorted(f.ct,end,side='right')
        ct=f.ct[ci:cj];commands=np.array([allocator.clip(u) for u in f.u[ci:cj]])
        peaks=allocator.peaks(commands,minimum_period)
        for node in f.nodes:
            original_t=node['t'];count=int(np.searchsorted(original_t,original_t[-1]-max(
                reporting['velocity_report_delay_s'],reporting['rate_report_delay_s']),side='right'))
            t,x=original_t[:count],node['x'][:count]
            windows=[(i,j) for i,j in node['windows'] if j<count]
            if len(t)<5 or not windows:continue
            v=interpolate(f.t,f.x[:,3:6],t+reporting['velocity_report_delay_s'])-reporting['velocity_report_bias_P_m_s']
            w=interpolate(f.t,f.x[:,10:13],t+reporting['rate_report_delay_s'])-reporting['rate_report_bias_B_rad_s']
            R=np.array([matrix(q).T for q in x[:,6:10]])
            pc=x[:,:3]-np.einsum('nij,j->ni',R,reporting['reporting_offset_B_m'])
            gp=np.array([gravity(p,mu,np.zeros(3)) for p in pc])
            J=np.array(f.metadata['inertia_B_kg_m2']);Ji=np.linalg.inv(J)
            gy=np.array([-Ji@np.cross(omega,J@omega) for omega in w])
            nodes.append(dict(t=t,R=R,v=v,w=w,gp=gp,gyro=gy,windows=windows,ct=ct,u=commands,peaks=peaks))
    return nodes

def regress(nodes,period,delay,angular=False):
    X=[];y=[]
    for n in nodes:
        t=n['t'];edges=n['ct']+delay
        grid=np.unique(np.r_[t,edges[(edges>t[0])&(edges<t[-1])]])
        mid=.5*(grid[1:]+grid[:-1]);ix=np.searchsorted(n['ct'],mid-delay,side='right')-1
        base=n['u']/np.maximum(period,n['peaks'])[:,None]
        sampled=base[np.maximum(ix,0)].copy();sampled[ix<0]=0
        dt_grid=np.diff(grid);at=np.searchsorted(grid,t)
        if angular:
            integ=np.r_[np.zeros((1,3)),np.cumsum(sampled[:,3:]*dt_grid[:,None],axis=0)][at]
            gyro=cumulative(t,n['gyro'])
        else:
            Rmid=interpolate(t,n['R'].reshape(len(t),9),mid).reshape(len(mid),3,3)
            features=np.einsum('nij,nk->nijk',Rmid,sampled[:,:3]).reshape(len(mid),3,9)
            integ=np.concatenate((np.zeros((1,3,9)),np.cumsum(features*dt_grid[:,None,None],axis=0)),axis=0)[at]
            known=cumulative(t,n['gp'])
        weight=1/np.sqrt(max(1,len(n['windows'])))
        for i,j in n['windows']:
            d=t[j]-t[i]
            if angular:
                phi=integ[j]-integ[i]
                rows=np.c_[np.kron(np.eye(3),phi[None,:]),np.eye(3)*d,(gyro[j]-gyro[i])[:,None]]
                target=n['w'][j]-n['w'][i]
            else:
                rows=np.c_[integ[j]-integ[i],np.eye(3)*d]
                target=n['v'][j]-n['v'][i]-(known[j]-known[i])
            X.extend(rows*weight/d);y.extend(target*weight/d)
    X=np.asarray(X);y=np.asarray(y)
    prior=np.r_[np.eye(3).ravel(),np.zeros(4 if angular else 3)]
    low=np.r_[np.full(9,-.5),np.full(3,-.05),0.] if angular else np.r_[np.full(9,-.5),np.full(3,-.05)]
    high=np.r_[np.full(9,.5),np.full(3,.05),2.] if angular else np.r_[np.full(9,.5),np.full(3,.05)]
    for i in (0,4,8):low[i]=.3;high[i]=3.
    norm=np.linalg.norm(X,axis=0);active=norm>max(norm.max()*1e-8,1e-10)
    # A nearly unexcited gyro coefficient is fixed, rather than fitted to noise.
    if angular and np.sqrt(np.mean(X[:,-1]**2))<1e-4:active[-1]=False
    columns=np.flatnonzero(active);scale=norm[active]
    A=X[:,active]/scale
    ridge=.002
    lhs=np.r_[A,np.sqrt(ridge)*np.eye(len(columns))]
    rhs=np.r_[y-X@prior,np.zeros(len(columns))]
    fit=lsq_linear(lhs,rhs,bounds=((low[active]-prior[active])*scale,(high[active]-prior[active])*scale),tol=1e-8)
    theta=prior.copy();theta[active]+=fit.x/scale
    residual=X@theta-y
    score=float(np.mean(residual**2)/max(np.mean(y*y),1e-5))
    condition=float(np.linalg.cond(A))
    return theta,score,dict(rows=len(y),rank=int(np.linalg.matrix_rank(A)),estimated_columns=columns.tolist(),
        fixed_prior_columns=np.flatnonzero(~active).tolist(),
        columns=len(columns),normalized_condition=condition if np.isfinite(condition) else None,
        increment_rate_component_rmse=float(np.sqrt(np.mean(residual**2))),
        active_bounds=[int(i) for i in np.flatnonzero(active&((abs(theta-low)<1e-5)|(abs(theta-high)<1e-5)))])

def fit_parameters(flights,allocator,mu):
    reporting,diagnostics=fit_reporting(flights)
    nodes=prepare_nodes(flights,reporting,allocator,mu)
    if not nodes:raise ValueError('No usable fitted windows')
    best=None
    hs=np.array([.005,.01,.015,.02,.025,.035,.05]);delays=np.array([0.,.01,.025,.05,.08,.12,.15])
    for stage in range(3):
        choices=[]
        for h in hs:
            channels=[]
            for angular in (False,True):
                options=[]
                for delay in delays:
                    theta,score,info=regress(nodes,float(h),float(delay),angular)
                    options.append((score,float(delay),theta,info))
                channels.append(min(options,key=lambda a:a[0]))
            choice=(sum(c[0] for c in channels),float(h),channels)
            choices.append(choice)
        selected=min(choices,key=lambda a:a[0])
        if best is None or selected[0]<best[0]:best=selected
        print('Training-only parameter search',stage+1,'period',best[1],'score',best[0],flush=True)
        radius=.004/(stage+1)
        hs=np.unique(np.clip(best[1]+np.array([-1,-.5,0,.5,1])*radius,.005,.05))
        centers=[c[1] for c in best[2]]
        delays=np.unique(np.clip(np.r_[centers[0]+np.array([-.01,0,.01])/(stage+1),centers[1]+np.array([-.01,0,.01])/(stage+1)],0,.15))
    _,h,(linear,angular)=best
    p=dict(reporting,mean_native_period_s=h,linear_effect_delay_s=linear[1],angular_effect_delay_s=angular[1],
        linear_gain=linear[2][:9].reshape(3,3).tolist(),uniform_external_acceleration_P_m_s2=linear[2][9:12].tolist(),
        angular_gain=angular[2][:9].reshape(3,3).tolist(),angular_bias_B_rad_s2=angular[2][9:12].tolist(),
        gyro_coefficient=float(angular[2][12]),phobos_mu_prior_m3_s2=mu)
    return p,dict(reporting=diagnostics,linear=linear[3],angular=angular[3],training_objective=best[0],
        note='Mean-period/gain/delay estimates are effective and can remain correlated; bound-active or weak-rank fits require review.')

def errors(pred,truth):
    angle=(Rotation.from_quat(pred[:,6:10])*Rotation.from_quat(truth[:,6:10]).inv()).as_rotvec()
    return np.c_[pred[:,:6]-truth[:,:6],angle,pred[:,10:13]-truth[:,10:13]]

def error_metrics(error):
    out={}
    for i,name in enumerate(('position_m','velocity_m_s','attitude_rad','rate_rad_s')):
        value=np.linalg.norm(error[:,3*i:3*i+3],axis=1)
        out[name]=dict(rmse=float(np.sqrt(np.mean(value**2))),peak=float(value.max()),terminal=float(value[-1]))
    return out

def plot_flight(f,t,truth,pred,folder,label):
    import matplotlib.pyplot as plt
    e=errors(pred,truth);elapsed=t-t[0]
    fig,axes=plt.subplots(4,3,figsize=(13,11),sharex=True)
    relative_truth=(Rotation.from_quat(truth[:,6:10])*Rotation.from_quat(truth[0,6:10]).inv()).as_rotvec()
    relative_pred=(Rotation.from_quat(pred[:,6:10])*Rotation.from_quat(truth[0,6:10]).inv()).as_rotvec()
    groups=[(truth[:,:3]-truth[0,:3],pred[:,:3]-truth[0,:3],'position [m]'),
            (truth[:,3:6],pred[:,3:6],'velocity [m/s]'),(relative_truth,relative_pred,'relative attitude [rad]'),
            (truth[:,10:13],pred[:,10:13],'body rate [rad/s]')]
    for row,(a,b,name) in enumerate(groups):
        for j in range(3):
            axes[row,j].plot(elapsed,a[:,j],label='reported truth');axes[row,j].plot(elapsed,b[:,j],'--',label='prediction')
            axes[row,j].grid(alpha=.3);axes[row,j].set_ylabel(name+' '+str(j))
    axes[0,0].legend();fig.suptitle(label+' / '+f.id+' / one causal initialization, no truth resets')
    fig.supxlabel('seconds since causal initialization')
    fig.tight_layout();fig.savefig(folder/(f.id+'_truth_prediction.png'),dpi=130);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
    for i,name in enumerate(('position [m]','velocity [m/s]','attitude geodesic [rad]','rate [rad/s]')):
        axes[i].plot(elapsed,np.linalg.norm(e[:,i*3:i*3+3],axis=1));axes[i].set_ylabel(name);axes[i].grid(alpha=.3)
    axes[-1].set_xlabel('seconds since causal initialization');fig.suptitle(label+' ERROR NORMS / '+f.id)
    fig.tight_layout();fig.savefig(folder/(f.id+'_errors.png'),dpi=130);plt.close(fig)

def evaluate(f,model,output,label):
    folder=output/label;folder.mkdir(exist_ok=True)
    a,b=max(f.segments,key=lambda ab:ab[1]-ab[0]);ix=np.flatnonzero((f.t>=a+.3)&(f.t<=b))
    if len(ix)<5:raise ValueError('Clean segment too short for free-run evaluation: '+f.id)
    ix=ix[::2];t=f.t[ix];truth=f.x[ix]
    pred=model.predict_reported(f.t,f.x,t[0],t,f.ct,f.u)
    e=errors(pred,truth);metrics=error_metrics(e);horizons=[]
    for desired in np.arange(a+.3,b-1.,5.):
        i=int(np.searchsorted(f.t,desired));t0=f.t[i]
        hs=[h for h in (1.,2.,5.,10.,20.) if t0+h<=b]
        if not hs:continue
        times=t0+np.array(hs);forecast=model.predict_reported(f.t,f.x,t0,times,f.ct,f.u)
        reference=f.values(times);err=errors(forecast,reference)
        for h,row in zip(hs,err):
            horizons.append(dict(initialization_s=float(t0),horizon_s=h,**{k:float(np.linalg.norm(row[i*3:i*3+3]))
                for i,k in enumerate(('position_m','velocity_m_s','attitude_rad','rate_rad_s'))}))
    write_json(folder/(f.id+'_metrics.json'),dict(cohort=label,full_clean_free_run=metrics,horizon_errors=horizons))
    np.savez_compressed(folder/(f.id+'_predictions.npz'),time_s=t,reported_truth=truth,prediction=pred)
    plot_flight(f,t,truth,pred,folder,label)
    return dict(trial_id=f.record['trial_id'],attempt_id=f.record['attempt_id'],cohort=label,
        censored_flight=bool(f.quality.get('censored')),span_s=float(t[-1]-t[0]),
        metrics=metrics,horizons=horizons)

def fit_session(root,session,output,jet_table=None,mu=707500.,solver='revised simplex',include_testing=False,publish=False):
    # Fail early rather than silently omit the requested scientific plots.
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError as exc:raise RuntimeError('Model fitting plots require Matplotlib in the analysis Python environment') from exc
    root,session,output=Path(root),Path(session),Path(output)
    if not np.isfinite(mu) or mu<0:raise ValueError('Phobos mu prior must be finite and nonnegative')
    plan=read_json(session/'plan.json');ledger=read_json(session/'ledger.json')
    if plan['config']['entry']!='system_identification/campaigns/identification.py':raise ValueError('Not a system-ID session')
    flights=[];excluded=[]
    for record in ledger['attempts']:
        directory=local_path(session,record['directory'])
        if not (directory/'metadata.json').is_file():excluded.append(dict(attempt=record['attempt_id'],reason='no acquisition metadata'));continue
        profile=read_json(directory/'metadata.json')['profile']
        if profile['split']=='smoke' or profile['split']=='testing' and not include_testing:continue
        try:flights.append(Flight(session,record))
        except (ValueError,KeyError,FileNotFoundError) as exc:excluded.append(dict(attempt=record['attempt_id'],reason=str(exc)))
    training=[f for f in flights if f.split=='training']
    if not training:raise ValueError('No usable training flights')
    for axis in range(6):
        for sign in (-1,1):
            levels={f.profile.get('amplitude') for f in training if f.nodes and f.observed_level and f.profile.get('family')=='saturation' and
                    f.profile.get('axis')==axis and f.profile.get('sign')==sign}
            if len(levels)<2:raise ValueError(f'Need two training request levels for axis {axis}, sign {sign}; run scout/full campaign')
    signatures={f.metadata['signature'] for f in flights}
    if len(signatures)!=1:raise ValueError('Mixed acquisition signatures; do not combine incompatible configurations')
    expected=signature(root,'system_identification/campaigns/settings.json')
    if next(iter(signatures))!=expected:
        raise ValueError('Current official/code/settings fingerprint differs from acquisition; use matching references, not a changed simulator epoch')
    first=training[0].metadata;mass=first['mass_kg'];J=np.array(first['inertia_B_kg_m2'])
    if any(not np.allclose(f.metadata['inertia_B_kg_m2'],J) or abs(f.metadata['mass_kg']-mass)>1e-6 for f in flights):
        raise ValueError('Mass/inertia changed across flights; constant-inertia model is incompatible')
    table=resolve_jet_table(root,jet_table)
    print('Current static jet table:',table,flush=True)
    A,forces=load_jet_matrix(table,mass,J);allocator=Allocator(A,forces,solver)
    output.mkdir(parents=True,exist_ok=False)
    selection=dict(training=[f.id for f in training],validation=[f.id for f in flights if f.split=='validation'],
        testing=[f.id for f in flights if f.split=='testing'],excluded=excluded,testing_unsealed=include_testing)
    write_json(output/'data_selection.json',selection)
    params,diagnostics=fit_parameters(training,allocator,mu)
    import scipy
    definition=dict(schema_version=1,model_type=MODEL_TYPE,state_dimension=13,input_dimension=6,
        state_order=['p_C_P[3]','v_C_P[3]','passive_q_PB_xyzw[4]','omega_relP_B[3]'],
        input_kind='public_body_acceleration_requests',input_units=['m/s^2']*3+['rad/s^2']*3,
        mass_kg=mass,inertia_B_kg_m2=J.tolist(),parameters=params,integration_max_step_s=.01,
        actuator=dict(jet_matrix=A.tolist(),jet_force_N=forces.tolist(),lp_solver=solver,request_norm_caps=[.5,.05],
            law='clip request norms; minimum-sum nonnegative LP; mean response b/max(fitted_mean_period,max(allocation))'),
        assumptions=['Continuous mean approximation to unknown native impulse timing; effective period is fitted, not read.',
            'Phobos monopole prior plus fitted uniform external acceleration; no private ephemeris or actuator telemetry.',
            'Frame angular transport neglected; local-domain candidate, not equivalent to a full dual-ephemeris model.',
            'Effective reporting point is not independently established physical COM.',
            'Nonzero command delays require timestamped history; step() assumes a previously held constant input.'],
        provenance=dict(session=str(session),acquisition_signature=next(iter(signatures)),selection=selection,
            analysis_versions=dict(numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
            jet_table_source=str(table),jet_table_selection='official JetConfigName' if jet_table is None else 'explicit --jet-table',
            jet_table_sha256=digest(table),fit_source_sha256=digest(__file__),backend_source_sha256=digest(root/'system_identification/model.py'),
            inputs={str(f.directory.relative_to(session)/n):digest(f.directory/n) for f in flights for n in ('telemetry.txt','commands.txt','quality.json','metadata.json')}),
        status='candidate_requires_independent_validation')
    parameter_hash=json_hash(params);definition['frozen_parameter_sha256']=parameter_hash
    write_json(output/'model.json',definition);write_json(output/'fit_diagnostics.json',diagnostics)
    model=Model(definition);results=[]
    for f in flights:
        cohort='training_replay' if f.split=='training' else f.split
        print('Free-run prediction:',cohort,f.id,flush=True)
        results.append(evaluate(f,model,output,cohort))
        if json_hash(model.p)!=parameter_hash:raise RuntimeError('Parameters changed during evaluation')
        write_json(output/'metrics.json',dict(frozen_parameter_sha256=parameter_hash,flights=results))
    independent=sum(r['cohort']=='validation' for r in results)
    definition['validation']=dict(independent_flights=independent,testing_flights=sum(r['cohort']=='testing' for r in results),
        status='independent_validation_evaluated_review_metrics' if independent else 'UNAVAILABLE_TRAINING_REPLAY_ONLY')
    write_json(output/'model.json',definition)
    with (output/'metrics.txt').open('w',encoding='utf-8',newline='') as stream:
        fields=['trial_id','attempt_id','cohort','span_s','position_rmse_m','velocity_rmse_m_s','attitude_rmse_rad','rate_rmse_rad_s']
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer.writeheader()
        for r in results:
            writer.writerow(dict(trial_id=r['trial_id'],attempt_id=r['attempt_id'],cohort=r['cohort'],span_s=r['span_s'],
                **{key:r['metrics'][name]['rmse'] for key,name in zip(fields[4:],('position_m','velocity_m_s','attitude_rad','rate_rad_s'))}))
    flags=[]
    for channel in ('linear','angular'):
        info=diagnostics[channel]
        if info['rank']<info['columns']:flags.append(channel+': rank deficient parameters')
        if info['active_bounds']:flags.append(channel+': parameter bounds active '+str(info['active_bounds']))
        if info['fixed_prior_columns']:flags.append(channel+': weak columns fixed at prior '+str(info['fixed_prior_columns']))
        if info['normalized_condition'] is None or info['normalized_condition']>1e6:flags.append(channel+': weak conditioning')
    if params['mean_native_period_s'] in (.005,.05):flags.append('Effective period at search boundary')
    definition['fit_review_flags']=flags;write_json(output/'model.json',definition)
    text=['MODEL FIT / CAUSAL FREE-RUN REPORT',f'Model: {output/"model.json"}',f'Independent validation flights: {independent}',
        definition['validation']['status'],f'Frozen parameter hash: {parameter_hash}',
        'Fit review flags: '+('; '.join(flags) or 'none under declared numerical checks'),
        'Training only selected parameters/delays. Future truth was used only to score each free run.',
        'Recorded future public requests are replay inputs; no truth resets inside a forecast.',
        'Review fit_diagnostics.json for rank, active bounds and correlated parameters.',
        'See metrics.txt/metrics.json and cohort folders for truth-prediction/error plots.',
        'State/output conventions and local constant external gravity remain explicit modeling assumptions.',
        'Fuel law and process noise are not identified by this stage.',
        'No score, collision guarantee or model qualification is inferred from these metrics.']
    (output/'report.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if publish:
        destination=root/'system_identification/nominal_model.json'
        write_json(destination,definition)
        print('Published explicit candidate selection:',destination,flush=True)
    return output

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    selected=parser.add_mutually_exclusive_group(required=True)
    selected.add_argument('--session');selected.add_argument('--latest',action='store_true')
    parser.add_argument('--out');parser.add_argument('--jet-table',type=Path)
    parser.add_argument('--mu-phobos',type=float,default=707500.)
    parser.add_argument('--lp-solver',default='revised simplex')
    parser.add_argument('--include-testing',action='store_true',help='Explicitly unseal testing data after fitting; never used for fit')
    parser.add_argument('--publish',action='store_true',help='Explicitly select exported candidate as system_identification/nominal_model.json')
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args(argv)
    name=read_json(ROOT/'sim_core/state/active_session.json')['session'] if args.latest else args.session.replace('\\','/')
    session=local_path(ROOT,name);stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output=local_path(ROOT,args.out) if args.out else ROOT/'system_identification/analysis_results'/session.name/('model_'+stamp)
    result=fit_session(ROOT,session,output,args.jet_table,args.mu_phobos,args.lp_solver,args.include_testing,args.publish)
    if args.show:print((result/'report.txt').read_text(encoding='utf-8'))
    print(result)
    return 0

if __name__=='__main__':raise SystemExit(main())
