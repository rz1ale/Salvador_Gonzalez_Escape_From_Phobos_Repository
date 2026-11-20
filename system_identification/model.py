"""Self-contained 13-state mean-allocation model; no simulator API imports.

State is effective-point p_P,v_P, passive q_PB xyzw, body omega (13 scalars).
Inputs are PUBLIC body acceleration requests (6), not achieved accelerations.
JSON embeds the jet matrix. No old clone, table, ephemeris or live allocator is read.
"""
from __future__ import annotations
import json
import copy
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize import linprog

MODEL_TYPE='public_request_mean_allocation_6dof_v1'

def matrix(q):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:raise ValueError('Expected finite nonzero xyzw quaternion')
    q=q/np.linalg.norm(q);x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def multiply(a,b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),a[3]*b[3]-a[:3]@b[:3]]

def increment(q,omega,dt):
    v=-np.asarray(omega)*dt;angle=np.linalg.norm(v)
    dq=np.r_[v*(np.sin(angle/2)/angle if angle>1e-12 else .5),np.cos(angle/2)]
    out=multiply(dq,q)
    return out/np.linalg.norm(out)

def gravity(p,mu,bias):
    radius=np.linalg.norm(p)
    if radius<1: raise ValueError('Model position must be expressed relative to the Phobos center in meters')
    return -mu*np.asarray(p)/radius**3+np.asarray(bias)

def load_jet_matrix(path,mass,inertia):
    data=np.loadtxt(path,delimiter=',',skiprows=1)
    if data.ndim!=2 or data.shape[1]!=7 or not np.isfinite(data).all() or np.any(data[:,6]<=0):
        raise ValueError('Expected current jet table: x,y,z,nx,ny,nz,thrust; seven finite columns')
    norm=np.linalg.norm(data[:,3:6],axis=1)
    if np.any(norm<=0):raise ValueError('Jet direction has zero norm')
    force=-data[:,6,None]*data[:,3:6]/norm[:,None]
    A=np.c_[force/mass,np.linalg.solve(inertia,np.cross(data[:,:3],force).T).T]
    if np.linalg.matrix_rank(A)<6:raise ValueError('Current jet table does not span six requested axes')
    return A,data[:,6]

class Allocator:
    """Predict a declared minimum-sum LP law; never silently switch LP solvers."""
    def __init__(self,A,forces=None,solver='revised simplex',caps=(.5,.05)):
        self.A=np.asarray(A,float);self.forces=np.asarray(forces if forces is not None else np.zeros(len(A)),float)
        if self.A.ndim!=2 or self.A.shape[1]!=6 or not np.isfinite(self.A).all():raise ValueError('Invalid embedded jet matrix')
        if np.linalg.matrix_rank(self.A)<6:raise ValueError('Embedded jet matrix must have rank six')
        if self.forces.shape!=(len(A),) or not np.isfinite(self.forces).all() or np.any(self.forces<0):raise ValueError('Invalid embedded jet forces')
        if len(caps)!=2 or not np.isfinite(caps).all() or min(caps)<=0:raise ValueError('Invalid request caps')
        self.solver=solver;self.caps=caps;self.cache={};self.basis_cost=np.zeros((2,6))
        for sign in (-1,1):
            for axis in range(6):
                u=np.zeros(6);u[axis]=sign
                allocation=self.solve(u)
                self.basis_cost[int(sign>0),axis]=np.sum(allocation)
    def clip(self,u):
        b=np.asarray(u,float).copy()
        if b.shape!=(6,) or not np.isfinite(b).all():raise ValueError('Expected finite six-vector public request')
        for s,cap in ((slice(0,3),self.caps[0]),(slice(3,6),self.caps[1])):
            norm=np.linalg.norm(b[s])
            if norm>cap:b[s]*=cap/norm
        return b
    def solve(self,b):
        if np.linalg.norm(b)<1e-12:return np.zeros(len(self.A))
        key=tuple(np.asarray(b,float))
        if key not in self.cache:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',DeprecationWarning)
                fit=linprog(np.ones(len(self.A)),A_eq=self.A.T,b_eq=b,bounds=(0,None),method=self.solver)
            if not fit.success:raise ValueError('Declared allocation model LP failed: '+fit.message)
            if len(self.cache)>50000:self.cache.clear()
            self.cache[key]=np.maximum(fit.x,0)
        return self.cache[key]
    def peaks(self,commands,minimum_period):
        peaks=[]
        for u in commands:
            b=self.clip(u)
            # Sum of feasible nonnegative axis allocations upper-bounds the optimal
            # minimum sum, which upper-bounds its largest component. Safe fast path.
            bound=float(np.sum(abs(b)*self.basis_cost[(b>0).astype(int),np.arange(6)]))
            peaks.append(0. if bound<minimum_period*(1-1e-8) else float(self.solve(b).max()))
        return np.asarray(peaks)
    def mean(self,u,period):
        if not np.isfinite(period) or period<=0:raise ValueError('Expected positive effective allocation period')
        b=self.clip(u)
        if np.linalg.norm(b)<1e-12:return np.zeros(6)
        bound=float(np.sum(abs(b)*self.basis_cost[(b>0).astype(int),np.arange(6)]))
        if bound<period*(1-1e-8):return b/period
        allocation=self.solve(b);denom=max(period,float(allocation.max()))
        return b/denom

class Model:
    n_dim=13
    m_dim=6
    def __init__(self,definition):
        if definition.get('schema_version')!=1 or definition.get('model_type')!=MODEL_TYPE:
            raise ValueError('Unsupported model schema/input contract; do not use an old acceleration-identity model')
        self.definition=copy.deepcopy(definition);self.p=self.definition['parameters']
        self.J=np.asarray(definition['inertia_B_kg_m2'],float);self.Ji=np.linalg.inv(self.J)
        if self.J.shape!=(3,3) or not np.isfinite(self.J).all() or not np.allclose(self.J,self.J.T) or np.min(np.linalg.eigvalsh(self.J))<=0:
            raise ValueError('Expected positive definite symmetric body inertia')
        for key,shape in (('linear_gain',(3,3)),('angular_gain',(3,3)),('reporting_offset_B_m',(3,)),
                ('velocity_report_bias_P_m_s',(3,)),('rate_report_bias_B_rad_s',(3,)),
                ('uniform_external_acceleration_P_m_s2',(3,)),('angular_bias_B_rad_s2',(3,))):
            value=np.asarray(self.p[key],float)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid model parameter: '+key)
        for key in ('mean_native_period_s','linear_effect_delay_s','angular_effect_delay_s',
                    'velocity_report_delay_s','rate_report_delay_s','gyro_coefficient','phobos_mu_prior_m3_s2'):
            if not np.isfinite(self.p[key]) or self.p[key]<0:raise ValueError('Invalid model parameter: '+key)
        if self.p['mean_native_period_s']<=0 or not 0<definition['integration_max_step_s']<=.1:
            raise ValueError('Invalid effective period/integration step')
        self.allocator=Allocator(definition['actuator']['jet_matrix'],definition['actuator'].get('jet_force_N'),
            definition['actuator']['lp_solver'],definition['actuator']['request_norm_caps'])
        self.Kl=np.asarray(self.p['linear_gain']);self.Ka=np.asarray(self.p['angular_gain'])
        self.ell=np.asarray(self.p['reporting_offset_B_m']);self.vbias=np.asarray(self.p['velocity_report_bias_P_m_s'])
        self.wbias=np.asarray(self.p['rate_report_bias_B_rad_s']);self.h=float(self.p['mean_native_period_s'])
    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text(encoding='utf-8-sig')))
    def mean_acceleration(self,u):
        a=self.allocator.mean(u,self.h)
        return np.r_[self.Kl@a[:3],self.Ka@a[3:]]
    def request_for_mean_acceleration(self,desired):
        a=np.asarray(desired,float)
        if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Expected finite desired six-axis acceleration')
        u=self.allocator.clip(self.h*np.r_[np.linalg.solve(self.Kl,a[:3]),np.linalg.solve(self.Ka,a[3:])])
        achieved=self.mean_acceleration(u)
        return dict(request=u,predicted_mean_acceleration=achieved,residual=achieved-a)
    def _advance(self,x,dt,base):
        w=x[10:13];gamma=self.p['gyro_coefficient'];bias=np.asarray(self.p['angular_bias_B_rad_s2'])
        alpha=self.Ka@base[3:]+bias
        wd=alpha-gamma*(self.Ji@np.cross(w,self.J@w));wm=w+.5*dt*wd
        wd=alpha-gamma*(self.Ji@np.cross(wm,self.J@wm))
        qm=increment(x[6:10],wm,dt/2)
        a=matrix(qm).T@(self.Kl@base[:3])+gravity(x[:3]+x[3:6]*dt/2,
            self.p['phobos_mu_prior_m3_s2'],self.p['uniform_external_acceleration_P_m_s2'])
        out=x.copy();out[:3]+=x[3:6]*dt+.5*a*dt*dt;out[3:6]+=a*dt
        out[6:10]=increment(x[6:10],wm,dt);out[10:13]+=wd*dt
        return out
    def step(self,x,u,dt):
        """Constant-input step: u assumed held before x0. For transitions use rollout."""
        x=np.asarray(x,float).copy();remaining=float(dt)
        if x.shape!=(13,) or not np.isfinite(x).all() or not np.isfinite(remaining) or remaining<0:raise ValueError('Invalid state or prediction interval')
        matrix(x[6:10]);x[6:10]/=np.linalg.norm(x[6:10])
        base=self.allocator.mean(u,self.h)
        while remaining>1e-10:
            h=min(remaining,self.definition['integration_max_step_s'])
            x=self._advance(x,h,base);remaining-=h
        return x
    def rollout(self,x0,t0,times,command_times,commands):
        """Latent free run from one initialization; timestamped public requests only."""
        times=np.asarray(times,float);ct=np.asarray(command_times,float);u=np.asarray(commands,float)
        if times.ndim!=1 or not np.isfinite(times).all() or np.any(np.diff(times)<0) or len(times)==0 or times[0]<t0-1e-9:raise ValueError('Invalid rollout times')
        if u.shape!=(len(ct),6) or not np.isfinite(u).all() or not np.isfinite(ct).all() or np.any(np.diff(ct)<0):raise ValueError('Invalid command history')
        if len(ct)==0:ct=np.array([t0-1.]);u=np.zeros((1,6))
        delay=max(self.p['linear_effect_delay_s'],self.p['angular_effect_delay_s'])
        first=max(0,int(np.searchsorted(ct,t0-delay,side='right')-1));last=int(np.searchsorted(ct,times[-1],side='right'))
        last=max(first+1,last);ct=ct[first:last];u=u[first:last]
        base=np.array([self.allocator.mean(b,self.h) for b in u])
        dl=self.p['linear_effect_delay_s'];da=self.p['angular_effect_delay_s']
        edges=np.unique(np.r_[ct+dl,ct+da]);x=np.asarray(x0,float).copy();clock=float(t0);out=[]
        if x.shape!=(13,) or not np.isfinite(x).all():raise ValueError('Invalid initial state')
        matrix(x[6:10]);x[6:10]/=np.linalg.norm(x[6:10])
        for end in times:
            while end-clock>1e-10:
                edge_index=np.searchsorted(edges,clock+1e-9,side='right')
                next_edge=edges[edge_index] if edge_index<len(edges) else float('inf')
                dt=min(end-clock,self.definition['integration_max_step_s'],next_edge-clock)
                if dt<=1e-10:raise ValueError('Prediction clock failed to advance')
                tm=clock+dt/2;il=np.searchsorted(ct,tm-dl,side='right')-1;ia=np.searchsorted(ct,tm-da,side='right')-1
                b=np.r_[base[il,:3] if il>=0 else np.zeros(3),base[ia,3:] if ia>=0 else np.zeros(3)]
                x=self._advance(x,dt,b);clock+=dt
            out.append(x.copy())
        return np.asarray(out)
    def causal_initial(self,t,reported,clock):
        """Only own observations at/before clock enter initialization."""
        t=np.asarray(t);reported=np.asarray(reported);i=int(np.searchsorted(t,clock+1e-9,side='right')-1)
        if i<0:raise ValueError('No causal initial state')
        now=float(t[i]);observed=reported[i].copy();use=(t>=now-.3)&(t<=now)
        slopes=[]
        for sl in (slice(3,6),slice(10,13)):
            slope=np.linalg.lstsq(np.c_[t[use]-now,np.ones(use.sum())],reported[use,sl],rcond=None)[0][0] if use.sum()>=3 else np.zeros(3)
            slopes.append(slope)
        latent=observed.copy();latent[:3]-=matrix(latent[6:10]).T@self.ell
        latent[3:6]-=self.vbias;latent[3:6]+=self.p['velocity_report_delay_s']*slopes[0]
        latent[10:13]-=self.wbias;latent[10:13]+=self.p['rate_report_delay_s']*slopes[1]
        return now,latent,slopes
    def predict_reported(self,t,reported,t0,times,command_times,commands):
        start,x0,slopes=self.causal_initial(t,reported,t0)
        times=np.asarray(times,float)
        grid=np.unique(np.r_[start,times,np.arange(start,times[-1],self.definition['integration_max_step_s'])])
        latent=self.rollout(x0,start,grid,command_times,commands)
        out=np.empty((len(times),13))
        for i,when in enumerate(times):
            index=int(np.searchsorted(grid,when));x=latent[index]
            out[i,:3]=x[:3]+matrix(x[6:10]).T@self.ell;out[i,6:10]=x[6:10]
            for sl,delay,bias,slope in ((slice(3,6),self.p['velocity_report_delay_s'],self.vbias,slopes[0]),
                    (slice(10,13),self.p['rate_report_delay_s'],self.wbias,slopes[1])):
                query=when-delay
                out[i,sl]=(x0[sl]+slope*(query-start) if query<start else
                    np.array([np.interp(query,grid,latent[:,j]) for j in range(sl.start,sl.stop)]))+bias
        return out
