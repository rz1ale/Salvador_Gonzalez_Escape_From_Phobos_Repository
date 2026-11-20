"""Offline free-final-time nonlinear shooting; never starts SpaceCRAFT.

CasADi/IPOPT is an OFFLINE dependency only. The embedded allocation LP is the
canonical SciPy law; its active-basis derivative is local/piecewise smooth.
Published caches require a separate 0.01 s canonical replay. Local optimality
or a feasible incumbent is reported honestly; no claim of global optimality.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
# Optional isolated Windows build dependency; not imported by flight code.
LOCAL=Path(__file__).with_name('_offline_packages')
if LOCAL.is_dir(): sys.path.insert(0,str(LOCAL))
try: import casadi as ca
except ImportError as exc:
    raise SystemExit('Offline preparation needs CasADi 3.7.2. See README.txt for a solution-local install.') from exc
from system_identification.model import Model, gravity, matrix
from .check import preflight
from .core import Plant, score
from .reference import error, retract, sha256


class Allocation(ca.Callback):
    """v -> [fuel kg/s, max lambda(v)]. Raw public request is h*v.

    With max(lambda)<=1 the canonical mean map yields exactly v, up to LP
    tolerance. Final native-law replay checks this representation explicitly.
    """
    def __init__(self, allocator):
        self.allocator=allocator; self.children=[]; self.last=None
        ca.Callback.__init__(self);self.construct('allocation',{'enable_fd':False})
    def get_n_in(self):return 1
    def get_n_out(self):return 1
    def get_sparsity_in(self,i):return ca.Sparsity.dense(6,1)
    def get_sparsity_out(self,i):return ca.Sparsity.dense(2,1)
    def values(self,v):
        lam=self.allocator.solve(np.asarray(v,float))
        return np.array([self.allocator.forces@lam/784.,lam.max()])
    def eval(self,args):return [self.values(np.array(args[0]).ravel())]
    def has_jacobian(self):return True
    def get_jacobian(self,name,inames,onames,opts):
        owner=self
        class Jacobian(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self);self.construct(name,opts)
            def get_n_in(self):return 2
            def get_n_out(self):return 1
            def get_sparsity_in(self,i):return ca.Sparsity.dense(6 if i==0 else 2,1)
            def get_sparsity_out(self,i):return ca.Sparsity.dense(2,6)
            def eval(self,args):
                v=np.array(args[0]).ravel();lam=owner.allocator.solve(v)
                use=np.flatnonzero(lam>1e-8)
                if len(use)==6 and np.linalg.cond(owner.allocator.A[use].T)<1e7:
                    inv=np.linalg.inv(owner.allocator.A[use].T)
                    J=np.vstack([owner.allocator.forces[use]@inv/784.,inv[np.argmax(lam[use])]])
                else:
                    eps=2e-5;J=np.empty((2,6))
                    for j in range(6):
                        d=np.eye(6)[j]*eps
                        J[:,j]=(owner.values(v+d)-owner.values(v-d))/(2*eps)
                return [J]
        child=Jacobian();self.children.append(child);return child


def symbolic_step(definition):
    """AD transcription of Model._advance, cross-checked before optimization."""
    p=definition['parameters'];Kl=ca.DM(p['linear_gain']);Ka=ca.DM(p['angular_gain'])
    J=ca.DM(definition['inertia_B_kg_m2']);Ji=ca.inv(J)
    def mul(a,b):return ca.vertcat(a[3]*b[:3]+b[3]*a[:3]+ca.cross(a[:3],b[:3]),a[3]*b[3]-ca.dot(a[:3],b[:3]))
    def inc(q,w,dt):
        v=-w*dt;r2=ca.dot(v,v);r=ca.sqrt(r2+1e-30)
        scale=ca.if_else(r2<1e-10,.5-r2/48+r2*r2/3840,ca.sin(r/2)/r)
        dq=ca.vertcat(v*scale,ca.cos(r/2));out=mul(dq,q)
        return out/ca.sqrt(ca.dot(out,out))
    def mat(q):
        q=q/ca.sqrt(ca.dot(q,q));x,y,z,w=ca.vertsplit(q)
        return ca.vertcat(ca.horzcat(1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
            ca.horzcat(2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
            ca.horzcat(2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))
    x=ca.SX.sym('x',13);b=ca.SX.sym('base',6);dt=ca.SX.sym('dt')
    w=x[10:13];alpha=Ka@b[3:]+ca.DM(p['angular_bias_B_rad_s2'])
    wd=alpha-p['gyro_coefficient']*Ji@ca.cross(w,J@w);wm=w+.5*dt*wd
    wd=alpha-p['gyro_coefficient']*Ji@ca.cross(wm,J@wm)
    qm=inc(x[6:10],wm,dt/2);pm=x[:3]+x[3:6]*dt/2
    g=-p['phobos_mu_prior_m3_s2']*pm/ca.norm_2(pm)**3+ca.DM(p['uniform_external_acceleration_P_m_s2'])
    a=mat(qm).T@Kl@b[:3]+g
    out=ca.vertcat(x[:3]+x[3:6]*dt+.5*a*dt*dt,x[3:6]+a*dt,inc(x[6:10],wm,dt),w+wd*dt)
    f=ca.Function('midpoint',[x,b,dt],[out])
    q=ca.SX.sym('q',4);R=ca.Function('rotation',[q],[mat(q)])
    return f,R


def transition_function(model,substeps):
    step,R=symbolic_step(model.definition)
    x=ca.MX.sym('x',13);v=ca.MX.sym('v',6);old=ca.MX.sym('old',6);dt=ca.MX.sym('dt')
    da=model.p['angular_effect_delay_s'];dl=model.p['linear_effect_delay_s']
    y=x
    edges=sorted(set([0.,da,dl]));last=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        b=ca.vertcat(v[:3] if (lo+hi)/2>=dl else old[:3],v[3:] if (lo+hi)/2>=da else old[3:])
        y=step(y,b,hi-lo);last=hi
    # Above covers [0,max(delay)] even when delays differ.
    if len(edges)==1:last=0.
    for _ in range(substeps):y=step(y,v,(dt-last)/substeps)
    return ca.Function('block_transition',[x,old,v,dt],[y]),step,R


def seed(model,x0,target,N,T,turn):
    plant=Plant(model,.15);xs=[x0.copy()];vs=[];dt=T/N
    up=(x0[:3]+matrix(x0[6:10]).T@model.ell);up/=np.linalg.norm(up)
    old=np.zeros(6)
    goalbody=matrix(x0[6:10])@(target-x0[:3]);goalbody/=np.linalg.norm(goalbody)
    turnaxis=np.cross(np.array([1.,0,0]),goalbody);turnangle=np.arccos(np.clip(goalbody[0],-1,1))
    turnaxis/=max(np.linalg.norm(turnaxis),1e-10)
    for k in range(N):
        x=xs[-1];B=matrix(x[6:10]);direction=target-(x[:3]+B.T@model.ell);direction/=np.linalg.norm(direction)
        g=gravity(x[:3],model.p['phobos_mu_prior_m3_s2'],model.p['uniform_external_acceleration_P_m_s2'])
        if turn:
            # This is only a numerical seed; the NLP decides all six channels.
            alpha=turnaxis*turnangle/(dt*dt)*(1 if k==0 else -1 if k==1 else 0)
            aP=(.02*up if k<2 else .14*direction if k<N*.4 else np.zeros(3))
        else:
            alpha=-.3*x[10:13];aP=.072*direction if k<N*.7 else np.zeros(3)
        v=np.r_[np.linalg.solve(model.Kl,B@(aP-g)),np.linalg.solve(model.Ka,alpha-np.array(model.p['angular_bias_B_rad_s2']))]
        peak=model.allocator.solve(v).max();v*=min(1.,.94/max(peak,1e-12))
        vs.append(v);xs.append(plant.transition(x,old,model.h*v,dt));old=model.h*v
    return np.asarray(xs),np.asarray(vs)


def solve_mission(model,x0,target,c,N=28,substeps=24,max_iterations=400,seed_mode='turn',initial_T=190.,warm=None,transcription='single'):
    phi,step,R=transition_function(model,substeps);alloc=Allocation(model.allocator)
    # The AD adapter must agree with the canonical midpoint map, including quaternion sign.
    probe=x0.copy();probe[10:13]=[.03,-.02,.01];b=np.array([.03,.01,-.02,.01,-.015,.004])
    mismatch=float(np.max(abs(np.array(step(probe,b,.01)).ravel()-model._advance(probe,.01,b))))
    if mismatch>1e-10:raise ValueError('AD/canonical dynamics mismatch '+str(mismatch))
    opt=ca.Opti();Tvar=opt.variable();T=200*Tvar
    L=opt.variable(len(model.allocator.A),N)
    xs=np.r_[np.full(3,1000.),np.full(3,10.),np.ones(4),np.full(3,.2)]
    V=ca.DM(model.allocator.A).T@L
    scale=ca.DM(xs);dt=T/N
    opt.subject_to(opt.bounded(60.,T,450.))
    if transcription=='multiple':
        Z=opt.variable(13,N+1)
        X=ca.repmat(ca.DM(np.r_[x0[:3],np.zeros(10)]),1,N+1)+ca.repmat(ca.DM(xs),1,N+1)*Z
        opt.subject_to((X[:,0]-x0)/scale==0)
    else:
        nodes=[ca.DM(x0)]
        for k in range(N):nodes.append(phi(nodes[-1],V[:,k-1] if k else ca.DM.zeros(6),V[:,k],dt))
        X=ca.horzcat(*nodes)
    dynamics=[];fuel=0.;launch=x0[:3]+matrix(x0[6:10]).T@model.ell;up=launch/np.linalg.norm(launch)
    for k in range(N):
        old=V[:,k-1] if k else ca.DM.zeros(6)
        if transcription=='multiple':
            constraint=(phi(X[:,k],old,V[:,k],dt)-X[:,k+1])/scale
            opt.subject_to(constraint==0);dynamics.append(constraint)
        allocout=alloc(V[:,k]);opt.subject_to(allocout[1]<=.995)
        # Smooth lifting of the min-sum fuel objective. If the native peak bound
        # holds, its minimum-sum LP solution is feasible inside 0<=L<=1. For a
        # fixed V the fuel minimization can therefore attain the native LP cost.
        # This removes the absolute-value cusp from IPOPT's objective; the native
        # peak constraint and mandatory canonical replay still enforce the law.
        fuel+=ca.dot(ca.DM(model.allocator.forces),L[:,k])/784.*dt
        opt.subject_to(ca.dot(X[10:13,k+1],X[10:13,k+1])<=.45**2)
        pr=X[:3,k+1]+R(X[6:10,k+1]).T@ca.DM(model.ell)
        opt.subject_to(ca.dot(pr-launch,up)>=-.05)
        opt.subject_to(opt.bounded(0.,L[:,k],1.))
    finalp=X[:3,-1]+R(X[6:10,-1]).T@ca.DM(model.ell)
    opt.subject_to(ca.sumsqr((finalp-target)/1000.)<=((c['KeepOutRadius_m']-1.5)/1000.)**2)
    J=-100*c['NominalTime_s']/T+100*fuel/c['NominalFuel_kg']
    opt.minimize(J)
    xseed,vseed=seed(model,x0,target,N,initial_T,seed_mode=='turn')
    if warm is not None:
        initial_T=warm['duration'];told=np.linspace(0,initial_T,len(warm['states']))
        tnew=np.linspace(0,initial_T,N+1)
        xseed=np.array([[np.interp(t,told,warm['states'][:,j]) for j in range(13)] for t in tnew])
        xseed[:,6:10]/=np.linalg.norm(xseed[:,6:10],axis=1)[:,None]
        vseed=np.array([warm['base'][min(len(warm['base'])-1,int(k*len(warm['base'])/N))] for k in range(N)])
    opt.set_initial(Tvar,initial_T/200)
    if transcription=='multiple':opt.set_initial(Z,(xseed.T-np.r_[x0[:3],np.zeros(10)][:,None])/xs[:,None])
    opt.set_initial(L,np.array([model.allocator.solve(v) for v in vseed]).T)
    opt.solver('ipopt',{'expand':False},{'print_level':3,'max_iter':max_iterations,
        'tol':2e-5,'acceptable_tol':1e-3,'acceptable_iter':8,'hessian_approximation':'limited-memory',
        'sb':'yes','linear_solver':'mumps','max_cpu_time':600.,'mu_strategy':'adaptive',
        # Default 0.01 pushes every inactive jet away from zero at startup,
        # destroying a long low-thrust coast seed. Preserve its physical scale.
        'bound_push':1e-8,'bound_frac':1e-8,'slack_bound_push':1e-8,'slack_bound_frac':1e-8,
        'limited_memory_max_history':20})
    best={};start=time.monotonic();solved=True
    def retain(iteration):
        if iteration%5:return
        try:
            g=np.asarray(opt.debug.value(opt.g)).ravel();lo=np.asarray(opt.debug.value(opt.lbg)).ravel();hi=np.asarray(opt.debug.value(opt.ubg)).ravel()
            v=max(np.max(lo-g),np.max(g-hi),0.);cost=float(opt.debug.value(J))
            if v<1e-4 and (not best or cost<best['objective']):
                best.update(objective=cost,T=float(opt.debug.value(T)),X=np.array(opt.debug.value(X)),V=np.array(opt.debug.value(V)),
                    lam=np.ravel(opt.debug.value(opt.lam_g)),violation=float(v),iteration=int(iteration))
        except RuntimeError:pass
    opt.callback(retain)
    try:solution=opt.solve()
    except RuntimeError:
        solved=False;solution=opt.debug
    value=lambda z:np.asarray(solution.value(z),float)
    duration=float(value(T));states=value(X).T;v=value(V).T
    g=value(opt.g);lo=value(opt.lbg);hi=value(opt.ubg)
    violation=float(max(np.max(lo-g),np.max(g-hi),0.))
    lam=value(opt.lam_g).ravel()
    if best and (violation>=1e-4 or best['objective']<float(value(J))):
        duration=best['T'];states=best['X'].T;v=best['V'].T;lam=best['lam'];violation=best['violation'];solved=False
    # Constraint insertion: T bound1, initial13, then blocks of dynamics13,
    # allocation1, rate1, floor1, lifted nonnegative jet coefficients.
    if transcription=='multiple':
        offset=14;duals=[]
        for k in range(N):duals.append(lam[offset:offset+13]/xs);offset+=16+len(model.allocator.A)
    else:duals=np.zeros((N,13))
    stats=opt.stats()
    return dict(duration=duration,states=states,base=v,duals=np.asarray(duals),
        converged=solved,status=stats.get('return_status'),constraint_violation=violation,
        wall_s=time.monotonic()-start,substeps=substeps,stages=N,ad_step_max_abs_error=mismatch,
        parameterization='nonnegative jet lift with canonical native-peak constraint',transcription=transcription,
        dual_infeasibility=float(stats.get('iterations',{}).get('inf_du',[float('nan')])[-1]))


def export(model,model_path,x0,target,c,evidence,result,folder):
    N=result['stages'];T=result['duration'];ct=np.arange(N)*T/N;u=model.h*result['base']
    times=np.unique(np.r_[np.arange(0,T,.2),T]);start=time.monotonic()
    # Mandatory independent canonical replay: original backend, 0.01 s, true LP.
    xs=model.rollout(x0,0.,times,np.r_[-1.,ct],np.vstack([np.zeros(6),u]))
    plant=Plant(model,.08);reported=plant.reported_positions(xs)
    distance=np.linalg.norm(reported-target,axis=1);floor=(reported-reported[0])@(reported[0]/np.linalg.norm(reported[0]))
    map_errors=[np.linalg.norm(plant.base(raw)-v) for raw,v in zip(u,result['base'])]
    final_mismatch=float(np.linalg.norm(error(xs[-1],result['states'][-1])[:6]))
    fuel_rates=np.array([plant.fuel_rate(raw) for raw in u]);fuel=float(fuel_rates.sum()*T/N)
    validation=dict(canonical_replay=True,integration_max_step_s=model.definition['integration_max_step_s'],
        terminal_distance_m=float(distance[-1]),minimum_launch_plane_margin_m=float(floor.min()+.1),
        max_body_rate_rad_s=float(np.max(np.linalg.norm(xs[:,10:13],axis=1))),
        max_base_allocation_error=float(max(map_errors)),nlp_replay_terminal_pv_error=final_mismatch,
        fuel_kg=fuel,score_estimate=score(T,fuel,0.,c),native_qualified=False)
    valid=(distance[-1]<c['KeepOutRadius_m']-.25 and floor.min()>=-.1 and max(map_errors)<.002
           and validation['max_body_rate_rad_s']<=.5 and result['constraint_violation']<.003)
    if not valid:raise ValueError('Reference NOT published: canonical validation failed: '+json.dumps(validation))
    print('Canonical replay passed; preparing coupled feedback gains',flush=True)
    # State/input-scaled finite-horizon LQR about the coupled nonlinear reference.
    gt=np.unique(np.r_[np.arange(0,T,1.),T]);gxs=np.array([xs[np.argmin(abs(times-t))] for t in gt])
    Q=np.diag([.018]*3+[.3]*3+[2.]*3+[.45]*3);R=np.diag([.35]*3+[.3]*3)
    P=np.diag([.04]*3+[.5]*3+[3.]*3+[.6]*3);gains=[];curv=[];grads=[]
    for i in range(len(gt)-1,-1,-1):
        t=gt[i];j=min(N-1,int(t/(T/N)));raw=u[j]
        x=gxs[i];_,A,B,D=plant.linearize(x,raw,raw,.1);B=B+D
        for _ in range(10):
            H=R+B.T@P@B;K=-np.linalg.solve(H,B.T@P@A)
            P=Q+A.T@P@A+A.T@P@B@K;P=(P+P.T)/2
        gains.append(K/plant.scales[:,None]);curv.append(P)
        k=min(N-1,int(t/(T/N)));dual=result['duals'][k]
        L=np.zeros((13,12));L[:6,:6]=np.eye(6);L[10:13,9:12]=np.eye(3)
        eps=1e-5
        for a in range(3):
            d=np.eye(12)[a+6]*eps
            L[6:10,6+a]=(retract(x,d)[6:10]-retract(x,-d)[6:10])/(2*eps)
        grads.append(L.T@dual)
    gains=np.asarray(gains[::-1]);curv=np.asarray(curv[::-1]);grads=np.asarray(grads[::-1])
    value_method='NLP equality multipliers'
    if not result['converged'] or result.get('transcription')=='single':
        # A feasible incumbent's multipliers need not be stationary. Use the
        # explicit cached tail score's tangent derivative, label that approximation.
        grads=[];W=np.diag([1/25.]*3+[1.]*9)
        from .reference import interpolate_state
        for t in gt:
            lo=max(0.,t-.2);hi=min(T,t+.2);x=interpolate_state(times,xs,t)
            tangent=(error(interpolate_state(times,xs,hi),x)-error(interpolate_state(times,xs,lo),x))/max(hi-lo,1e-9)
            j=min(N-1,int(t/(T/N)));dvalue=-100*c['NominalTime_s']/T**2-100*fuel_rates[j]/c['NominalFuel_kg']
            grads.append(dvalue*(W@tangent)/max(tangent@W@tangent,1e-8))
        grads=np.asarray(grads);value_method='cached remaining-score tangent; NLP multipliers not treated as a stationary value function'
    if not np.isfinite(grads).all():raise ValueError('Nonfinite value gradients')
    cum=np.r_[0.,np.cumsum(fuel_rates*T/N)]
    remaining=fuel-np.interp(times,np.r_[ct,T],cum)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'nominal_trajectory.npz'
    np.savez_compressed(dest,times=times,states=xs,command_times=ct,commands=u,gain_times=gt,
        gains=gains,value_gradient=grads,curvature=curv,fuel_remaining=remaining)
    manifest=dict(schema_version=1,model_sha256=sha256(model_path),trajectory_sha256=sha256(dest),
        static_contract=evidence,score_constants=c,target_P_m=target.tolist(),initial_state=x0.tolist(),
        request_norm_caps=model.allocator.caps,request_scales=plant.scales.tolist(),
        objective='minimize -100*NominalTime_s/T + 100*fuel_kg/NominalFuel_kg; damage=0 optimistic',
        terminal_constraints='reported-origin target radius only; free terminal velocity and attitude',
        geometry='local launch plane only; no offline obstacle map or terrain certificate',
        feedback='coupled 12-state finite-horizon LQR; frozen 0.1 s input-delay approximation B+D',
        online_value=value_method+'; LQR curvature is a stabilizing regularizer, not the exact score Hessian',
        solver={k:v for k,v in result.items() if k not in ('states','base','duals')},validation=validation,
        build_versions=dict(casadi=ca.__version__,numpy=np.__version__),export_wall_s=time.monotonic()-start,
        generator_sha256=sha256(__file__))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='nominal_model.json');p.add_argument('--stages',type=int,default=28)
    p.add_argument('--substeps',type=int,default=24);p.add_argument('--max-iterations',type=int,default=400)
    p.add_argument('--seed',choices=['turn','fixed'],default='turn');p.add_argument('--initial-time-s',type=float,default=190.)
    p.add_argument('--output',default='policy/trajectory_handoff/assets')
    p.add_argument('--resume-candidate',help='Offline optimizer candidate NPZ, not a simulator state')
    p.add_argument('--warm-start',help='Refine a previous offline_candidate.npz')
    p.add_argument('--transcription',choices=['single','multiple'],default='single')
    args=p.parse_args();model_path=ROOT/args.model
    if not 8<=args.stages<=80 or not 4<=args.substeps<=200 or not 1<=args.max_iterations<=10000:
        p.error('Supported bounds: stages8..80, substeps4..200, iterations1..10000')
    def within(name):
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('All model/cache/output paths must be inside this solution clone')
        return path
    model_path=within(args.model);output=within(args.output)
    definition,x0,target,c,evidence=preflight(ROOT,model_path);model=Model(definition)
    if args.resume_candidate:
        z=np.load(within(args.resume_candidate),allow_pickle=False)
        r=json.loads(str(z['metadata']));r.update(states=z['states'],base=z['base'],duals=z['duals'])
        if r.get('candidate_model_sha256')!=sha256(model_path):raise ValueError('Offline candidate belongs to another model; use it only as a --warm-start seed')
    else:
        warm=None
        if args.warm_start:
            z=np.load(within(args.warm_start),allow_pickle=False);warm=json.loads(str(z['metadata']))
            warm.update(states=z['states'],base=z['base'],duals=z['duals'])
        elif (output/'manifest.json').is_file():
            # The included feasible cache is a seed, never an optimizer substitute.
            from .reference import Reference
            try:
                cached=Reference(output,model_path)
                warm=dict(duration=cached.duration,states=cached.states,base=cached.commands/model.h)
            except ValueError:pass
        r=solve_mission(model,x0,target,c,args.stages,args.substeps,args.max_iterations,args.seed,args.initial_time_s,warm,args.transcription)
        # Single replaceable optimizer checkpoint, outside flight assets. No simulator snapshots.
        r['candidate_model_sha256']=sha256(model_path)
        meta={k:v for k,v in r.items() if k not in ('states','base','duals')}
        np.savez_compressed(Path(__file__).with_name('offline_candidate.npz'),metadata=json.dumps(meta),states=r['states'],base=r['base'],duals=r['duals'])
    manifest=export(model,model_path,x0,target,c,evidence,r,output)
    print(json.dumps(dict(published=True,solver=manifest['solver'],validation=manifest['validation']),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
