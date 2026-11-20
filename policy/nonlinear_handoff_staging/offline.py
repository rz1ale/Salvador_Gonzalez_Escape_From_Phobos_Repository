"""Canonical-command coordinates + exact allocator feasibility + multiple shooting.

At feasible points peak(lambda)<=h, so delivered base equals public_request/h.
The exact native minimum-sum allocation enforces that condition; all exports are
replayed through its original clipping/normalization, without this coordinate lift.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import casadi as ca
from .math_core import matrix,increment,direction_rotation,reported,mission_reference,terminal_limits


class Allocation(ca.Callback):
    def __init__(self,model,scales,fd=2e-4):
        self.model=model;self.scales=np.asarray(scales);self.fd=fd;self.children=[]
        ca.Callback.__init__(self);self.construct("canonical_allocation",{"enable_fd":False})
    def get_n_in(self):return 1
    def get_n_out(self):return 1
    def get_sparsity_in(self,i):return ca.Sparsity.dense(6,1)
    def get_sparsity_out(self,i):return ca.Sparsity.dense(6+len(self.model.allocator.A),1)
    def values(self,z):
        b=self.model.allocator.clip(self.scales*np.asarray(z));lam=self.model.allocator.solve(b)
        peak=float(lam.max());h=self.model.h
        return np.r_[b/max(h,peak),lam/h]
    def derivative(self,z):
        raw=self.scales*np.asarray(z);b=self.model.allocator.clip(raw);C=np.eye(6)
        for sl,cap in ((slice(0,3),self.model.allocator.caps[0]),(slice(3,6),self.model.allocator.caps[1])):
            n=np.linalg.norm(raw[sl])
            if n>cap:C[sl,sl]=cap/n*(np.eye(3)-np.outer(raw[sl],raw[sl])/n**2)
        lam=self.model.allocator.solve(b);peak=float(lam.max());h=self.model.h
        active=np.flatnonzero(lam>max(1e-12,peak*1e-9))
        if len(active)==6 and np.linalg.cond(self.model.allocator.A[active].T)<1e8:
            inverse=np.linalg.inv(self.model.allocator.A[active].T)
            dpeak=inverse[int(np.argmax(lam[active]))]
            if peak>h*(1+1e-7):
                D=(np.eye(6)/peak-np.outer(b,dpeak)/peak**2)@C@np.diag(self.scales)
            else:D=C@np.diag(self.scales)/h
            Jlam=np.zeros((len(lam),6));Jlam[active]=inverse@C@np.diag(self.scales)/h
            return np.vstack([D,Jlam])
        D=np.column_stack([(self.values(z+np.eye(6)[j]*self.fd)-self.values(z-np.eye(6)[j]*self.fd))/(2*self.fd) for j in range(6)])
        if peak<=h*(1+1e-7):D[:6]=C@np.diag(self.scales)/h
        return D
    def eval(self,args):return [self.values(np.asarray(args[0]).ravel())]
    def has_jacobian(self):return True
    def get_jacobian(self,name,inames,onames,opts):
        owner=self
        class Derivative(ca.Callback):
            def __init__(self):ca.Callback.__init__(self);self.construct(name,opts)
            def get_n_in(self):return 2
            def get_n_out(self):return 1
            def get_sparsity_in(self,i):return ca.Sparsity.dense(6 if i==0 else 6+len(owner.model.allocator.A),1)
            def get_sparsity_out(self,i):return ca.Sparsity.dense(6+len(owner.model.allocator.A),6)
            def eval(self,args):return [owner.derivative(np.asarray(args[0]).ravel())]
        child=Derivative();self.children.append(child);return child


def symbolic(definition,dt,maximum_step):
    p=definition["parameters"];Kl=ca.DM(p["linear_gain"]);Ka=ca.DM(p["angular_gain"])
    J=ca.DM(definition["inertia_B_kg_m2"]);Ji=ca.inv(J)
    def mul(a,b):return ca.vertcat(a[3]*b[:3]+b[3]*a[:3]+ca.cross(a[:3],b[:3]),a[3]*b[3]-ca.dot(a[:3],b[:3]))
    def inc(q,w,h):
        v=-w*h;r2=ca.dot(v,v);r=ca.sqrt(r2+1e-30)
        scale=ca.if_else(r2<1e-10,.5-r2/48+r2*r2/3840,ca.sin(r/2)/r)
        out=mul(ca.vertcat(v*scale,ca.cos(r/2)),q)
        return out/ca.sqrt(ca.dot(out,out)+1e-30)
    def rot(q):
        q=q/ca.sqrt(ca.dot(q,q)+1e-30);x,y,z,w=ca.vertsplit(q)
        return ca.vertcat(ca.horzcat(1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
            ca.horzcat(2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
            ca.horzcat(2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))
    def step(x,b,h):
        w=x[10:13];alpha=Ka@b[3:]+ca.DM(p["angular_bias_B_rad_s2"])
        wd=alpha-p["gyro_coefficient"]*Ji@ca.cross(w,J@w);wm=w+h*.5*wd
        wd=alpha-p["gyro_coefficient"]*Ji@ca.cross(wm,J@wm)
        qm=inc(x[6:10],wm,h/2);pm=x[:3]+x[3:6]*h/2
        g=-p["phobos_mu_prior_m3_s2"]*pm/(ca.dot(pm,pm)+1e-30)**1.5+ca.DM(p["uniform_external_acceleration_P_m_s2"])
        a=rot(qm).T@Kl@b[:3]+g
        return ca.vertcat(x[:3]+x[3:6]*h+.5*a*h*h,x[3:6]+a*h,inc(x[6:10],wm,h),w+wd*h)
    x=ca.MX.sym("x",13);old=ca.MX.sym("old",6);new=ca.MX.sym("new",6)
    y=x;dl=p["linear_effect_delay_s"];da=p["angular_effect_delay_s"]
    edges=sorted(set([0.,min(dt,dl),min(dt,da),dt]))
    for lo,hi in zip(edges[:-1],edges[1:]):
        b=ca.vertcat(new[:3] if (lo+hi)/2>=dl else old[:3],new[3:] if (lo+hi)/2>=da else old[3:])
        count=max(1,int(np.ceil((hi-lo)/maximum_step)))
        for _ in range(count):y=step(y,b,(hi-lo)/count)
    phi=ca.Function("transition",[x,old,new],[y])
    jac=ca.Function("transition_jac",[x,old,new],[ca.jacobian(y,x),ca.jacobian(y,old),ca.jacobian(y,new)])
    q=ca.MX.sym("q",4);rotation=ca.Function("rotation",[q],[rot(q)])
    b=ca.MX.sym("b",6);h=ca.MX.sym("h")
    midpoint=ca.Function("midpoint",[x,b,h],[step(x,b,h)])
    qref=ca.MX.sym("qref",4)
    delta=mul(q,ca.vertcat(-qref[:3],qref[3]));delta=ca.if_else(delta[3]<0,-delta,delta)
    r2=ca.dot(delta[:3],delta[:3]);r=ca.sqrt(r2+1e-30)
    scale=ca.if_else(r2<1e-12,2+r2/3,2*ca.atan2(r,delta[3])/r)
    log=ca.Function("attitude_error",[q,qref],[-delta[:3]*scale])
    return phi,jac,rotation,midpoint,log


def input_scales(model):
    return np.array([model.h/max(float(model.allocator.solve(sign*np.eye(6)[j]).max()) for sign in (-1,1)) for j in range(6)])


def weights(d,name):return np.concatenate([np.asarray(d[name][k],float) for k in ("position","velocity","attitude","rate")])


def solve(model,initial,target,s,progress=None,*,audit_only=False,warm=None):
    d=s["design"];T=d["duration_s"];N=d["steps"];dt=T/N
    if dt<=max(model.p["linear_effect_delay_s"],model.p["angular_effect_delay_s"]):
        raise ValueError("Optimization intervals must exceed model input delays; choose fewer steps")
    ref=mission_reference(initial,target,model.definition,d);scales=input_scales(model)
    phi,jac,rotation,midpoint,log=symbolic(model.definition,dt,s["solver"]["integration_step_s"])
    allocator=Allocation(model,scales,s["solver"]["allocation_derivative_step"])
    probe=initial.copy();probe[10:13]=[.02,-.01,.03];b=np.array([.01,-.02,.03,.01,-.02,.01])
    map_error=float(np.max(abs(np.asarray(midpoint(probe,b,.01)).ravel()-model._advance(probe,.01,b))))
    if map_error>1e-10:raise ValueError("Symbolic/canonical dynamics mismatch")
    # These are numerical initial guesses, never programmed flight phases.
    controls=[];nodes=[initial.copy()];old=np.zeros(6);turn_T=max(T-dt*2,dt)
    for k in range(N):
        x=nodes[-1];time_mid=(k+.5)*dt;r=np.clip(time_mid/turn_T,0.,1.)
        f=10*r**3-15*r**4+6*r**5;fp=(30*r**2-60*r**3+30*r**4)/turn_T
        fpp=(60*r-180*r**2+120*r**3)/turn_T**2
        qref=increment(initial[6:10],ref["axis"]*ref["angle"]*f)
        from .math_core import error
        desired=x.copy();desired[6:10]=qref;desired[10:13]=ref["axis"]*ref["angle"]*fp
        e=error(x,desired)
        alpha=ref["axis"]*ref["angle"]*fpp-1.0*e[6:9]-1.8*e[9:]
        u=np.r_[np.zeros(3),model.h*np.linalg.solve(model.Ka,alpha)]
        if k==N-1:u=np.array([model.h/model.allocator.solve(np.eye(6)[0]).max(),0,0,0,0,0.])
        u=model.h*model.allocator.mean(u,model.h)
        base=model.allocator.mean(u,model.h)
        nodes.append(np.asarray(phi(x,old,base)).ravel());old=base;controls.append(u/scales)
    uend=np.array([model.h/model.allocator.solve(np.eye(6)[0]).max(),0,0,0,0,0.])
    if warm is not None:
        old=np.zeros(6);nodes=[initial.copy()];controls=[]
        for k in range(N):
            phase=k*dt/ T*float(warm["times"][-1])
            j=min(len(warm["commands"])-1,max(0,int(np.searchsorted(warm["times"][:-1],phase,side="right")-1)))
            u=model.h*model.allocator.mean(warm["commands"][j],model.h)
            if k==N-1:u=uend.copy()
            b=model.allocator.mean(u,model.h)
            nodes.append(np.asarray(phi(nodes[-1],old,b)).ravel());old=b;controls.append(u/scales)
    maximum=model.mean_acceleration([.5,0,0,0,0,0])
    if not np.allclose(model.mean_acceleration(uend),maximum,rtol=1e-4,atol=1e-7):
        raise ValueError("Canonical terminal input does not reproduce maximum +X; actuator review required")
    opt=ca.Opti();Z=opt.variable(13,N);U=opt.variable(6,N)
    state_scale=np.r_[np.full(3,max(d["distance_m"],1.)),np.ones(3),np.ones(4),np.full(3,.3)]
    origin=np.r_[initial[:3],np.zeros(10)]
    X=ca.horzcat(ca.DM(initial),ca.repmat(ca.DM(origin),1,N)+ca.repmat(ca.DM(state_scale),1,N)*Z)
    allocations=[allocator(U[:,k]) for k in range(N)]
    # Smooth dynamics in canonical delivered-base coordinates. Allocation is
    # retained as exact algebraic feasibility, not replaced by free jet controls.
    bases=[ca.DM(scales/model.h)*U[:,k] for k in range(N)]
    Q=ca.DM(weights(d,"Q"));Qf=ca.DM(weights(d,"Q_final"));R=ca.DM(d["R"])
    def state_error(x):
        pr=x[:3]+rotation(x[6:10]).T@ca.DM(model.ell)
        return ca.vertcat(pr-ref["pd"],x[3:6]-ref["vd"],log(x[6:10],ref["qd"]),x[10:13])
    objective=0;residuals=[];smooth_terms=[];smooth_indices=[]
    def constrain(expr,bound,equality=False):
        before=opt.g.numel()
        opt.subject_to(expr==bound if equality else expr<=bound)
        smooth_terms.append(expr);smooth_indices.extend(range(before,before+expr.numel()))
    for k in range(N):
        old=bases[k-1] if k else ca.DM.zeros(6)
        constrain((phi(X[:,k],old,bases[k])-X[:,k+1])/ca.DM(state_scale),0,True)
        raw=ca.DM(scales)*U[:,k]
        constrain(ca.sumsqr(raw[:3]),model.allocator.caps[0]**2)
        constrain(ca.sumsqr(raw[3:]),model.allocator.caps[1]**2)
        # Each minimum-sum native LP on-time must fit the mean period. Separate
        # jet inequalities avoid the nonsmooth maximum of tied peak on-times.
        if k<N-1:opt.subject_to(ca.mmax(allocations[k][6:])<=1.0)
        e=state_error(X[:,k]);objective+=dt*(ca.dot(Q*e,e)+ca.dot(R*U[:,k],U[:,k]))
        residuals.extend([ca.sqrt(dt*Q)*e,ca.sqrt(dt*R)*U[:,k]])
    e=state_error(X[:,-1]);objective+=ca.dot(Qf*e,e);residuals.append(ca.sqrt(Qf)*e)
    finalp=X[:3,-1]+rotation(X[6:10,-1]).T@ca.DM(model.ell)
    direction=ca.DM(ref["xf"])-finalp;direction/=ca.sqrt(ca.dot(direction,direction)+1e-30)
    forward=rotation(X[6:10,-1]).T@ca.DM([1.,0,0])
    # Squared chord is numerically better than subtracting nearly equal cosines.
    limits=terminal_limits(s,ref)
    constrain(ca.sumsqr(forward-direction)/(2*np.sin(np.radians(limits["pointing_deg"])/2))**2,1)
    constrain(ca.sumsqr(X[10:13,-1])/limits["rate_rad_s"]**2,1)
    if d["terminal_position_tolerance_m"] is not None:
        constrain(ca.sumsqr(finalp-ref["pd"]),d["terminal_position_tolerance_m"]**2)
    opt.subject_to(U[:,-1]==uend/scales)
    opt.minimize(objective)
    opt.set_initial(Z,(np.asarray(nodes[1:]).T-origin[:,None])/state_scale[:,None])
    opt.set_initial(U,np.asarray(controls).T)
    if audit_only:
        fg=ca.Function("nlp_fg",[opt.x],[opt.f,opt.g])
        jf=ca.Function("nlp_jac",[opt.x],[ca.gradient(opt.f,opt.x),ca.jacobian(opt.g,opt.x)])
        z0=np.asarray(opt.debug.value(opt.x,opt.initial())).ravel()
        f0,g0=fg(z0);grad,J=jf(z0);grad=np.asarray(grad).ravel();J=np.asarray(J)
        lo=np.asarray(opt.debug.value(opt.lbg)).ravel();hi=np.asarray(opt.debug.value(opt.ubg)).ravel();g0=np.asarray(g0).ravel()
        bad=np.maximum(np.maximum(lo-g0,g0-hi),0)
        rng=np.random.default_rng(981);checks=[]
        for _ in range(3):
            v=rng.normal(size=len(z0));v/=np.linalg.norm(v);eps=1e-5
            fp,gp=fg(z0+eps*v);fm,gm=fg(z0-eps*v)
            fd=(float(fp)-float(fm))/(2*eps);gd=(np.asarray(gp).ravel()-np.asarray(gm).ravel())/(2*eps)
            checks.append(dict(objective_error=float(abs(fd-grad@v)),constraint_error=float(np.max(abs(gd-J@v))),
                constraint_worst_row=int(np.argmax(abs(gd-J@v)))))
        return dict(initial_objective=float(f0),initial_violation=float(bad.max()),worst_constraint=int(np.argmax(bad)),
            variables=len(z0),constraints=len(g0),checks=checks),allocator
    # Exact smooth Lagrangian curvature. Native LP allocation is piecewise
    # affine in canonical requests, hence zero second derivative inside a basis.
    # Its first derivatives and exact feasibility values remain in the NLP.
    sigma=ca.MX.sym("sigma");lambda_g=ca.MX.sym("lambda_g",opt.g.numel())
    lag=sigma*objective+ca.dot(lambda_g[smooth_indices],ca.vertcat(*smooth_terms))
    H=ca.triu(ca.hessian(lag,opt.x)[0])
    hess=ca.Function("nlp_hess_l",[opt.x,opt.p,sigma,lambda_g],[H])
    cfg=s["solver"];opt.solver("ipopt",{"expand":False,"print_time":False,"hess_lag":hess},{"print_level":0,"sb":"yes",
        "tol":cfg["optimality_tolerance"],"constr_viol_tol":cfg["constraint_tolerance"],
        "acceptable_iter":0,"max_iter":cfg["max_iterations"],"max_wall_time":cfg["max_wall_s"],
        "mu_strategy":"adaptive",
        "bound_push":1e-8,"bound_frac":1e-8})
    start=time.monotonic();best={}
    lo_values=np.asarray(opt.debug.value(opt.lbg)).ravel();hi_values=np.asarray(opt.debug.value(opt.ubg)).ravel()
    def retain(iteration,initial_guess=False):
        try:
            if initial_guess:
                v=lambda z:opt.debug.value(z,opt.initial())
            else:v=opt.debug.value
            gv=np.asarray(v(opt.g)).ravel();violation=float(max(0.,np.max(lo_values-gv),np.max(gv-hi_values)))
            cost=float(v(objective))
            if violation<=1e-3 and (not best or cost<best["objective"]):
                candidate_u=np.asarray(v(U)).T*scales
                actual=model.rollout(initial,0,np.array([T]),np.arange(N)*dt,candidate_u)[-1]
                actual_p=reported(actual,model.definition)
                _,actual_angle=direction_rotation(matrix(actual[6:10])@(ref["xf"]-actual_p))
                if (np.degrees(actual_angle)>limits["pointing_deg"] or np.linalg.norm(actual[10:13])>limits["rate_rad_s"] or
                    (limits["position_m"] is not None and np.linalg.norm(actual_p-ref["pd"])>limits["position_m"])):return
                best.update(objective=cost,states=np.asarray(v(X)).T,commands=candidate_u,
                    violation=violation,iteration=iteration,kind="initial_feasible_seed" if initial_guess else "feasible_solver_iterate")
        except RuntimeError:pass
    retain(0,True)
    def report(iteration):
        if iteration%5==0:retain(iteration)
        if progress is not None and (iteration%10==0):
            try:progress(iteration,float(opt.debug.value(objective)),time.monotonic()-start)
            except RuntimeError:pass
    opt.callback(report)
    success=True
    try:sol=opt.solve()
    except RuntimeError:success=False;sol=opt.debug
    stats=opt.stats();x=np.asarray(sol.value(X)).T;u=np.asarray(sol.value(U)).T*scales
    g=np.asarray(sol.value(opt.g)).ravel();lo=np.asarray(sol.value(opt.lbg)).ravel();hi=np.asarray(sol.value(opt.ubg)).ravel()
    violation=float(max(0.,np.max(lo-g),np.max(g-hi)))
    status=stats.get("return_status","");iterations=stats.get("iterations",{})
    cost=float(sol.value(objective));selected="solver_final"
    if best and (violation>1e-3 or best["objective"]<cost-1e-6*max(1.,abs(cost))):
        x=best["states"];u=best["commands"];cost=best["objective"];violation=best["violation"]
        selected=best["kind"];success=False
    info=dict(method="direct multiple shooting / IPOPT with exact smooth-region Lagrangian curvature",success=bool(success and status=="Solve_Succeeded"),
        status=status,objective=cost,selected_iterate=selected,iterations=int(stats.get("iter_count",0)),
        constraint_violation=violation,dual_infeasibility=float(iterations.get("inf_du",[np.inf])[-1]) if selected=="solver_final" else None,
        wall_s=time.monotonic()-start,canonical_symbolic_error=map_error,globally_optimal=False,
        all_six_channels_free=True,programmed_maneuver_segments=False)
    return dict(times=np.linspace(0,T,N+1),states=x,commands=u,scales=scales,reference=ref,solver=info),allocator
