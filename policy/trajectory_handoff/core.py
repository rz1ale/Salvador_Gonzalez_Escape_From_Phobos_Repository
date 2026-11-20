"""Numeric adapter to the existing canonical model; no simulator imports."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from system_identification.model import Model, matrix, gravity
from .reference import error, retract


class CertifiedAllocationCache:
    """Reuse ONLY a unique optimum of the unchanged minimum-sum allocation LP.

    A feasible primal basis and strictly positive nonbasic reduced costs certify
    the UNIQUE solution. Ties, degeneracy, bad conditioning and boundary numerical
    failures fall back to the declared original solver. Different tied optimal
    allocations can have different pulse peaks/fuel, so they are never substituted.
    """
    def __init__(self,original,maximum_bases=64):
        self.original=original;self.maximum_bases=int(maximum_bases)
        self.bases=OrderedDict();self.stats=dict(exact_hits=0,certified_hits=0,full_solves=0)

    def __getattr__(self,name):return getattr(self.original,name)

    def _learn(self,allocation):
        indices=tuple(np.flatnonzero(allocation>max(1e-12,1e-9*float(allocation.max()))))
        if len(indices)!=6:return
        if indices in self.bases:self.bases.move_to_end(indices);return
        B=self.A[list(indices)].T
        try:
            inverse=np.linalg.inv(B);dual=np.linalg.solve(B.T,np.ones(6))
        except np.linalg.LinAlgError:return
        if not np.isfinite(inverse).all() or np.linalg.norm(B@inverse-np.eye(6),ord=np.inf)>1e-8:return
        inactive=np.ones(len(self.A),bool);inactive[list(indices)]=False
        reduced=1.-self.A@dual
        tolerance=max(1e-8,256*np.finfo(float).eps*(1+np.linalg.norm(self.A,ord=np.inf)*np.linalg.norm(dual,ord=np.inf)))
        if np.any(inactive) and np.min(reduced[inactive])<=tolerance:return
        self.bases[indices]=(B,inverse)
        while len(self.bases)>self.maximum_bases:self.bases.popitem(last=False)

    def solve(self,b):
        b=np.asarray(b,float);key=tuple(b)
        if np.linalg.norm(b)<1e-12:return self.original.solve(b)
        if key in self.cache:
            self.stats['exact_hits']+=1;out=self.cache[key];self._learn(out);return out
        for indices,(B,inverse) in reversed(self.bases.items()):
            basic=inverse@b
            if np.min(basic)<0 or not np.isfinite(basic).all():continue
            if np.linalg.norm(B@basic-b,ord=np.inf)>1e-12*max(1.,np.linalg.norm(b,ord=np.inf)):continue
            out=np.zeros(len(self.A));out[list(indices)]=basic
            if len(self.cache)>50000:self.cache.clear()
            self.cache[key]=out;self.stats['certified_hits']+=1
            self.bases.move_to_end(indices);return out
        self.stats['full_solves']+=1;out=self.original.solve(b);self._learn(out);return out

    def mean(self,u,period):return type(self.original).mean(self,u,period)
    def peaks(self,commands,minimum_period):return type(self.original).peaks(self,commands,minimum_period)


class Plant:
    def __init__(self, model, step=.1,*,certified_allocation=True):
        if certified_allocation and not isinstance(model.allocator,CertifiedAllocationCache):
            model.allocator=CertifiedAllocationCache(model.allocator)
        self.model = model; self.step_s = float(step)
        self.scales = np.array([model.h/max(model.allocator.solve(np.eye(6)[j]*sgn).max()
                            for sgn in (-1,1)) for j in range(6)])
        self.scales = np.maximum(self.scales, 1e-7)
        self.delay_l = float(model.p['linear_effect_delay_s']); self.delay_a = float(model.p['angular_effect_delay_s'])

    def base(self, u): return self.model.allocator.mean(u, self.model.h)

    def canonical_request(self,u):
        """Model-equivalent request at/below its allocation knee.

        Saturated raw magnitudes have a flat radial derivative. Canonicalizing
        before linearization restores local authority to reduce achieved thrust.
        This does not replace or alter the declared six-axis allocation LP.
        """
        return self.model.allocator.clip(self.model.h*self.base(u))

    def fuel_rate(self, u):
        m = self.model; b = m.allocator.clip(u); lam = m.allocator.solve(b)
        return float(m.allocator.forces@lam/max(m.h, float(lam.max()))/(80.*9.8))

    def advance_base(self, x, dt, base):
        out = np.asarray(x, float).copy(); left = float(dt)
        while left > 1e-9:
            d = min(left, self.step_s); out = self.model._advance(out, d, base); left -= d
        return out

    def transition(self, x, old, u, dt):
        """Constant old/new request, separate channel effect delays at the edge."""
        a = self.base(old); b = self.base(u); out = np.asarray(x, float).copy()
        edges = sorted(set([0., min(dt,self.delay_l), min(dt,self.delay_a), float(dt)]))
        for lo, hi in zip(edges[:-1], edges[1:]):
            mid = (lo+hi)/2
            base = np.r_[b[:3] if mid>=self.delay_l else a[:3], b[3:] if mid>=self.delay_a else a[3:]]
            out = self.advance_base(out, hi-lo, base)
        return out

    def rollout(self, x, times, commands, old=None):
        times = np.asarray(times, float); commands = np.asarray(commands, float)
        if commands.shape != (len(times)-1,6): raise ValueError('Each interval needs one request')
        states = [np.asarray(x,float).copy()]; prev = commands[0] if old is None else old
        for dt,u in zip(np.diff(times),commands):
            states.append(self.transition(states[-1],prev,u,dt)); prev=u
        return np.asarray(states)

    def linearize(self, x, old, u, dt):
        """Error-state A, new/previous normalized-request B. Includes allocation."""
        nominal = self.transition(x,old,u,dt); A=np.empty((12,12)); B=np.empty((12,6)); D=np.empty((12,6))
        epsx = np.r_[np.full(3,.01),np.full(3,.001),np.full(3,1e-4),np.full(3,1e-4)]
        for j,e in enumerate(epsx):
            d=np.eye(12)[j]*e
            yp=self.transition(retract(x,d),old,u,dt); ym=self.transition(retract(x,-d),old,u,dt)
            A[:,j]=(error(yp,nominal)-error(ym,nominal))/(2*e)
        # LP tolerance needs a request perturbation well above 1e-10.
        eps=.002
        for j in range(6):
            d=np.eye(6)[j]*self.scales[j]*eps
            B[:,j]=(error(self.transition(x,old,u+d,dt),nominal)-error(self.transition(x,old,u-d,dt),nominal))/(2*eps)
            D[:,j]=(error(self.transition(x,old+d,u,dt),nominal)-error(self.transition(x,old-d,u,dt),nominal))/(2*eps)
        return nominal,A,B,D

    def affine_linearize(self,x,old,u,dt):
        """Frozen continuous error Jacobian, exact affine ZOH and channel delay.

        Cheap LTV planning approximation; nonlinear coupled rollouts remain the
        acceptance check. The allocator derivative includes shared saturation.
        """
        from scipy.linalg import expm
        def skew(v):
            a,b,c=v;return np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])
        m=self.model;F=np.zeros((12,12));F[:3,3:6]=np.eye(3)
        p=np.asarray(x[:3]);r=max(np.linalg.norm(p),1.)
        F[3:6,:3]=m.p['phobos_mu_prior_m3_s2']*(3*np.outer(p,p)/r**5-np.eye(3)/r**3)
        R=matrix(x[6:10]).T;base=self.base(u);w=x[10:13]
        F[3:6,6:9]=-R@skew(m.Kl@base[:3]);F[6:9,6:9]=-skew(w);F[6:9,9:]=np.eye(3)
        F[9:,9:]=m.p['gyro_coefficient']*m.Ji@(skew(m.J@w)-skew(w)@m.J)
        derivative=np.empty((6,6));eps=.002
        for j in range(6):
            d=np.eye(6)[j]*self.scales[j]*eps
            derivative[:,j]=(self.base(u+d)-self.base(u-d))/(2*eps)
        Gl=np.zeros((12,6));Ga=Gl.copy()
        Gl[3:6]=R@m.Kl@derivative[:3];Ga[9:]=m.Ka@derivative[3:]
        def zoh(G,h):
            M=np.zeros((18,18));M[:12,:12]=F;M[:12,12:]=G
            E=expm(M*max(0.,h));return E[:12,:12],E[:12,12:]
        A,total=zoh(Gl+Ga,dt)
        _,Bl=zoh(Gl,dt-self.delay_l);_,Ba=zoh(Ga,dt-self.delay_a)
        return A,Bl+Ba,total-Bl-Ba

    def reported_positions(self, xs):
        return np.array([x[:3]+matrix(x[6:10]).T@self.model.ell for x in xs])


def score(time_s, fuel, damage, constants):
    if time_s<=0 or fuel<0: raise ValueError('Score estimate requires positive time/nonnegative fuel')
    return 50.+100.*(constants['NominalTime_s']/time_s-2*damage/constants['DamageThreshold_J']-fuel/constants['NominalFuel_kg'])
