"""NumPy-only quaternion/error-state math. Passive P->B, scalar-last xyzw."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(settings):
    # Execution-only label/status/test cutoff do not change the solved problem.
    value={k:settings[k] for k in ("schema_version","design","feedback","solver")}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def trajectory_signature(settings):
    value={k:settings[k] for k in ("schema_version","design","solver")}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def settings(path):
    s=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if set(s)!={"schema_version","design","feedback","solver","execution"} or s["schema_version"]!=1:
        raise ValueError("Unsupported settings schema")
    d=s["design"]
    if set(d)!={"distance_m","duration_s","steps","forward_velocity_m_s","Q","Q_final","R",
            "terminal_pointing_deg","terminal_pointing_fraction","terminal_rate_rad_s","terminal_position_tolerance_m"}:
        raise ValueError("Unknown/missing design key")
    for k in ("distance_m","duration_s","terminal_rate_rad_s"):
        if not isinstance(d[k],(int,float)) or not np.isfinite(d[k]) or d[k]<=0:
            raise ValueError("Expected positive design."+k)
    if d["terminal_pointing_deg"] is not None and not 0<d["terminal_pointing_deg"]<90:raise ValueError("Invalid pointing degrees")
    if not 0<d["terminal_pointing_fraction"]<=.5:raise ValueError("Invalid pointing fraction")
    if type(d["steps"]) is not int or not 4<=d["steps"]<=200:
        raise ValueError("Invalid terminal pointing tolerance/steps")
    for name in ("Q","Q_final"):
        if set(d[name])!={"position","velocity","attitude","rate"}:
            raise ValueError("Invalid "+name)
        for a in d[name].values():
            v=np.asarray(a,float)
            if v.shape!=(3,) or not np.isfinite(v).all() or np.any(v<0):raise ValueError("Invalid "+name)
    r=np.asarray(d["R"],float)
    if r.shape!=(6,) or not np.isfinite(r).all() or np.any(r<=0):raise ValueError("R must have six positive entries")
    if d["forward_velocity_m_s"] is None:
        if any(np.any(np.asarray(d[k]["velocity"])!=0) for k in ("Q","Q_final")):
            raise ValueError("Set forward_velocity_m_s before assigning velocity weights; no implicit zero-velocity goal")
    elif not np.isfinite(d["forward_velocity_m_s"]) or d["forward_velocity_m_s"]<0:
        raise ValueError("Invalid forward velocity")
    p=d["terminal_position_tolerance_m"]
    if p is not None and (not np.isfinite(p) or p<=0):raise ValueError("Invalid position tolerance")
    expected={"feedback":{"position_weight_multiplier","rate_weight_multiplier","period_s","maximum_normalized_correction","continuation_angle_weight","continuation_rate_weight",
                "continuation_effort_weight","continuation_alpha_limit_rad_s2","continuation_pointing_deadband_deg","continuation_rate_deadband_rad_s"},
              "solver":{"integration_step_s","max_iterations","max_wall_s","optimality_tolerance","constraint_tolerance","allocation_derivative_step","accept_verified_candidate"},
              "execution":{"label","test_duration_s","status_period_s"}}
    for group,keys in expected.items():
        if set(s[group])!=keys:raise ValueError("Unknown/missing "+group+" setting")
        for k,v in s[group].items():
            if k in ("label","test_duration_s"):continue
            if not np.isfinite(v) or v<0:raise ValueError("Invalid "+group+"."+k)
    if not 0<s["feedback"]["period_s"]<=.5 or min(s["feedback"]["continuation_effort_weight"],s["feedback"]["continuation_angle_weight"])<=0:
        raise ValueError("Invalid feedback period/continuation weights")
    if not 0<s["solver"]["integration_step_s"]<=.1 or s["solver"]["max_wall_s"]<=0:
        raise ValueError("Invalid solver integration/budget")
    if type(s["solver"]["max_iterations"]) is not int or s["solver"]["max_iterations"]<1:
        raise ValueError("Invalid max_iterations")
    if type(s["solver"]["accept_verified_candidate"]) is not bool:raise ValueError("accept_verified_candidate must be Boolean")
    limit=s["execution"]["test_duration_s"]
    if limit is not None and (not np.isfinite(limit) or limit<=0):raise ValueError("Invalid test duration")
    return s


def matrix(q):
    q=np.asarray(q);q=q/np.sqrt(q@q);x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def multiply(a,b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),a[3]*b[3]-a[:3]@b[:3]]


def increment(q,theta):
    theta=np.asarray(theta,float);v=-theta;a=np.linalg.norm(v)
    dq=np.r_[v*(np.sin(a/2)/a if a>1e-10 else .5),np.cos(a/2)]
    out=multiply(dq,q);return out/np.linalg.norm(out)


def error(x,ref):
    dq=multiply(x[6:10],np.r_[-ref[6:9],ref[9]])
    if dq[3]<0:dq=-dq
    n=np.linalg.norm(dq[:3]);theta=-dq[:3]*(2*np.arctan2(n,max(dq[3],0))/n if n>1e-10 else 2.)
    return np.r_[x[:6]-ref[:6],theta,x[10:13]-ref[10:13]]


def retract(x,delta):
    out=np.asarray(x,float).copy();out[:6]+=delta[:6]
    out[6:10]=increment(out[6:10],delta[6:9]);out[10:]+=delta[9:];return out


def direction_rotation(direction):
    d=np.asarray(direction,float);d=d/max(np.linalg.norm(d),1e-12)
    v=np.cross([1.,0,0],d);n=np.linalg.norm(v)
    return (v/n if n>1e-10 else np.array([0.,1,0])),float(np.arctan2(n,np.clip(d[0],-1,1)))


def reported(x,definition):
    return x[:3]+matrix(x[6:10]).T@np.asarray(definition["parameters"]["reporting_offset_B_m"])


def mission_reference(x0,target,definition,d):
    p0=reported(x0,definition);target=np.asarray(target,float)
    xf=target+50.*(p0-target)/np.linalg.norm(p0-target)
    n=(xf-p0)/np.linalg.norm(xf-p0);pd=p0+d["distance_m"]*n
    if d["distance_m"]>=np.linalg.norm(xf-p0):raise ValueError("Handoff distance must precede the target sphere")
    axis,angle=direction_rotation(matrix(x0[6:10])@n)
    qd=increment(x0[6:10],axis*angle)
    vd=n*d["forward_velocity_m_s"] if d["forward_velocity_m_s"] is not None else np.zeros(3)
    return dict(p0=p0,pd=pd,xf=xf,n=n,qd=qd,vd=vd,axis=axis,angle=angle)


def terminal_limits(s,reference):
    d=s["design"]
    return dict(pointing_deg=float(d["terminal_pointing_deg"] if d["terminal_pointing_deg"] is not None
        else np.degrees(reference["angle"])*d["terminal_pointing_fraction"]),
        rate_rad_s=float(d["terminal_rate_rad_s"]),position_m=d["terminal_position_tolerance_m"])


def interpolate(times,states,t):
    i=max(0,min(len(times)-2,int(np.searchsorted(times,t,side="right")-1)))
    r=np.clip((t-times[i])/(times[i+1]-times[i]),0,1)
    a=states[i];b=states[i+1].copy()
    if a[6:10]@b[6:10]<0:b[6:10]*=-1
    out=(1-r)*a+r*b;out[6:10]/=np.linalg.norm(out[6:10]);return out


def clip(u,caps):
    out=np.asarray(u,float).copy()
    for sl,cap in ((slice(0,3),caps[0]),(slice(3,6),caps[1])):
        n=np.linalg.norm(out[sl])
        if n>cap:out[sl]*=cap/n
    return out
