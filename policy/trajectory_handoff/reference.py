"""Flight-side math. Imports neither SciPy, CasADi, SDK nor the allocation LP.

13-state plant: p_C(P), v_C(P), passive q_PB(xyzw), omega_B.
12-state error: dp, dv, physical body attitude increment, domega.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(q):
    q = np.asarray(q, float); q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def multiply(a, b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3], b[:3]), a[3]*b[3]-a[:3]@b[:3]]


def increment(q, theta):
    """Physical body rotation theta; q is passive P->B."""
    v = -np.asarray(theta, float); a = np.linalg.norm(v)
    dq = np.r_[v*(np.sin(a/2)/a if a > 1e-10 else .5-a*a/48), np.cos(a/2)]
    out = multiply(dq, q)
    return out/np.linalg.norm(out)


def error(x, ref):
    a = np.asarray(x, float); b = np.asarray(ref, float)
    dq = multiply(a[6:10], np.r_[-b[6:9], b[9]])
    if dq[3] < 0: dq = -dq
    n = np.linalg.norm(dq[:3])
    theta = -dq[:3]*(2*np.arctan2(n, max(dq[3], 0))/n if n > 1e-10 else 2.)
    return np.r_[a[:6]-b[:6], theta, a[10:13]-b[10:13]]


def retract(ref, delta):
    out = np.asarray(ref, float).copy(); d = np.asarray(delta, float)
    out[:6] += d[:6]; out[6:10] = increment(out[6:10], d[6:9]); out[10:13] += d[9:12]
    return out


def interpolate_state(times, states, t):
    i = max(0, min(len(times)-2, int(np.searchsorted(times, t, side='right')-1)))
    r = np.clip((t-times[i])/max(times[i+1]-times[i], 1e-12), 0, 1)
    # Shortest quaternion interpolation. Normalized lerp on the dense cache.
    a = states[i]; b = states[i+1].copy()
    if a[6:10]@b[6:10] < 0: b[6:10] *= -1
    out = (1-r)*a+r*b; out[6:10] /= np.linalg.norm(out[6:10])
    return out


def clip_request(u, caps):
    out = np.asarray(u, float).copy()
    for sl, cap in ((slice(0, 3), caps[0]), (slice(3, 6), caps[1])):
        n = np.linalg.norm(out[sl])
        if n > cap: out[sl] *= cap/n
    return out


def plan_end(plan):
    if plan.get('kind')=='schedule':return plan_end(plan['next']) if plan['next'] is not None else plan['times'][-1]
    return float(min(plan['times'][-1],plan.get('execute_until_sim_s',plan['times'][-1])))


def coast_plan(state,when,phase,timeout=600.):
    """Explicit continuation; no hidden return to a retired nominal trajectory."""
    return dict(kind='coast',times=[float(when),float(when+timeout)],phase=float(phase),
        hold_quaternion=np.asarray(state[6:10],float).tolist(),source='explicit_coast',
        clearance_verified=False)


def coast_control(reference,phase,state,s,quaternion=None):
    # Fallback: zero translational request, bounded angular feedback. This does
    # not promise hover, braking, or geometric clearance under external gravity.
    _,_,K,_,_=reference.at(phase);desired=np.asarray(state,float).copy()
    if quaternion is not None:desired[6:10]=quaternion
    desired[10:13]=0.;fb=K@error(state,desired)
    ratio=np.max(abs(fb[3:])/reference.scales[3:])
    if ratio>s['feedback_limit']:fb*=s['feedback_limit']/ratio
    return clip_request(np.r_[np.zeros(3),fb[3:]],reference.caps)


def schedule_control(plan,state,when,reference,phase,s,*,check_tracking=False):
    if plan is None:return reference.control(phase,state,s['feedback_limit'])
    if plan.get('kind')=='schedule':
        if when>=plan['switch_sim_s']:
            return schedule_control(plan['next'],state,when,reference,plan['next_phase']+when-plan['switch_sim_s'],s,check_tracking=check_tracking)
        return schedule_control(plan['current'],state,when,reference,phase,s,check_tracking=check_tracking)
    if plan.get('kind')=='coast' or when>=plan_end(plan):
        q=plan.get('hold_quaternion')
        if q is None and 'states' in plan:
            q=interpolate_state(np.asarray(plan['times']),np.asarray(plan['states']),plan_end(plan))[6:10]
        return coast_control(reference,phase,state,s,q)
    if when<plan['times'][0]:return None
    ts=np.asarray(plan['times']);xs=np.asarray(plan['states']);us=np.asarray(plan['commands'])
    j=min(len(us)-1,max(0,int(np.searchsorted(ts,when,side='right')-1)))
    delta=error(state,interpolate_state(ts,xs,when))
    if check_tracking and (np.linalg.norm(delta[:3])>s['plan_position_tolerance_m'] or
        np.linalg.norm(delta[3:6])>s['plan_velocity_tolerance_m_s'] or
        np.linalg.norm(delta[6:9])>s['plan_attitude_tolerance_rad'] or
        np.linalg.norm(delta[9:])>s['plan_rate_tolerance_rad_s']):return None
    fb=np.asarray(plan['gains'][j])@delta;ratio=np.max(abs(fb)/reference.scales)
    if ratio>s['feedback_limit']:fb*=s['feedback_limit']/ratio
    return clip_request(us[j]+fb,reference.caps)


def packet_row(packet):
    q = np.asarray(packet['q_PB_xyzw_reported'], float)
    return np.r_[packet['position_P_m'], packet['velocity_P_m_s'], q,
                 matrix(q)@np.asarray(packet['omega_relP_P_rad_s'], float)]


def latent_state(times, rows, parameters):
    """Same causal reporting adapter as nominal_model.json's published backend."""
    ts = np.asarray(times, float); ys = np.asarray(rows, float)
    now = ts[-1]; use = ts >= now-.3
    x = ys[-1].copy(); slopes = []
    for sl in (slice(3, 6), slice(10, 13)):
        slope = (np.linalg.lstsq(np.c_[ts[use]-now, np.ones(use.sum())], ys[use, sl], rcond=None)[0][0]
                 if use.sum() >= 3 else np.zeros(3))
        slopes.append(slope)
    x[:3] -= matrix(x[6:10]).T@np.asarray(parameters['reporting_offset_B_m'])
    x[3:6] += parameters['velocity_report_delay_s']*slopes[0]-np.asarray(parameters['velocity_report_bias_P_m_s'])
    x[10:13] += parameters['rate_report_delay_s']*slopes[1]-np.asarray(parameters['rate_report_bias_B_rad_s'])
    return x


class Reference:
    def __init__(self, folder, model_path, verify=True):
        folder = Path(folder)
        self.manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8-sig'))
        if self.manifest.get('schema_version') != 1: raise ValueError('Unsupported trajectory cache')
        if verify and sha256(model_path) != self.manifest['model_sha256']:
            raise ValueError('Trajectory/model mismatch; regenerate offline, never silently reuse')
        if verify and sha256(folder/'nominal_trajectory.npz') != self.manifest['trajectory_sha256']:
            raise ValueError('Trajectory cache checksum mismatch')
        with np.load(folder/'nominal_trajectory.npz', allow_pickle=False) as data:
            for key in ('times', 'states', 'command_times', 'commands', 'gain_times', 'gains', 'value_gradient', 'curvature', 'fuel_remaining'):
                setattr(self, key, np.asarray(data[key], float))
        if (self.states.shape != (len(self.times), 13) or self.gains.shape != (len(self.gain_times), 6, 12)
                or self.commands.shape != (len(self.command_times), 6)
                or self.value_gradient.shape != (len(self.gain_times),12)
                or self.curvature.shape != (len(self.gain_times),12,12)
                or self.fuel_remaining.shape != self.times.shape
                or not all(np.isfinite(getattr(self, k)).all() for k in ('times','states','commands','gains','value_gradient','curvature','fuel_remaining'))):
            raise ValueError('Invalid trajectory cache arrays')
        if (len(self.times)<2 or self.times[0]!=0 or np.any(np.diff(self.times)<=0)
                or np.any(np.diff(self.command_times)<=0) or np.any(np.diff(self.gain_times)<=0)
                or self.command_times[0]!=0 or self.gain_times[0]!=0
                or not np.allclose(np.linalg.norm(self.states[:,6:10],axis=1),1.,atol=1e-6)):
            raise ValueError('Invalid cache time/quaternion contract')
        self.duration = float(self.times[-1]); self.caps = self.manifest['request_norm_caps']
        self.scales = np.asarray(self.manifest['request_scales'])

    def at(self, phase):
        phase = float(np.clip(phase, 0, self.duration))
        x = interpolate_state(self.times, self.states, phase)
        j = max(0, min(len(self.commands)-1, int(np.searchsorted(self.command_times, phase, side='right')-1)))
        k = max(0, min(len(self.gains)-1, int(np.searchsorted(self.gain_times, phase, side='right')-1)))
        return x, self.commands[j].copy(), self.gains[k], self.value_gradient[k], self.curvature[k]

    def control(self, phase, state, feedback_limit=1.):
        x, ff, K, _, _ = self.at(phase)
        fb = K@error(state, x)
        ratio = np.max(abs(fb)/self.scales)
        if ratio > feedback_limit: fb *= feedback_limit/ratio
        return clip_request(ff+fb, self.caps)

    def remaining(self, phase):
        return float(np.interp(phase, self.times, self.fuel_remaining))

    def closest_phase(self, state, center, window=15.):
        use = np.flatnonzero(abs(self.times-center) <= window)
        if not len(use): return float(np.clip(center, 0, self.duration))
        # Position AND velocity; never resolve a crossing solely by position.
        cost = np.sum((self.states[use,:3]-state[:3])**2, axis=1)+4*np.sum((self.states[use,3:6]-state[3:6])**2, axis=1)
        return float(self.times[use[np.argmin(cost)]])
