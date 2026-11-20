"""Pure math for the acquisition controller and public radar guard; no SDK imports."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

def clip_norm(v, limit):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v * min(1., limit/n) if n > 0 else v.copy()

def rotation(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()  # passive P -> B convention

def attitude_error(B, reference):
    # Bdot=-[omega_B]x B, hence this rotation vector has restorative sign.
    return Rotation.from_matrix(B @ reference.T).as_rotvec()

def relative_geometry(p, launch, up):
    delta = np.asarray(p)-launch
    h = float(delta@up)
    return h, delta-h*up

def braking_distance(speed, accel, delay, margin, adverse=0.):
    speed=max(0.,speed);adverse=max(0.,adverse)
    after=speed+adverse*delay
    return speed*delay+.5*adverse*delay*delay+after*after/(2*accel)+margin

def navigation(state, target, B_reference, gravity, settings):
    p, v, B, wB = state['p'], state['v'], state['B'], state['wB']
    desired_v = clip_norm(settings['position_gain_s_inv']*(target-p), settings['control_speed_m_s'])
    net = clip_norm(settings['velocity_gain_s_inv']*(desired_v-v), settings['control_net_accel_m_s2'])
    thrust_P = clip_norm(net-gravity, settings['control_request_accel_m_s2'])
    alpha = clip_norm(settings['attitude_gain_s_inv2']*attitude_error(B,B_reference)
                      - settings['rate_gain_s_inv']*wB, settings['control_angular_accel_rad_s2'])
    # Historical unsaturated front end: requested b/h ~= mean achieved acceleration.
    # h is a documented PRIOR, never the controller period or a private live tick.
    h = settings['native_step_prior_s']
    return np.r_[B@thrust_P, alpha]*h

def radar_threat(rows, now, sample, settings):
    """Conservative local check using measured spheres/radial rates, not debris truth.

    Tangential motion remains a bounded assumption. Missing objects, noisy ranges,
    nonspherical geometry and finite coverage prevent a collision guarantee.
    """
    nearest, considered = float('inf'), 0
    issue = None
    H = settings['radar_lookahead_s']
    for row in rows:
        z = np.asarray(row,float)
        if z.shape != (3,3) or not np.isfinite(z).all():
            continue
        r = z[0].copy()
        rc, vr, diameter = z[1]
        stamp = z[2,0]
        if diameter <= 0 or stamp <= 0 or stamp > now+.1:
            continue
        age = now-stamp
        if age > settings['radar_max_age_s']:
            continue
        distance = np.linalg.norm(r)
        if distance < 1e-6:
            return 'radar_near_zero_range', -1., considered+1
        considered += 1
        direction = r/distance
        radius = max(diameter/2, distance-max(0.,rc))
        own_tangent = sample['v'] - float(sample['v']@direction)*direction
        relative_v = vr*direction - own_tangent
        padding = settings['radar_min_clearance_m'] + settings['radar_sigma_factor']*settings['radar_position_sigma_m']
        for horizon in np.linspace(0,H,5):
            t = age+horizon
            # Covers bounded unknown tangential speed and own future acceleration.
            uncertainty = settings['radar_tangent_speed_bound_m_s']*t
            uncertainty += .5*settings['radar_own_accel_bound_m_s2']*t*t
            clearance = np.linalg.norm(r+relative_v*t)-radius-padding-uncertainty
            nearest = min(nearest,float(clearance))
        if nearest <= 0:
            issue = 'radar_clearance_guard'
    return issue, (nearest if np.isfinite(nearest) else None), considered

def fit_coast(samples, washout):
    if len(samples) < 4:
        raise ValueError('Too few coast samples')
    t = np.array([x['t'] for x in samples]); v = np.array([x['v'] for x in samples])
    mask = t >= t[0]+washout
    t,v = t[mask],v[mask]
    if len(t)<3 or np.ptp(t)<.3:
        raise ValueError('Too little coast after reporting washout')
    design = np.c_[np.ones(len(t)),t-t[0]]
    fit = np.linalg.lstsq(design,v,rcond=None)[0]
    rms = float(np.sqrt(np.mean((design@fit-v)**2)))
    return fit[1], rms
