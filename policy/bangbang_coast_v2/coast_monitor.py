"""Public-state-only ballistic coast forecast and event-triggered reinterception.

Offline-safe. Uses the same local gravity, body-rate and reporting-offset
conventions as the nominal identified 13-state model. Zero requested thrust.

Forecasts are heuristics, not certified collision-free or native-dynamics truth.
"""
from __future__ import annotations
import math
import numpy as np
from policy.bangbang_coast_v2.core import latent_from_reported
from system_identification.model import matrix, increment


def predict_ballistic(state, model, target, horizon_s, step_s, capture_radius_m):
    """Propagate current state with no controls; find future closest approach.

    All positions/velocities are in Phobos P coordinates. q is passive P->B;
    reporting offset is body-fixed and rotates with q. The local model does
    not include arbitrary debris impacts, uncertain terrain or full ephemeris.
    """
    if (not np.isfinite(horizon_s) or horizon_s <= 0 or
            not np.isfinite(step_s) or step_s <= 0 or
            not 0 < capture_radius_m < 50):
        raise ValueError('Invalid ballistic forecast horizon/step/capture radius')
    p,v,q,w,_=latent_from_reported(state,model)
    p=p.copy();v=v.copy();q=q.copy();w=w.copy()
    mu=float(model.p['phobos_mu_prior_m3_s2'])
    gbias=np.asarray(model.p['uniform_external_acceleration_P_m_s2'],float)
    abias=np.asarray(model.p['angular_bias_B_rad_s2'],float)
    gamma=float(model.p['gyro_coefficient'])
    ell=np.asarray(model.ell,float)
    closest=float('inf');at_closest=0.;arrival=None
    closest_position=None;closest_target=None
    trailing_distance=float('inf');last_distance=float('inf')
    nsteps=int(math.ceil(horizon_s/step_s))
    elapsed=0.
    for i in range(nsteps+1):
        current_reported=p+matrix(q).T@ell
        dist=float(np.linalg.norm(current_reported-target.at(elapsed)))
        if not np.isfinite(dist):raise ValueError('Nonfinite ballistic distance')
        if dist<closest:
            closest=dist;at_closest=elapsed
            closest_position=current_reported.copy()
            closest_target=np.asarray(target.at(elapsed),float).copy()
        if arrival is None and dist<=capture_radius_m:
            arrival=elapsed
            break  # A conservative modeled interception already exists.
        if i==nsteps:break
        trailing_distance=last_distance
        last_distance=dist
        dt=min(step_s,horizon_s-elapsed)
        if dt<=1e-10:break
        # Midpoint Euler rigid-body rate, matching model.py's effective law.
        def rate_deriv(omega):
            return abias-gamma*(model.Ji@np.cross(omega,model.J@omega))
        dw1=rate_deriv(w)
        wm=w+.5*dt*dw1
        dw=rate_deriv(wm)
        pm=p+.5*dt*v
        radius=float(np.linalg.norm(pm))
        if radius<1.:raise ValueError('Predicted trajectory enters gravity singularity')
        accel=-mu*pm/radius**3+gbias
        p=p+v*dt+.5*accel*dt*dt
        v=v+accel*dt
        q=increment(q,wm,dt)
        w=w+dw*dt
        elapsed+=dt
    closing_at_horizon=bool(arrival is None and elapsed>=horizon_s-1e-8 and
                             np.isfinite(last_distance) and dist<last_distance)
    return {'closest_position_P_m':closest_position.tolist(),
            'closest_target_P_m':closest_target.tolist(),
            'predicted_closest_m':float(closest),
            'time_to_closest_s':float(at_closest),
            'predicted_arrival_s':(None if arrival is None else float(arrival)),
            'predicted_capture':bool(arrival is not None),
            'distance_at_horizon_m':float(dist),
            'prediction_horizon_s':float(elapsed),
            'still_closing_at_horizon':closing_at_horizon}


class CoastMissMonitor:
    """Consecutive-confirmation hysteresis, no thrust while observing."""
    def __init__(self, settings):
        self.config=settings
        self.consecutive=0

    def update(self, forecast, horizon_covers_deadline=False):
        # A model still closing beyond a shorter-than-mission horizon has not
        # yet demonstrated a miss. Demand longer prediction before replanning.
        unresolved=(forecast['still_closing_at_horizon'] and
                    not horizon_covers_deadline)
        unsafe=(not forecast['predicted_capture'] and not unresolved and
                forecast['predicted_closest_m']>self.config['trigger_closest_m'])
        self.consecutive=self.consecutive+1 if unsafe else 0
        return {'unsafe':bool(unsafe), 'unresolved':bool(unresolved),
                'confirmations':int(self.consecutive),
                'trigger_replan':bool(self.consecutive>=self.config['confirmations_required'])}
