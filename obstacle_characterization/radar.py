"""Policy-independent public radar tracking in Phobos coordinates.

Numeric six-state CV filters, global gated association, explicit unresolved
observations, packet deduplication, own-pose history, and clock registration.
Only caller-supplied public packets are used. No simulator/SDK imports.
"""
from __future__ import annotations
from collections import OrderedDict, deque
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = '0.1.1'
SCHEMA_VERSION = 1
FRAME = 'Phobos_body_fixed'
DEFAULTS = Path(__file__).with_name('settings.json')


def load_settings(value=None):
    defaults = json.loads(DEFAULTS.read_text(encoding='utf-8-sig'))
    if value is None:
        supplied = {}
    elif isinstance(value, dict):
        supplied = copy.deepcopy(value)
    else:
        supplied = json.loads(Path(value).read_text(encoding='utf-8-sig'))
    if not isinstance(supplied, dict) or set(supplied)-set(defaults):
        raise ValueError('Unknown obstacle-characterization setting')
    s = dict(defaults, **supplied)
    if s['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported settings schema')
    for key, v in s.items():
        if key in ('schema_version', 'radar_origin_offset_B_m'):
            continue
        if key == 'track_expiry_mode':
            if v not in ('drop_unassessed','retain_occupancy'):
                raise ValueError('Invalid track_expiry_mode')
            continue
        if key == 'birth_initialization':
            if v not in ('radial_prior','two_position'):
                raise ValueError('Invalid birth_initialization')
            continue
        if key == 'timestamp_offset_s':
            if v is not None and (type(v) not in (float, int) or not math.isfinite(v)):
                raise ValueError('Invalid timestamp_offset_s')
        elif type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError('Expected nonnegative finite setting: '+key)
    for key in ('poll_period_s', 'history_s', 'history_max_gap_s', 'velocity_window_s',
                'position_sigma_m', 'radial_velocity_sigma_m_s', 'diameter_sigma_fraction',
                'initial_velocity_std_m_s', 'association_cost_gate',
                'max_observation_age_s', 'track_timeout_s', 'forecast_horizon_s',
                'diagnostic_log_period_s'):
        if s[key] <= 0:
            raise ValueError('Expected positive setting: '+key)
    for key in ('confirmation_hits', 'max_tracks', 'max_unresolved_observations', 'max_cached_measurements'):
        if type(s[key]) is not int or s[key] <= 0:
            raise ValueError('Expected positive integer: '+key)
    offset = np.asarray(s['radar_origin_offset_B_m'], float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('radar_origin_offset_B_m must be a finite body vector')
    if s['history_s'] < s['max_observation_age_s']+1:
        raise ValueError('Own history must exceed retained radar age')
    if s['track_timeout_s'] < s['max_observation_age_s']:
        raise ValueError('Track timeout must not precede maximum observation age')
    if s['confirmation_hits'] < 2 or s['max_tracks'] < 500:
        raise ValueError('Need at least two independent confirmations and room for 500 observations')
    return s


def rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < .5:
        raise ValueError('Invalid own quaternion')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def cv_predict(x, P, dt, q):
    if dt < -1e-8:
        raise ValueError('Out-of-order filter prediction')
    dt = max(0., float(dt))
    out = x.copy()
    out[:3] += dt*x[3:]
    covariance = P.copy()
    I = np.eye(3)
    covariance[:3,:3] += dt*(P[:3,3:]+P[3:,:3])+dt*dt*P[3:,3:]+q*dt**3/3*I
    covariance[:3,3:] += dt*P[3:,3:]+q*dt*dt/2*I
    covariance[3:,:3] = covariance[:3,3:].T
    covariance[3:,3:] += q*dt*I
    return out, covariance


def cv_predict_many(x, P, dt, q):
    """Same equations as cv_predict, batched for snapshot publication."""
    out, covariance = x.copy(), P.copy()
    d = dt[:,None,None]
    I = np.eye(3)[None,:,:]
    out[:,:3] += dt[:,None]*x[:,3:]
    covariance[:,:3,:3] += d*(P[:,:3,3:]+P[:,3:,:3])+d*d*P[:,3:,3:]+q*d**3/3*I
    covariance[:,:3,3:] += d*P[:,3:,3:]+q*d*d/2*I
    covariance[:,3:,:3] = covariance[:,:3,3:].transpose(0,2,1)
    covariance[:,3:,3:] += q*d*I
    return out, covariance


def kalman(x, P, H, z, variance):
    R = np.diag(np.atleast_1d(variance))
    innovation = np.atleast_1d(z)-H@x
    S = H@P@H.T+R
    K = np.linalg.solve(S, H@P).T
    out = x+K@innovation
    M = np.eye(6)-K@H
    covariance = M@P@M.T+K@R@K.T  # Joseph form
    return out, .5*(covariance+covariance.T), innovation


class OwnHistory:
    """Reported-origin position history; derivative avoids assuming SDK v=dp/dt."""
    def __init__(self, settings):
        self.s = settings
        self.samples = deque()
        self._cache = None

    def add(self, packet):
        self._cache = None
        begin, end = float(packet['read_begin_sim_s']), float(packet['read_end_sim_s'])
        p = np.asarray(packet['position_P_m'], float)
        v = np.asarray(packet['velocity_P_m_s'], float)
        q = packet['q_PB_xyzw_reported']
        if not np.isfinite([begin, end]).all() or end < begin:
            raise ValueError('Invalid own-state read bracket')
        if p.shape != (3,) or v.shape != (3,) or not np.isfinite(np.r_[p, v]).all():
            raise ValueError('Invalid own-state vectors')
        B = rotation(q)
        p = p+B.T@np.asarray(self.s['radar_origin_offset_B_m'])
        stamp = .5*(begin+end)
        item = (stamp, p, v, end-begin)
        if self.samples and stamp < self.samples[-1][0]-1e-8:
            raise ValueError('Own clock moved backwards; create a new tracker per native run')
        if self.samples and abs(stamp-self.samples[-1][0]) < 1e-8:
            self.samples[-1] = item
        else:
            self.samples.append(item)
        while len(self.samples) > 2 and self.samples[1][0] < stamp-self.s['history_s']:
            self.samples.popleft()

    def at(self, stamp):
        if not self.samples:
            raise ValueError('own_history_missing')
        if self._cache is None:
            items = list(self.samples)
            self._cache = items, np.array([i[0] for i in items])
        items, t = self._cache
        if stamp < t[0]-1e-8 or stamp > t[-1]+self.s['history_extrapolation_s']:
            raise ValueError('own_history_missing')
        i = min(len(t)-1, max(0, int(np.searchsorted(t, stamp, side='right')-1)))
        if i+1 < len(t):
            gap = t[i+1]-t[i]
            if gap > self.s['history_max_gap_s']:
                raise ValueError('own_history_gap')
            fraction = np.clip((stamp-t[i])/gap, 0., 1.)
            p = items[i][1]*(1-fraction)+items[i+1][1]*fraction
            bracket = max(items[i][3], items[i+1][3])
        else:
            p, bracket = items[i][1].copy(), items[i][3]
        # Use only already received samples. Position fitting is appropriate
        # because reported SDK velocity need not differentiate this origin.
        use = (t >= stamp-self.s['velocity_window_s']) & (t <= stamp+self.s['history_max_gap_s'])
        if np.sum(use) >= 2 and np.ptp(t[use]) > 1e-5:
            pp = np.array([items[j][1] for j in np.flatnonzero(use)])
            design = np.c_[t[use]-stamp, np.ones(np.sum(use))]
            v = np.linalg.lstsq(design, pp, rcond=None)[0][0]
            source = 'reported_position_history'
        else:
            v = items[i][2].copy()
            source = 'SDK_velocity_fallback'
        extrapolation = max(0., stamp-t[-1])
        if extrapolation:
            p = p+v*extrapolation
        return p, v, bracket, source, extrapolation

    def maximum_speed(self):
        if not self.samples:
            return 0.
        recent = list(self.samples)[-20:]
        values = [np.linalg.norm(i[2]) for i in recent]
        for a, b in zip(recent[:-1], recent[1:]):
            if b[0]-a[0] > 1e-5:
                values.append(np.linalg.norm(b[1]-a[1])/(b[0]-a[0]))
        return float(max(values, default=0.))


@dataclass
class Observation:
    key: str
    stamp: float
    raw_stamp: float
    received: float
    center: np.ndarray
    origin: np.ndarray
    ego_velocity: np.ndarray
    direction: np.ndarray
    radial: float
    position_variance: float
    radial_variance: float
    radius: float
    radius_variance: float


@dataclass
class Track:
    id: str
    x: np.ndarray
    P: np.ndarray
    stamp: float
    radius: float
    radius_variance: float
    first_stamp: float
    received: float
    raw_stamp: float
    hits: int = 1
    last_radial: float = 0.
    ambiguity: bool = False
    velocity_initialization: str = 'radial_prior'


class RadarTracker:
    """update(public_rows, receive_sim_s) -> JSON-compatible schema-v1 frame."""
    def __init__(self, settings=None):
        self.settings = load_settings(settings)
        self.history = OwnHistory(self.settings)
        self.tracks = []
        self.unresolved = {}
        # Track association and hazard occupancy have different lifetimes.
        # Retired tracks stop participating in assignment, but their last
        # posterior remains an uncertain moving hazard until this run ends.
        self.retired_tracks = {}
        self.retired_unresolved = {}
        self.occupancy_capacity_lost = False
        self.birth_reinitialization_candidates = set()
        self.birth_reinitializations_total = 0
        self.seen = OrderedDict()
        self.sequence = self.counter = 0
        self.last_time = None
        self.offset_upper = None
        self.maximum_own_speed = 0.
        self.lost_hazard_until = -math.inf

    def add_own_state(self, packet):
        self.history.add(packet)

    def _clock(self, raw, now, counts):
        s = self.settings
        if s['timestamp_offset_s'] is not None:
            offset = s['timestamp_offset_s']
            delta = s['timestamp_offset_uncertainty_s']
            return offset, [offset-delta, offset+delta], 'configured'
        stamps = [z[2, 0] for z in raw if z[2, 0] > 0]
        if stamps:
            upper = float(now-max(stamps))
            if upper < s['auto_offset_min_s']-s['future_tolerance_s']:
                counts['clock_inconsistent'] += 1
            else:
                upper = max(s['auto_offset_min_s'], upper)
                self.offset_upper = upper if self.offset_upper is None else min(self.offset_upper, upper)
        # Use the lower bound as a fixed nominal epoch; shrinking the upper
        # bound never changes filter timestamps or spuriously creates new data.
        lower = s['auto_offset_min_s']
        return lower, [lower, self.offset_upper] if self.offset_upper is not None else None, 'bounded_not_synchronized'

    def _remember(self, key, now):
        self.seen[key] = now
        self.seen.move_to_end(key)
        while len(self.seen) > self.settings['max_cached_measurements']:
            self.seen.popitem(last=False)

    def _observation(self, z, now, offset):
        s = self.settings
        stamp = float(z[2, 0]+offset)
        if z[2, 0] <= 0:
            raise ValueError('initial_unstamped')
        if stamp > now+1e-8:
            raise ValueError('future_measurement')
        if now-stamp > s['max_observation_age_s']:
            raise ValueError('stale_measurement')
        origin, ego_v, span, source, extrapolation = self.history.at(stamp)
        relative = z[0]
        distance = float(np.linalg.norm(relative))
        closest, radial, diameter = z[1]
        if distance < 1e-6 or diameter <= 0:
            raise ValueError('invalid_geometry')
        direction = relative/distance
        position_var = s['position_sigma_m']**2+s['ego_position_sigma_m']**2
        position_var += ((span/2+extrapolation)*max(np.linalg.norm(ego_v), .1))**2
        radial_var = s['radial_velocity_sigma_m_s']**2+s['ego_velocity_sigma_m_s']**2
        if source == 'SDK_velocity_fallback':
            radial_var += s['initial_velocity_std_m_s']**2
        # Diameter and closest-point proxy are independent noisy public fields.
        rd = diameter/2
        vd = max((rd*s['diameter_sigma_fraction'])**2, 1e-6)
        rc = max(0., distance-closest)
        vc = 2*s['position_sigma_m']**2
        radius_var = 1/(1/vd+1/vc)
        radius = max(.01, radius_var*(rd/vd+rc/vc))
        key = hashlib.sha256(np.asarray(z, dtype='<f8').tobytes()).hexdigest()
        return Observation(key, stamp, float(z[2, 0]), now, origin+relative,
            origin, ego_v, direction, float(radial), position_var, radial_var,
            float(radius), float(radius_var))

    def _costs(self, observations):
        s = self.settings
        N, M = len(self.tracks), len(observations)
        cost = np.full((N, M), 1e9)
        self.birth_reinitialization_candidates=set()
        if not N or not M:
            return cost
        stamps = np.array([o.stamp for o in observations])
        centers = np.array([o.center for o in observations])
        directions = np.array([o.direction for o in observations])
        radial_absolute = np.array([o.radial+o.direction@o.ego_velocity for o in observations])
        rpos = np.array([o.position_variance for o in observations])
        rv = np.array([o.radial_variance for o in observations])
        radii = np.array([o.radius for o in observations])
        radius_vars = np.array([o.radius_variance for o in observations])
        q = s['acceleration_spectral_density_m2_s3']
        x = np.array([tr.x for tr in self.tracks])
        P = np.array([tr.P for tr in self.tracks])
        dt = stamps[None,:]-np.array([tr.stamp for tr in self.tracks])[:,None]
        d = np.maximum(dt,0.)
        innovation = centers[None,:,:]-(x[:,None,:3]+d[:,:,None]*x[:,None,3:])
        trace = (np.trace(P[:,:3,:3],axis1=1,axis2=2)[:,None]+
            2*d*np.trace(P[:,:3,3:],axis1=1,axis2=2)[:,None]+
            d*d*np.trace(P[:,3:,3:],axis1=1,axis2=2)[:,None]+q*d**3+3*rpos[None,:])
        # Cheap necessary condition before solving only nearby 3x3 systems.
        ti,oi = np.nonzero((dt>=-1e-8)&
            (np.sum(innovation**2,axis=2)<=s['association_cost_gate']*trace))
        if not len(ti):
            return cost
        du=d[ti,oi];PP=P[ti]
        covariance = (PP[:,:3,:3]+du[:,None,None]*(PP[:,:3,3:]+PP[:,3:,:3])+
            du[:,None,None]**2*PP[:,3:,3:]+
            (q*du**3/3+rpos[oi])[:,None,None]*np.eye(3))
        y=innovation[ti,oi]
        pos_cost=np.einsum('ni,ni->n',y,np.linalg.solve(covariance,y[...,None])[...,0])
        los=directions[oi];velocity=x[ti,3:]
        variance=np.einsum('ni,nij,nj->n',los,PP[:,3:,3:],los)+q*du+rv[oi]
        ego=np.array([o.ego_velocity for o in observations])[oi]
        relative=velocity-ego
        transverse=relative-los*np.sum(relative*los,axis=1)[:,None]
        ranges=np.array([max(np.linalg.norm(o.center-o.origin),1.) for o in observations])[oi]
        transverse_variance = np.maximum(0.,np.trace(PP[:,3:,3:],axis1=1,axis2=2)-
            np.einsum('ni,nij,nj->n',los,PP[:,3:,3:],los))
        variance+=(np.sum(transverse**2,axis=1)+transverse_variance)*s['position_sigma_m']**2/ranges**2
        velocity_cost=(radial_absolute[oi]-np.sum(los*velocity,axis=1))**2/variance
        tr_radius=np.array([tr.radius for tr in self.tracks])[ti]
        size_var=np.array([tr.radius_variance for tr in self.tracks])[ti]+radius_vars[oi]+(.2*np.maximum(tr_radius,.1))**2
        size_cost=.25*(radii[oi]-tr_radius)**2/size_var
        joint=pos_cost+velocity_cost+size_cost
        if s['birth_initialization']=='two_position':
            # A one-hit radial prior is not an identified transverse velocity.
            # If its later radial gate fails, permit geometric association for
            # two-position initialization; the ordinary ambiguity gate still
            # retains alternatives rather than forcing an identity match.
            bootstrap=(np.array([self.tracks[i].hits==1 for i in ti]) &
                (du>=s['confirmation_span_s']) & (joint>=s['association_cost_gate']) &
                (pos_cost+size_cost<s['association_cost_gate']))
            for i,j in zip(ti[bootstrap],oi[bootstrap]):
                self.birth_reinitialization_candidates.add((self.tracks[i].id,observations[j].key))
            joint=np.where(bootstrap,pos_cost+size_cost,joint)
        cost[ti,oi]=joint
        return cost

    def _new_track(self, observation):
        s = self.settings
        d = observation.direction
        # Unknown absolute tangential motion has zero mean and broad covariance.
        speed = observation.radial+d@observation.ego_velocity
        velocity = speed*d
        distance=max(np.linalg.norm(observation.center-observation.origin),1.)
        los_variance=observation.radial_variance+(
            2*s['initial_velocity_std_m_s']**2*s['position_sigma_m']**2/distance**2)
        Pv = (s['initial_velocity_std_m_s']**2*(np.eye(3)-np.outer(d, d))+
              los_variance*np.outer(d, d))
        P = np.zeros((6, 6))
        P[:3, :3] = np.eye(3)*observation.position_variance
        P[3:, 3:] = Pv
        self.counter += 1
        return Track('radar_%06d' % self.counter, np.r_[observation.center, velocity],
            P, observation.stamp, observation.radius, observation.radius_variance,
            observation.stamp, observation.received, observation.raw_stamp,
            last_radial=observation.radial)

    def _update_track(self, tr, o):
        s = self.settings
        dt = max(0., o.stamp-tr.stamp)
        if (tr.id,o.key) in self.birth_reinitialization_candidates:
            # Diffuse-velocity two-position initialization. Position errors
            # imply NONZERO position/velocity cross covariance R_new/dt.
            # q*dt/3 covers end velocity versus interval-average velocity under
            # the same white-acceleration model. Only the latest radial rate
            # updates this reinitialized posterior; no mature track is reset.
            R=np.eye(3)*o.position_variance
            x=np.r_[o.center,(o.center-tr.x[:3])/dt]
            P=np.block([[R,R/dt],[R/dt,(tr.P[:3,:3]+R)/dt**2+
                s['acceleration_spectral_density_m2_s3']*dt/3*np.eye(3)]])
            innovation=o.center-(tr.x[:3]+dt*tr.x[3:])
            tr.velocity_initialization='two_position'
            self.birth_reinitializations_total+=1
        else:
            x, P = cv_predict(tr.x, tr.P, dt, s['acceleration_spectral_density_m2_s3'])
            H = np.c_[np.eye(3), np.zeros((3, 3))]
            x, P, innovation = kalman(x, P, H, o.center, np.full(3, o.position_variance))
        Hr = np.r_[np.zeros(3), o.direction][None, :]
        relative_v = x[3:]-o.ego_velocity
        transverse = relative_v-o.direction*(o.direction@relative_v)
        distance = max(np.linalg.norm(o.center-o.origin), 1.)
        tangent_variance=max(0.,float(np.trace(P[3:,3:])-o.direction@P[3:,3:]@o.direction))
        variance = o.radial_variance+(np.linalg.norm(transverse)**2+tangent_variance)*s['position_sigma_m']**2/distance**2
        x, P, _ = kalman(x, P, Hr, [o.radial+o.direction@o.ego_velocity], [variance])
        size_var = tr.radius_variance+1e-4*dt
        size_gain = size_var/(size_var+o.radius_variance)
        tr.radius += size_gain*(o.radius-tr.radius)
        tr.radius_variance = (1-size_gain)*size_var
        tr.x, tr.P, tr.stamp = x, P, o.stamp
        tr.received, tr.raw_stamp = o.received, o.raw_stamp
        tr.hits += 1
        tr.last_radial, tr.ambiguity = o.radial, False
        return float(np.linalg.norm(innovation))

    def update(self, rows, receive_sim_s):
        started = time.perf_counter()
        previous_birth_reinitializations=self.birth_reinitializations_total
        now = float(receive_sim_s)
        if not math.isfinite(now):
            raise ValueError('Invalid receive_sim_s')
        if self.last_time is not None and now < self.last_time-1e-8:
            raise ValueError('Simulation clock moved backwards; start a new tracker')
        if self.history.samples and self.history.samples[-1][0] > now+1e-8:
            raise ValueError('Own-state history contains future observations')
        if len(rows) > 500:
            raise ValueError('Public radar packet exceeds the 500-row interface contract')
        s = self.settings
        counts = dict(received=len(rows), fresh=0, duplicates=0, malformed=0,
            initial_unstamped=0, future_measurement=0, stale_measurement=0,
            own_history_missing=0, own_history_gap=0, invalid_geometry=0,
            clock_inconsistent=0, associated=0, ambiguous=0, same_epoch_alternative=0, new_tracks=0,
            capacity_rejected=0, expired_tracks=0, expired_unresolved_observations=0)
        raw = []
        for row in rows:
            try:
                z = np.asarray(row, float)
                if z.shape != (3, 3) or not np.isfinite(z).all():
                    raise ValueError()
                raw.append(z)
            except (TypeError, ValueError):
                counts['malformed'] += 1
        offset, interval, clock_status = self._clock(raw, now, counts)
        observations = []
        for z in raw:
            key = hashlib.sha256(np.asarray(z, dtype='<f8').tobytes()).hexdigest()
            if key in self.seen:
                counts['duplicates'] += 1
                self._remember(key, now)
                continue
            try:
                o = self._observation(z, now, offset)
            except ValueError as exc:
                reason = str(exc)
                counts[reason if reason in counts else 'malformed'] += 1
                continue
            observations.append(o)
            self._remember(key, now)
        counts['fresh'] = len(observations)
        kept = []
        for tr in self.tracks:
            if now-tr.stamp <= s['track_timeout_s']:
                kept.append(tr)
            else:
                counts['expired_tracks'] += 1
                if s['track_expiry_mode']=='retain_occupancy':
                    self.retired_tracks[tr.id]=tr
        self.tracks = kept
        old_unresolved={k:o for k,o in self.unresolved.items() if now-o.stamp>s['max_observation_age_s']}
        counts['expired_unresolved_observations']=len(old_unresolved)
        if s['track_expiry_mode']=='retain_occupancy':
            self.retired_unresolved.update(old_unresolved)
        self.unresolved = {k:o for k,o in self.unresolved.items() if now-o.stamp <= s['max_observation_age_s']}
        cost = self._costs(observations)
        assigned = {}
        if cost.size:
            dummy = np.full((len(observations), len(observations)), s['association_cost_gate'])
            rr, cc = linear_sum_assignment(np.r_[cost, dummy])
            assigned = {int(c):int(r) for r,c in zip(rr,cc) if r < len(self.tracks) and cost[r,c] < s['association_cost_gate']}
        innovations = []
        for j, o in enumerate(observations):
            valid = np.sort(cost[:, j][cost[:, j] < s['association_cost_gate']]) if len(self.tracks) else np.array([])
            ambiguous = len(valid)>1 and valid[1]-valid[0] < s['association_ambiguity_gap']
            i = assigned.get(j)
            if i is not None and len(valid) and cost[i,j]-valid[0] > s['association_ambiguity_gap']:
                ambiguous = True
            if ambiguous or (i is None and len(valid)):
                counts['ambiguous'] += 1
                self.unresolved[o.key] = o
                for k in np.flatnonzero(cost[:,j] < s['association_cost_gate']):
                    self.tracks[k].ambiguity = True
            elif i is not None:
                if o.stamp <= self.tracks[i].stamp+1e-8:
                    # A changed payload at the same epoch may be a different
                    # object, but cannot be independent evidence for this track.
                    self.unresolved[o.key] = o
                    self.tracks[i].ambiguity = True
                    counts['same_epoch_alternative'] += 1
                else:
                    innovations.append(self._update_track(self.tracks[i], o))
                    counts['associated'] += 1
            elif len(self.tracks)+len(self.retired_tracks) < s['max_tracks']:
                self.tracks.append(self._new_track(o))
                counts['new_tracks'] += 1
            else:
                counts['capacity_rejected'] += 1
                self.unresolved[o.key] = o
        self.maximum_own_speed = max(self.maximum_own_speed, self.history.maximum_speed())
        while len(self.unresolved)+len(self.retired_unresolved) > s['max_unresolved_observations']:
            combined=dict(self.retired_unresolved,**self.unresolved)
            oldest=min(combined,key=lambda k:combined[k].stamp)
            if oldest in self.unresolved:del self.unresolved[oldest]
            else:del self.retired_unresolved[oldest]
            counts['capacity_rejected'] += 1
            self.lost_hazard_until = now+s['max_observation_age_s']
            if s['track_expiry_mode']=='retain_occupancy':self.occupancy_capacity_lost=True
        counts['retained_expired_tracks']=len(self.retired_tracks)
        counts['retained_expired_unresolved_observations']=len(self.retired_unresolved)
        counts['birth_reinitializations']=self.birth_reinitializations_total-previous_birth_reinitializations
        counts['birth_reinitializations_total']=self.birth_reinitializations_total
        delta = max(abs(x-offset) for x in interval) if interval is not None else None
        counts['mean_position_innovation_m'] = float(np.mean(innovations)) if innovations else None
        counts['processing_wall_ms'] = 0.
        self.sequence += 1
        self.last_time = now
        frame = self._frame(now, offset, interval, delta, clock_status, counts)
        counts['processing_wall_ms'] = (time.perf_counter()-started)*1000
        return frame

    def _object(self, tr, now, offset, interval, delta, kind='track', prediction=None):
        s = self.settings
        x, P = prediction if prediction is not None else cv_predict(
            tr.x, tr.P, now-tr.stamp, s['acceleration_spectral_density_m2_s3'])
        confirmed = tr.hits >= s['confirmation_hits'] and tr.stamp-tr.first_stamp >= s['confirmation_span_s']
        status = 'confirmed' if confirmed else 'tentative'
        if now-tr.stamp > s['max_observation_age_s']:
            status = 'stale'
        elif now-tr.stamp > s['poll_period_s']*2 and confirmed:
            status = 'coasting'
        clock_pad = ((s['clock_obstacle_speed_bound_m_s']+self.maximum_own_speed)*delta) if delta is not None else 0.
        time_bounds = [tr.raw_stamp+interval[0], min(tr.received,tr.raw_stamp+interval[1])] if interval is not None else None
        age_bounds = [max(0.,now-time_bounds[1]), max(0.,now-time_bounds[0])] if time_bounds is not None else None
        return dict(id=tr.id, kind=kind, sources=['radar'], status=status,
            state_epoch_sim_s=now, center_P_m=x[:3].tolist(), velocity_P_m_s=x[3:].tolist(),
            covariance_P=P.tolist(), radius_m=float(tr.radius), radius_sigma_m=float(np.sqrt(tr.radius_variance)),
            extent_kind='radar_sphere_proxy', bounding_geometry_verified=False,
            extra_position_margin_m=float(clock_pad),
            last_measurement_sim_s=tr.stamp, measurement_time_bounds_sim_s=time_bounds,
            measurement_age_s=float(now-tr.stamp), measurement_age_bounds_s=age_bounds,
            last_received_sim_s=tr.received,
            independent_observations=tr.hits, association_ambiguous=bool(tr.ambiguity),
            velocity_initialization=tr.velocity_initialization,
            last_radial_relative_m_s=tr.last_radial)

    def _frame(self, now, offset, interval, delta, clock_status, counts):
        objects = []
        tracks=self.tracks+list(self.retired_tracks.values())
        if tracks:
            x,P = cv_predict_many(np.array([tr.x for tr in tracks]),
                np.array([tr.P for tr in tracks]),np.array([now-tr.stamp for tr in tracks]),
                self.settings['acceleration_spectral_density_m2_s3'])
            objects = [self._object(tr,now,offset,interval,delta,'track',(x[i],P[i]))
                       for i,tr in enumerate(tracks)]
            for obj in objects[len(self.tracks):]:
                obj.update(status='lost',occupancy_retained=True,association_active=False)
        unresolved=list(self.unresolved.items())+list(self.retired_unresolved.items())
        for key, observation in unresolved:
            tr = self._new_track(observation)
            # Unresolved observations are alternatives, not additional confirmed objects.
            self.counter -= 1
            tr.id = 'unresolved_'+key[:16]
            tr.ambiguity = True
            obj = self._object(tr, now, offset, interval, delta, 'unassociated_observation')
            obj['status'] = 'unresolved'
            if key in self.retired_unresolved:
                obj.update(status='lost_unresolved',occupancy_retained=True,association_active=False)
            objects.append(obj)
        conditions = [k for k in ('malformed', 'initial_unstamped', 'invalid_geometry', 'future_measurement', 'stale_measurement',
            'own_history_missing', 'own_history_gap', 'clock_inconsistent',
            'capacity_rejected') if counts[k]]
        if counts['expired_tracks'] and self.settings['track_expiry_mode']=='drop_unassessed':
            conditions.append('expired_tracks')
        if not objects:
            conditions.append('no_assessed_objects')
        if not counts['received']:
            conditions.append('empty_radar_packet')
        if interval is None:
            conditions.append('clock_unregistered')
        if now <= self.lost_hazard_until:
            conditions.append('unresolved_capacity_loss')
        if self.occupancy_capacity_lost:
            conditions.append('retained_occupancy_capacity_loss')
        own = None
        try:
            p, v, span, source, extra = self.history.at(now)
            own = dict(time_sim_s=now, radar_origin_P_m=p.tolist(), velocity_P_m_s=v.tolist(),
                       velocity_source=source, pose_read_span_s=span, extrapolated_s=extra)
        except ValueError:
            conditions.append('current_own_pose_unavailable')
        return dict(schema_version=SCHEMA_VERSION, producer='obstacle_characterization',
            producer_version=VERSION, sequence=self.sequence, frame=FRAME, units='SI',
            generated_sim_s=now, state_order=['x','y','z','vx','vy','vz'],
            motion_model='constant_velocity_with_white_acceleration',
            acceleration_spectral_density_m2_s3=self.settings['acceleration_spectral_density_m2_s3'],
            default_query=dict(uncertainty_multiplier=self.settings['uncertainty_multiplier'],
                shape_padding_m=self.settings['shape_padding_m'], forecast_horizon_s=self.settings['forecast_horizon_s']),
            clock=dict(mode=clock_status, nominal_offset_s=offset, offset_bounds_s=interval,
                nominal_to_bound_s=delta, exact=bool(delta == 0 if delta is not None else False)),
            own_state=own, objects=objects,
            coverage=dict(scope='public_nearest_observations_and_retained_tracks', guarantees_free_space=False,
                returned_observations=counts['received'], tracked_objects=len(self.tracks),
                unresolved_observations=len(self.unresolved),
                track_expiry_mode=self.settings['track_expiry_mode'],
                retained_expired_tracks=len(self.retired_tracks),
                retained_expired_unresolved_observations=len(self.retired_unresolved),
                conditions=conditions),
            diagnostics=counts)
