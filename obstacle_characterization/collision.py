"""Shared statistical occupancy forecasts and swept-sphere trajectory queries.

These queries assess supplied trajectories, never choose or command a path.
They do not certify physical safety or infer free space from missing detections.
"""
from __future__ import annotations
import math
import numpy as np
from .radar import SCHEMA_VERSION, FRAME


def _inputs(frame, times):
    if frame.get('schema_version') != SCHEMA_VERSION or frame.get('frame') != FRAME or frame.get('units') != 'SI':
        raise ValueError('Expected obstacle-characterization schema v1, Phobos frame, SI units')
    epoch = float(frame['generated_sim_s'])
    t = np.asarray(times, float)
    if t.ndim != 1 or len(t) == 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Prediction times must be finite and strictly increasing')
    if not math.isfinite(epoch) or t[0] < epoch-1e-8:
        raise ValueError('Cannot predict before the frame epoch')
    return epoch, t


def _all_arrays(objects, epoch):
    if not objects:
        return np.empty((0,6)),np.empty((0,6,6))
    x=np.concatenate((np.asarray([o['center_P_m'] for o in objects],float),
        np.asarray([o['velocity_P_m_s'] for o in objects],float)),axis=1)
    P=np.asarray([o['covariance_P'] for o in objects],float)
    if x.shape!=(len(objects),6) or P.shape!=(len(objects),6,6) or not np.isfinite(x).all() or not np.isfinite(P).all():
        raise ValueError('Invalid object state/covariance')
    if not np.allclose(P,P.transpose(0,2,1),atol=1e-8) or np.linalg.eigvalsh(P).min() < -1e-7:
        raise ValueError('Object covariance is not positive semidefinite')
    if any(not math.isfinite(o['state_epoch_sim_s']) or abs(o['state_epoch_sim_s']-epoch)>1e-7 for o in objects):
        raise ValueError('Mixed object epochs in frame')
    geometry=np.array([[o[k] for k in ('radius_m','radius_sigma_m','extra_position_margin_m')] for o in objects])
    if not np.isfinite(geometry).all() or geometry.min()<0:
        raise ValueError('Invalid object geometry/margin')
    return x,P


def _options(frame, uncertainty_multiplier=None, shape_padding_m=None):
    defaults = frame['default_query']
    k = defaults['uncertainty_multiplier'] if uncertainty_multiplier is None else uncertainty_multiplier
    padding = defaults['shape_padding_m'] if shape_padding_m is None else shape_padding_m
    if not np.isfinite([k, padding]).all() or min(k, padding) < 0:
        raise ValueError('Invalid uncertainty multiplier or shape padding')
    horizon=float(defaults['forecast_horizon_s'])
    if not math.isfinite(horizon) or horizon<=0:
        raise ValueError('Invalid forecast horizon')
    return float(k), float(padding), horizon


def _sigma_many(P, dt, q):
    # Monotone spectral upper bound covers all intermediate times, including
    # covariance with negative position/velocity cross terms.
    a=np.maximum(0.,np.linalg.eigvalsh(P[:,:3,:3]).max(axis=1))[:,None]
    b=np.linalg.norm(P[:,:3,3:],ord=2,axis=(1,2))[:,None]
    c=np.maximum(0.,np.linalg.eigvalsh(P[:,3:,3:]).max(axis=1))[:,None]
    t=dt[None,:]
    return np.sqrt(np.maximum(0.,a+2*t*b+t*t*c+q*t**3/3))


def position_support_sigma(covariance, normal):
    """One-sigma support of a position ellipsoid along a UNIT normal.

    Batch dimensions follow NumPy broadcasting. No covariance or confidence
    multiplier is reduced; this retains the ellipsoid's directional geometry.
    """
    P=np.asarray(covariance,float);n=np.asarray(normal,float)
    if P.shape[-2:]!=(3,3) or n.shape[-1:]!=(3,) or not np.isfinite(P).all() or not np.isfinite(n).all():
        raise ValueError('Expected finite position covariance and unit normal')
    if not np.allclose(np.linalg.norm(n,axis=-1),1.,atol=1e-7):raise ValueError('Support normal must have unit length')
    return np.sqrt(np.maximum(0.,np.einsum('...i,...ij,...j->...',n,P,n)))


def ellipsoid_separating_normal(relative, covariance, multiplier=3.):
    """Normal toward a point from its closest covariance-ellipsoid point.

    Uses the scalar Lagrange-multiplier equation in the eigenbasis. A tiny
    regularization chooses a stable normal for singular ellipsoids; callers
    always evaluate support against the ORIGINAL covariance. Any resulting
    positive separating-plane margin is therefore conservative for that set.
    """
    r=np.asarray(relative,float);P=np.asarray(covariance,float)
    if P.shape[-2:]!=(3,3) or r.shape[-1:]!=(3,) or not np.isfinite(P).all() or not np.isfinite(r).all() or not np.isfinite(multiplier) or multiplier<0:
        raise ValueError('Invalid ellipsoid normal inputs')
    values,Q=np.linalg.eigh(P);axes2=multiplier**2*np.maximum(values,0.)
    if np.any(values < -1e-7):raise ValueError('Position covariance must be positive semidefinite')
    regular=axes2+1e-12*np.maximum(np.max(axes2,axis=-1,keepdims=True),1.)
    local=np.einsum('...ji,...j->...i',Q,r)
    outside=np.sum(local**2/regular,axis=-1)>1.
    low=np.zeros(outside.shape);high=np.sqrt(np.sum(regular*local**2,axis=-1))
    for _ in range(40):
        mid=.5*(low+high)
        value=np.sum(regular*local**2/(regular+mid[...,None])**2,axis=-1)
        low=np.where(value>1.,mid,low);high=np.where(value>1.,high,mid)
    tau=.5*(low+high)
    direction=np.where(outside[...,None],local*tau[...,None]/(regular+tau[...,None]),local/regular)
    n=np.einsum('...ij,...j->...i',Q,direction);norm=np.linalg.norm(n,axis=-1,keepdims=True)
    fallback=np.zeros_like(n);fallback[...,0]=1.
    return np.divide(n,norm,out=fallback,where=norm>1e-15)


def _sigma_along_segments(P, start_dt, end_dt, normals, q):
    # For fixed n, n'P_position(t)n is a convex cubic for t>=0:
    # a+2bt+ct^2+q*t^3/3, with c=n'P_vv*n>=0. Its maximum on a
    # segment is at an endpoint, including negative p-v cross covariance.
    a=np.einsum('mki,mij,mkj->mk',normals,P[:,:3,:3],normals)
    b=np.einsum('mki,mij,mkj->mk',normals,P[:,:3,3:],normals)
    c=np.einsum('mki,mij,mkj->mk',normals,P[:,3:,3:],normals)
    va=a+2*b*start_dt+c*start_dt**2+q*start_dt**3/3
    vb=a+2*b*end_dt+c*end_dt**2+q*end_dt**3/3
    return np.sqrt(np.maximum(0.,np.maximum(va,vb)))


def predict_objects(frame, times_sim_s, *, uncertainty_multiplier=None, shape_padding_m=None, array_output=False):
    """Forecast all objects, including tentative/unassociated hazards.

    array_output=True keeps numeric arrays for solvers; default uses JSON lists.
    Output inflated radii include object extent + shape/clock/statistical
    allowances. Add vehicle extent and the desired extra clearance separately.
    """
    epoch, t = _inputs(frame, times_sim_s)
    k, padding, horizon = _options(frame, uncertainty_multiplier, shape_padding_m)
    if t[-1]-epoch > horizon+1e-8:
        raise ValueError('Requested times exceed configured forecast_horizon_s')
    q = float(frame['acceleration_spectral_density_m2_s3'])
    if not math.isfinite(q) or q < 0:
        raise ValueError('Invalid acceleration spectral density')
    objects = []
    x,P=_all_arrays(frame['objects'],epoch)
    if len(x):
        dt=np.maximum(0.,t-epoch)
        centers=x[:,None,:3]+dt[None,:,None]*x[:,None,3:]
        d=dt[None,:,None,None]
        covariance=(P[:,None,:3,:3]+d*(P[:,None,:3,3:]+P[:,None,3:,:3])+
            d*d*P[:,None,3:,3:]+q*d**3/3*np.eye(3)[None,None,:,:])
        base=np.array([o['radius_m']+k*o['radius_sigma_m']+padding+o['extra_position_margin_m'] for o in frame['objects']])
        radius=base[:,None]+k*_sigma_many(P,dt,q)
        convert=(lambda a:a) if array_output else (lambda a:a.tolist())
        for i,obj in enumerate(frame['objects']):
            objects.append(dict(id=obj['id'],kind=obj['kind'],status=obj['status'],
                center_P_m=convert(centers[i]),position_covariance_P=convert(covariance[i]),
                extent_radius_with_margins_m=float(base[i]),
                inflated_obstacle_radius_m=convert(radius[i])))
    return dict(schema_version=SCHEMA_VERSION, frame=FRAME, units='SI',
        times_sim_s=t.tolist(), objects=objects, coverage=frame['coverage'],
        numeric_representation='numpy' if array_output else 'json',
        statistical_inflation_only=True, certified_safe=False)


def check_trajectory(frame, times_sim_s, positions_P_m, *, vehicle_radius_m,
                     clearance_m=0., uncertainty_multiplier=None, shape_padding_m=None,
                     vehicle_acceleration_bound_m_s2=None, uncertainty_geometry='sphere'):
    """Check a piecewise-linear center path against moving uncertainty tubes.

    Closest relative approach is checked continuously within every segment.
    Optional vehicle acceleration bound adds a_max*segment_dt**2/8 to cover
    deviation of a bounded-acceleration physical trajectory from its chord.
    Obstacle uncertainty remains statistical and geometry remains a radar proxy.
    sphere preserves the original largest-axis enclosing-sphere calculation.
    ellipsoid uses a fixed separating plane for EACH whole swept segment and
    the maximum covariance support on that segment. A positive margin excludes
    the declared ellipsoid plus sphere for the whole chord, not just its knots.
    """
    epoch, t = _inputs(frame, times_sim_s)
    if uncertainty_geometry not in ('sphere','ellipsoid'):raise ValueError('Unknown uncertainty geometry')
    positions = np.asarray(positions_P_m, float)
    if len(t) < 2 or positions.shape != (len(t), 3) or not np.isfinite(positions).all():
        raise ValueError('Expected at least two finite 3D trajectory centers')
    if not np.isfinite([vehicle_radius_m, clearance_m]).all() or vehicle_radius_m <= 0 or clearance_m < 0:
        raise ValueError('Need a positive enclosing vehicle radius and nonnegative clearance')
    if vehicle_acceleration_bound_m_s2 is not None and (
            not math.isfinite(vehicle_acceleration_bound_m_s2) or vehicle_acceleration_bound_m_s2 < 0):
        raise ValueError('Invalid vehicle acceleration bound')
    k, shape, horizon = _options(frame, uncertainty_multiplier, shape_padding_m)
    conditions = list(frame['coverage']['conditions'])
    if t[-1]-epoch > horizon+1e-8:
        conditions.append('forecast_horizon_exceeded')
        return dict(schema_version=SCHEMA_VERSION, assessment='unassessed',
            minimum_inflated_clearance_m=None, critical_object_id=None,
            objects=[], conditions=list(dict.fromkeys(conditions)), certified_safe=False)
    q = float(frame['acceleration_spectral_density_m2_s3'])
    if not math.isfinite(q) or q < 0:
        raise ValueError('Invalid acceleration spectral density')
    durations = np.diff(t)
    start_dt, end_dt = np.maximum(0., t[:-1]-epoch), np.maximum(0., t[1:]-epoch)
    chord_padding = np.zeros(len(durations)) if vehicle_acceleration_bound_m_s2 is None else (
        vehicle_acceleration_bound_m_s2*durations**2/8)
    results = []
    x,P=_all_arrays(frame['objects'],epoch)
    if len(x):
        r0=positions[None,:-1,:]-(x[:,None,:3]+start_dt[None,:,None]*x[:,None,3:])
        dr=np.diff(positions,axis=0)[None,:,:]-durations[None,:,None]*x[:,None,3:]
        denom=np.sum(dr*dr,axis=2)
        fraction=np.clip(np.divide(-np.sum(r0*dr,axis=2),denom,
            out=np.zeros_like(denom),where=denom>1e-15),0.,1.)
        distances=np.linalg.norm(r0+fraction[:,:,None]*dr,axis=2)
        base=np.array([vehicle_radius_m+clearance_m+o['radius_m']+
            k*o['radius_sigma_m']+shape+o['extra_position_margin_m'] for o in frame['objects']])
        if uncertainty_geometry=='ellipsoid':
            closest=r0+fraction[:,:,None]*dr
            normals=np.divide(closest,distances[:,:,None],out=np.zeros_like(closest),where=distances[:,:,None]>1e-12)
            normals[:,:,0]=np.where(distances>1e-12,normals[:,:,0],1.)
            sigma=_sigma_along_segments(P,start_dt,end_dt,normals,q)
        else:sigma=_sigma_many(P,end_dt,q)
        required=base[:,None]+k*sigma+chord_padding[None,:]
        clearances=distances-required
        separation=distances.copy()
        if uncertainty_geometry=='ellipsoid':
            # A mean-radial normal can still be very conservative for a tilted
            # ellipsoid. Try its closest-point normal where that plane failed.
            ii,jj=np.nonzero(clearances<=0.)
            if len(ii):
                PP=P[ii];ta=start_dt[jj];tb=end_dt[jj];a=ta[:,None,None];b=tb[:,None,None]
                cross=PP[:,:3,3:]+PP[:,3:,:3]
                Ca=PP[:,:3,:3]+a*cross+a*a*PP[:,3:,3:]+q*a**3/3*np.eye(3)
                Cb=PP[:,:3,:3]+b*cross+b*b*PP[:,3:,3:]+q*b**3/3*np.eye(3)
                n=ellipsoid_separating_normal(closest[ii,jj],Cb,k)
                support=np.maximum(position_support_sigma(Ca,n),position_support_sigma(Cb,n))
                projection=np.minimum(np.sum(n*r0[ii,jj],axis=1),np.sum(n*(r0[ii,jj]+dr[ii,jj]),axis=1))
                need=base[ii]+k*support+chord_padding[jj];margin=projection-need
                better=margin>clearances[ii,jj];ii=ii[better];jj=jj[better]
                clearances[ii,jj]=margin[better];required[ii,jj]=need[better];separation[ii,jj]=projection[better]
                normals[ii,jj]=n[better]
        selected=np.argmin(clearances,axis=1)
        for i,obj in enumerate(frame['objects']):
            j=int(selected[i])
            results.append(dict(id=obj['id'],kind=obj['kind'],status=obj['status'],
                minimum_inflated_clearance_m=float(clearances[i,j]),
                minimum_nominal_center_distance_m=float(distances[i].min()),
                critical_time_sim_s=float(t[j]+fraction[i,j]*durations[j]),
                critical_segment_sim_s=[float(t[j]),float(t[j+1])],
                required_center_separation_m=float(required[i,j]),
                separating_projection_m=float(separation[i,j]),
                separating_normal_P=normals[i,j].tolist() if uncertainty_geometry=='ellipsoid' else None))
    results.sort(key=lambda o:o['minimum_inflated_clearance_m'])
    minimum = results[0]['minimum_inflated_clearance_m'] if results else None
    if minimum is not None and minimum <= 0:
        assessment = 'potential_collision'
    elif not results or conditions:
        assessment = 'unassessed'
    else:
        assessment = 'no_predicted_intersection'
    return dict(schema_version=SCHEMA_VERSION, frame=FRAME, units='SI',
        assessment=assessment, minimum_inflated_clearance_m=minimum,
        critical_object_id=results[0]['id'] if results else None,
        objects=results, conditions=list(dict.fromkeys(conditions)),
        trajectory_model='piecewise_linear_centers',
        uncertainty_geometry=uncertainty_geometry,
        bounded_acceleration_chord_padding=vehicle_acceleration_bound_m_s2 is not None,
        all_objects_included=True, continuous_segment_check=True, certified_safe=False)


def rank_threats(frame, *, vehicle_radius_m, clearance_m=0., horizon_s=None):
    """Constant-own-velocity SCREENING query; not an avoidance maneuver."""
    own = frame.get('own_state')
    if own is None:
        return dict(schema_version=SCHEMA_VERSION, assessment='unassessed',
                    objects=[], conditions=['current_own_pose_unavailable'], certified_safe=False)
    maximum = frame['default_query']['forecast_horizon_s']
    horizon = maximum if horizon_s is None else float(horizon_s)
    if not math.isfinite(horizon) or not 0 < horizon <= maximum:
        raise ValueError('Invalid screening horizon')
    epoch = frame['generated_sim_s']
    p, v = np.asarray(own['radar_origin_P_m']), np.asarray(own['velocity_P_m_s'])
    result = check_trajectory(frame, [epoch, epoch+horizon], [p, p+v*horizon],
        vehicle_radius_m=vehicle_radius_m, clearance_m=clearance_m)
    result['screening_assumption'] = 'own radar-origin constant velocity; no proposed avoidance command'
    return result
