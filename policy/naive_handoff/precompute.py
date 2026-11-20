"""Offline constrained quadratic attitude setup and nonlinear model replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.naive_handoff.control import (Policy, closest_point, direction_rotation,
    load_settings, matrix, sha256)
from policy.naive_handoff.accounting import ARRIVAL_RADIUS_M, NOMINAL_FUEL_KG, model_fuel_rate


def riccati(dt, count, weights):
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5*dt*dt], [dt]])
    Q = dt*np.diag([weights["angle"], weights["rate"]])
    R = dt*weights["effort"]
    P = np.diag([weights["terminal_angle"], weights["terminal_rate"]])
    gains = []
    for _ in range(count):
        K = np.linalg.solve(np.array([[R]])+B.T@P@B, B.T@P@A)
        P = Q + A.T@P@(A-B@K)
        P = 0.5*(P+P.T)
        gains.append(K[0])
    return np.asarray(gains[::-1])


def setup_problem(model, initial, target, settings):
    """Fixed-axis, bounded linear attitude model; actual fixed-axis fuel slopes.

    This is optimal only for the declared convex reduced problem. Full nonlinear
    replay is separately reported and no global mission optimum is claimed.
    """
    from scipy.optimize import lsq_linear
    from scipy.linalg import solve_triangular
    x = np.asarray(initial, float)
    p = x[:3] + matrix(x[6:10]).T@model.ell
    xf = closest_point(p, target, ARRIVAL_RADIUS_M)
    axis, angle = direction_rotation(matrix(x[6:10]) @ (xf-p))
    basis = np.r_[np.zeros(3), np.linalg.solve(model.Ka, axis)]
    limits = []
    slopes = []
    for sign in (1, -1):
        lam = model.allocator.solve(sign*basis)
        peak = float(lam.max())
        cap_limit = model.allocator.caps[1] / (model.h*np.linalg.norm(basis[3:]))
        limit = min(1.0/peak, cap_limit)
        limits.append(limit)
        slopes.append(model_fuel_rate(model, sign*model.h*basis*limit*0.5)/(limit*0.5))
    T = settings["slew_duration_s"]
    N = settings["slew_steps"]
    dt = T/N
    G = np.zeros((2*(N+1), N))
    for j in range(1, N+1):
        for i in range(j):
            G[2*j, i] = dt*dt*(j-i-0.5)
            G[2*j+1, i] = dt
    w = settings["slew_weights"]
    weights = np.tile([dt*w["angle"], dt*w["rate"]], N+1)
    weights[-2:] = [w["terminal_angle"], w["terminal_rate"]]
    desired = np.tile([angle, 0.0], N+1)
    Q = G.T@(weights[:, None]*G)
    b = -2*G.T@(weights*desired)
    fuel = w["fuel"]*dt*np.asarray(slopes)/NOMINAL_FUEL_KG
    effort = w["effort"]*dt
    def objective(z):
        a = z[:N]-z[N:]
        value = float(a@Q@a + b@a + effort*(z@z) + fuel[0]*z[:N].sum() + fuel[1]*z[N:].sum())
        grad = 2*Q@a + b
        return value, np.r_[grad+2*effort*z[:N]+fuel[0], -grad+2*effort*z[N:]+fuel[1]]
    # Complete the positive definite quadratic square and solve bounded linear
    # least squares. This avoids a generic nonlinear optimizer and soft-cost
    # cancellation near a very large terminal weight.
    D = np.c_[np.eye(N),-np.eye(N)]
    upper = np.r_[np.full(N,limits[0]),np.full(N,limits[1])]
    H = D.T@Q@D + effort*np.eye(2*N)
    linear = D.T@b + np.r_[np.full(N,fuel[0]),np.full(N,fuel[1])]
    L = np.linalg.cholesky(H)
    rhs = solve_triangular(L,-0.5*linear,lower=True)
    fit = lsq_linear(L.T*upper[None,:],rhs,bounds=(0.0,1.0),method="bvls",tol=1e-11,max_iter=2000)
    z = fit.x*upper
    alpha = z[:N]-z[N:]
    states = (G@alpha).reshape(-1, 2)
    grad = objective(z)[1]
    projected = grad.copy()
    projected[(z < 1e-9)&(grad>0)] = 0
    projected[(z > upper-1e-9)&(grad<0)] = 0
    info = dict(method="bounded fixed-axis quadratic setup plus exact reduced fuel",
        success=bool(fit.success and np.max(abs(projected))<1e-4), message=str(fit.message), iterations=int(fit.nit),
        projected_gradient_inf=float(np.max(abs(projected))),
        selected_duration_s=T, steps=N, angle_rad=angle, axis_B=axis.tolist(),
        achievable_alpha_positive_rad_s2=limits[0], achievable_alpha_negative_rad_s2=limits[1],
        minimum_rest_to_rest_time_s=float(np.sqrt(2*angle*(1/limits[0]+1/limits[1]))),
        ideal_terminal_pointing_error_deg=float(np.degrees(angle-states[-1,0])),
        ideal_terminal_rate_rad_s=float(states[-1,1]),
        reduced_slew_fuel_kg=float(dt*(slopes[0]*np.maximum(alpha,0).sum()+slopes[1]*np.maximum(-alpha,0).sum())),
        scope="Fixed-axis zero initial rate, no angular bias/delay; full model replay follows")
    return np.linspace(0,T,N+1), states, alpha, info


def packet_from_latent(model, x, target, sim_s=0.0):
    q = x[6:10]
    return dict(position_P_m=(x[:3]+matrix(q).T@model.ell).tolist(),
        velocity_P_m_s=(x[3:6]+model.vbias).tolist(),
        q_PB_xyzw_reported=q.tolist(), omega_relP_P_rad_s=(matrix(q).T@(x[10:13]+model.wbias)).tolist(),
        target_P_m=np.asarray(target).tolist(), read_begin_sim_s=sim_s, read_end_sim_s=sim_s,
        fuel_used_kg=0.0, damage_J=0.0)


def replay(folder, model_path, model, initial, target, duration, period=None):
    policy = Policy(folder, model_path)
    dt = period or policy.settings["control_period_s"]
    times = np.r_[np.arange(0, duration, dt), duration]
    x = np.asarray(initial,float).copy()
    commands, ct, bases = [], [], []
    xs, us, fuels, stages, points = [], [], [], [], []
    fuel = 0.0
    handoff = None
    p = model.p
    def integrate(start,end):
        nonlocal x
        edges = np.unique(np.r_[ct[-1]+p["linear_effect_delay_s"], ct[-1]+p["angular_effect_delay_s"], end])
        t = start
        while t < end-1e-11:
            upcoming = edges[edges>t+1e-10]
            step = min(end-t, model.definition["integration_max_step_s"], float(upcoming[0]-t) if len(upcoming) else end-t)
            mid = t+0.5*step
            il = int(np.searchsorted(ct,mid-p["linear_effect_delay_s"],side="right")-1)
            ia = int(np.searchsorted(ct,mid-p["angular_effect_delay_s"],side="right")-1)
            base = np.r_[bases[il][:3] if il>=0 else np.zeros(3), bases[ia][3:] if ia>=0 else np.zeros(3)]
            x = model._advance(x,step,base)
            t += step
    for i,t in enumerate(times):
        packet = packet_from_latent(model,x,target,t)
        # Replay includes report lags using causally stored latent velocity/rate.
        if xs:
            for sl,key,delay,bias,rotation in ((slice(3,6),"velocity_P_m_s",p["velocity_report_delay_s"],model.vbias,False),
                    (slice(10,13),"omega_relP_P_rad_s",p["rate_report_delay_s"],model.wbias,True)):
                values = np.asarray(xs+[x])[:,sl]
                past = np.array([np.interp(t-delay,times[:i+1],values[:,j]) for j in range(3)])+bias
                packet[key] = (matrix(x[6:10]).T@past if rotation else past).tolist()
        u,detail = policy.command(t,packet)
        if detail["stage"] == "forward" and handoff is None:
            handoff = dict(elapsed_s=float(t),pointing_error_deg=detail["pointing_error_deg"],
                angular_rate_rad_s=float(np.linalg.norm(x[10:13])), fuel_kg=float(fuel),
                position_P_m=packet["position_P_m"], velocity_P_m_s=x[3:6].tolist())
        xs.append(x.copy()); us.append(u); fuels.append(fuel)
        stages.append(detail["stage"]);points.append(detail["pointing_error_deg"])
        if i == len(times)-1:
            break
        ct.append(float(t)); commands.append(u); bases.append(model.allocator.mean(u,model.h))
        fuel += model_fuel_rate(model,u)*(times[i+1]-t)
        integrate(t,times[i+1])
    return dict(times=times,states=np.asarray(xs),commands=np.asarray(us),fuel=np.asarray(fuels),
        stages=np.asarray(stages),pointing_deg=np.asarray(points)),handoff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--prediction-s",type=float)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    folder = root/"policy"/"naive_handoff"
    model_path = root/"nominal_model.json"
    from system_identification.model import Model
    model = Model.load(model_path)
    settings = load_settings(folder/"settings.json")
    mission = json.loads((folder/"mission.json").read_text(encoding="utf-8-sig"))
    initial = np.asarray(mission["initial_state_latent"],float)
    target = np.asarray(mission["target_center_P_m"],float)
    start = time.monotonic()
    times,states,alpha,solver = setup_problem(model,initial,target,settings)
    dt = settings["slew_duration_s"]/settings["slew_steps"]
    gains = riccati(dt,settings["slew_steps"],settings["tracking_weights"])
    from scipy.linalg import solve_discrete_are
    h = settings["control_period_s"]
    A = np.array([[1.,h],[0.,1.]])
    B = np.array([[h*h/2],[h]])
    w = settings["hold_weights"]
    Q = h*np.diag([w["angle"],w["rate"]]); R=h*w["effort"]
    P = solve_discrete_are(A,B,Q,np.array([[R]]))
    hold = np.linalg.solve(np.array([[R]])+B.T@P@B,B.T@P@A)[0]
    assets = folder/"assets"
    assets.mkdir(exist_ok=True)
    asset = assets/"reference.npz"
    np.savez_compressed(asset,slew_times=times,slew_states=states,slew_alpha=alpha,
        tracking_gains=gains,hold_gain=hold)
    knees = [float(model.h/model.allocator.solve(np.eye(6)[j]).max()) for j in (0,2)]
    manifest = dict(schema_version=1,model_sha256=sha256(model_path),settings_sha256=sha256(folder/"settings.json"),
        mission_sha256=sha256(folder/"mission.json"),reference_sha256=sha256(asset),arrival_radius_m=ARRIVAL_RADIUS_M,
        slew_angle_rad=solver["angle_rad"],forward_request_at_saturation_knee=knees[0],
        boost_request_at_saturation_knee=knees[1],solver=solver,
        maximum_thrust_note="Reference ratios use model knees; full-thrust phases common-scale to a native request cap, preserving six-axis ratios and saturated delivery. Angular corrections share normalization.",
        native_qualified=False)
    manifest["generator_sources_sha256"]={name:sha256(folder/name) for name in ("precompute.py","control.py","accounting.py")}
    manifest_path = assets/"manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    duration = args.prediction_s or settings["prediction_duration_s"]
    data,handoff = replay(folder,model_path,model,initial,target,duration)
    np.savez_compressed(assets/"prediction.npz",**data)
    # Integration refinement on the SAME command sequence, without a second controller.
    original_step = model.definition["integration_max_step_s"]
    model.definition["integration_max_step_s"] = original_step/2
    refined = model.rollout(initial,0,data["times"],data["times"][:-1],data["commands"][:-1])
    model.definition["integration_max_step_s"] = original_step
    refinement = dict(max_position_difference_m=float(np.max(np.linalg.norm(refined[:,:3]-data["states"][:,:3],axis=1))),
        max_rate_difference_rad_s=float(np.max(np.linalg.norm(refined[:,10:13]-data["states"][:,10:13],axis=1))))
    manifest.update(prediction_duration_s=duration,model_handoff=handoff,integration_refinement=refinement,
        precompute_wall_s=time.monotonic()-start)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (assets/"precompute_report.txt").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    t=data["times"]
    axs[0,0].plot(t,data["pointing_deg"]);axs[0,0].set_ylabel("+X pointing error [deg]")
    axs[0,1].plot(t,data["states"][:,10:13]);axs[0,1].set_ylabel("Body angular rates [rad/s]")
    axs[1,0].plot(t,data["commands"]);axs[1,0].set_ylabel("Six public requests [mixed units]")
    axs[1,1].plot(t,data["fuel"]);axs[1,1].set_ylabel("Model fuel estimate [kg]")
    for ax in axs.flat:
        ax.set_xlabel("Controller elapsed [s]");ax.grid(True,alpha=.3)
        for when in (settings["boost_duration_s"],settings["boost_duration_s"]+settings["slew_duration_s"]):
            ax.axvline(when,color="gray",linestyle=":")
    fig.savefig(assets/"prediction.png",dpi=150)
    print(json.dumps(dict(solver=solver,model_handoff=handoff,integration_refinement=refinement,
        assets=str(assets),wall_s=manifest["precompute_wall_s"]),indent=2))
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
