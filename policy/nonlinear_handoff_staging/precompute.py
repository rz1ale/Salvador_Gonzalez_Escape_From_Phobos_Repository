"""Prepare, independently replay, build TVLQR gains, optionally select. No SDK."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
LOCAL=Path(__file__).with_name("_offline_packages")
if LOCAL.is_dir():sys.path.insert(0,str(LOCAL))


def ensure_solver(install=False):
    if importlib.util.find_spec("casadi") is None:
        if not install:raise ValueError("Offline solver missing. Run this same command with --install-solver --select (solution-local CasADi only).")
        subprocess.run([sys.executable,"-m","pip","install","--target",str(LOCAL),"--no-deps","casadi==3.7.2"],check=True)
        sys.path.insert(0,str(LOCAL));importlib.invalidate_caches()


def tangent_maps(q):
    import numpy as np
    v=q[:3];w=q[3];a,b,c=v;S=np.array([[0,-c,b],[c,0,-a],[-b,a,0.]])
    E=np.zeros((13,12));E[:6,:6]=np.eye(6);E[6:9,6:9]=-.5*(w*np.eye(3)-S)
    E[9,6:9]=.5*v;E[10:,9:]=np.eye(3)
    G=np.zeros((12,13));G[:6,:6]=np.eye(6);G[6:9,6:9]=-2*(w*np.eye(3)+S)
    G[6:9,9]=2*v;G[9:,10:]=np.eye(3)
    return E,G


def endpoint_metrics(states,model,reference):
    import numpy as np
    from policy.nonlinear_handoff_staging.math_core import reported,matrix,direction_rotation
    x=states[-1];p=reported(x,model.definition)
    _,angle=direction_rotation(matrix(x[6:10])@(reference["xf"]-p))
    return dict(pointing_deg=float(np.degrees(angle)),rate_rad_s=float(np.linalg.norm(x[10:13])),
        position_error_m=float(np.linalg.norm(p-reference["pd"])),position_P_m=p.tolist(),
        velocity_P_m_s=x[3:6].tolist(),forward_velocity_m_s=float(reference["n"]@x[3:6]))


def validate_replay(model,initial,result,s):
    import numpy as np
    from policy.nonlinear_handoff_staging.math_core import error,terminal_limits
    times=result["times"];ct=times[:-1];commands=result["commands"]
    coarse=model.rollout(initial,0,times,ct,commands)
    original=model.definition["integration_max_step_s"];model.definition["integration_max_step_s"]=original/2
    try:fine=model.rollout(initial,0,times,ct,commands)
    finally:model.definition["integration_max_step_s"]=original
    endpoint=endpoint_metrics(fine,model,result["reference"]);d=s["design"];limits=terminal_limits(s,result["reference"])
    defects=np.asarray([error(a,b) for a,b in zip(coarse,result["states"])])
    refinement=np.asarray([error(a,b) for a,b in zip(fine,coarse)])
    checks=dict(pointing=endpoint["pointing_deg"]<=limits["pointing_deg"],
        rate=endpoint["rate_rad_s"]<=limits["rate_rad_s"],
        position=d["terminal_position_tolerance_m"] is None or endpoint["position_error_m"]<=d["terminal_position_tolerance_m"],
        trajectory_defects=float(np.max(abs(defects)))<.005,
        input_norms=bool(np.max(np.linalg.norm(commands[:,:3],axis=1))<=model.allocator.caps[0]*(1+1e-7) and
            np.max(np.linalg.norm(commands[:,3:],axis=1))<=model.allocator.caps[1]*(1+1e-7)))
    report=dict(endpoint=endpoint,limits=limits,checks=checks,passed=all(checks.values()),
        max_nlp_replay_error=float(np.max(abs(defects))),max_refinement_error=float(np.max(abs(refinement))),
        canonical_step_s=original,refined_step_s=original/2,native_qualified=False)
    return report


def feedback_cache(model,initial,result,s):
    import numpy as np
    from scipy.linalg import solve_discrete_are
    from policy.nonlinear_handoff_staging.offline import symbolic,Allocation,weights
    h=s["feedback"]["period_s"];T=s["design"]["duration_s"]
    # A final short interval keeps the exact requested maneuver duration.
    grid=np.unique(np.r_[np.arange(0,T,h),T]);states=model.rollout(initial,0,grid,result["times"][:-1],result["commands"])
    counts=len(grid)-1;scales=result["scales"]
    commands=np.asarray([result["commands"][min(len(result["commands"])-1,int(np.searchsorted(result["times"][:-1],t+1e-10,side="right")-1))] for t in grid[:-1]])
    allocator=Allocation(model,scales,s["solver"]["allocation_derivative_step"])
    derivatives={};transitions={};linear=[]
    for i,dt in enumerate(np.diff(grid)):
        key=round(float(dt),10)
        if key not in transitions:transitions[key]=symbolic(model.definition,float(dt),model.definition["integration_max_step_s"])
        phi,jac,_,_,_=transitions[key]
        new=commands[i];old=commands[i-1] if i else np.zeros(6)
        bnew=model.allocator.mean(new,model.h);bold=model.allocator.mean(old,model.h)
        yn=np.asarray(phi(states[i],bold,bnew)).ravel()
        Jx,Jo,Jn=[np.asarray(a) for a in jac(states[i],bold,bnew)]
        E,_=tangent_maps(states[i,6:10]);_,G=tangent_maps(yn[6:10])
        ds=[]
        for u in (old,new):
            k=tuple(u)
            if k not in derivatives:
                # Flight corrections can leave a canonical feasible face. Use
                # the ORIGINAL saturated map on both sides, not the interior
                # one-sided derivative used by canonical NLP coordinates.
                eps=s["solver"]["allocation_derivative_step"]
                derivatives[k]=np.column_stack([(model.allocator.mean(u+scales*np.eye(6)[j]*eps,model.h)-
                    model.allocator.mean(u-scales*np.eye(6)[j]*eps,model.h))/(2*eps) for j in range(6)])
            ds.append(derivatives[k])
        A=G@Jx@E;D=G@Jo@ds[0];B=G@Jn@ds[1]
        Aaug=np.block([[A,D],[np.zeros((6,12)),np.zeros((6,6))]])
        Baug=np.vstack([B,np.eye(6)])
        linear.append((Aaug,Baug,float(dt)))
    def tracking_weights(name):
        q=weights(s["design"],name).copy();q[:3]*=s["feedback"]["position_weight_multiplier"]
        q[9:]*=s["feedback"]["rate_weight_multiplier"];return q
    P=np.diag(np.r_[tracking_weights("Q_final"),np.zeros(6)])
    gains=[]
    for A,B,dt in reversed(linear):
        R=np.diag(s["design"]["R"])*dt;Q=np.diag(np.r_[tracking_weights("Q"),np.zeros(6)])*dt
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
        P=Q+A.T@P@(A-B@K);P=.5*(P+P.T);gains.append(K)
    A=np.array([[1.,h],[0.,1.]]);B=np.array([[h*h/2],[h]])
    f=s["feedback"];Q=h*np.diag([f["continuation_angle_weight"],f["continuation_rate_weight"]]);R=h*f["continuation_effort_weight"]
    P=solve_discrete_are(A,B,Q,np.array([[R]]));hold=np.linalg.solve(np.array([[R]])+B.T@P@B,B.T@P@A)[0]
    return dict(times=grid,states=states,command_times=result["times"][:-1],commands=result["commands"],
        gain_times=grid[:-1],gains=np.asarray(gains[::-1]),input_scales=scales,continuation_gain=hold)


def select_policy(root):
    root=Path(root).resolve()
    subprocess.run([sys.executable,str(root/"run_simulation.py"),"select","--config","policy/nonlinear_handoff_staging/run_plan.json"],cwd=root,check=True)
    subprocess.run([sys.executable,str(root/"run_simulation.py"),"check"],cwd=root,check=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--select",action="store_true",help="Select ONLY after successful preparation; never launches a sim")
    parser.add_argument("--install-solver",action="store_true",help="If missing, install pinned CasADi into this policy's _offline_packages")
    parser.add_argument("--check",action="store_true",help="Check current prepared asset without optimization")
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--force",action="store_true",help="Re-optimize even when the prepared trajectory is unchanged")
    args=parser.parse_args(argv)
    try:
        os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("OMP_NUM_THREADS","1")
        import numpy as np
        from policy.nonlinear_handoff_staging.math_core import settings,signature,trajectory_signature,sha256,mission_reference,terminal_limits
        folder=args.root.resolve()/"policy"/"nonlinear_handoff_staging";model_path=args.root.resolve()/"nominal_model.json"
        if args.check:
            from policy.nonlinear_handoff_staging.flight import Policy
            p=Policy(folder,model_path);print(json.dumps(p.manifest["validation"],indent=2));return 0
        s=settings(folder/"settings.json")
        if not args.force:
            from policy.nonlinear_handoff_staging.flight import Policy
            try:cached=Policy(folder,model_path)
            except (ValueError,FileNotFoundError):cached=None
            if cached is not None:
                if args.install_solver:ensure_solver(True)
                print("Verified cached reference reused; no optimizer run. Status: "+cached.manifest["candidate_status"])
                if args.select:select_policy(args.root)
                return 0
        ensure_solver(args.install_solver)
        from system_identification.model import Model
        from policy.nonlinear_handoff_staging.offline import solve
        from policy.nonlinear_handoff_staging.accounting import model_fuel_rate
        model=Model.load(model_path)
        mission=json.loads((folder/"mission.json").read_text(encoding="utf-8-sig"))
        initial=np.asarray(mission["initial_state_latent"],float);target=np.asarray(mission["target_center_P_m"],float)
        print("One coupled nonlinear maneuver; all six public request channels. No programmed boost/slew segments.",flush=True)
        limits=terminal_limits(s,mission_reference(initial,target,model.definition,s["design"]))
        print("Distance %.3f m | duration %.3f s | %d intervals | terminal angle %.4f deg | rate %.5f rad/s" %
            (s["design"]["distance_m"],s["design"]["duration_s"],s["design"]["steps"],limits["pointing_deg"],limits["rate_rad_s"]),flush=True)
        def progress(i,j,t):print("iteration %d | quadratic objective %.7g | wall %.1f s"%(i,j,t),flush=True)
        warm=None;old_meta={};previous_cache=None
        old_manifest=folder/"assets"/"manifest.json";old_asset=folder/"assets"/"reference.npz"
        if old_manifest.is_file() and old_asset.is_file():
            old_meta=json.loads(old_manifest.read_text(encoding="utf-8-sig"))
            if old_meta.get("model_sha256")==sha256(model_path) and old_meta.get("mission_sha256")==sha256(folder/"mission.json"):
                with np.load(old_asset,allow_pickle=False) as previous:
                    warm=dict(times=np.r_[previous["command_times"],previous["times"][-1]],commands=previous["commands"])
                    previous_cache={key:np.asarray(previous[key]) for key in previous.files}
        if (not args.force and warm is not None and old_meta.get("export_verified") and
                old_meta.get("trajectory_signature")==trajectory_signature(s)):
            print("Trajectory objective unchanged: replaying existing controls and rebuilding feedback only.",flush=True)
            from policy.nonlinear_handoff_staging.offline import input_scales
            result=dict(times=warm["times"],commands=warm["commands"],scales=input_scales(model),
                reference=mission_reference(initial,target,model.definition,s["design"]),solver=dict(old_meta["solver"],reused_reference=True))
            result["states"]=model.rollout(initial,0,result["times"],result["times"][:-1],result["commands"])
        else:result,_=solve(model,initial,target,s,progress,warm=warm)
        validation=validate_replay(model,initial,result,s)
        report=dict(solver=result["solver"],validation=validation,export_verified=False)
        assets=folder/"assets";assets.mkdir(exist_ok=True)
        (assets/"prepare_report.txt").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        stationary=result["solver"]["success"] and result["solver"]["constraint_violation"]<=s["solver"]["constraint_tolerance"]
        good=validation["passed"] and (stationary or s["solver"]["accept_verified_candidate"])
        if not good:
            np.savez_compressed(assets/"unverified_candidate.npz",times=result["times"],states=result["states"],commands=result["commands"])
            print(json.dumps(report,indent=2));raise ValueError("Preparation failed its declared convergence/replay checks. No new trajectory exported or selected. Inspect assets/prepare_report.txt; do not launch this changed design.")
        cache=feedback_cache(model,initial,result,s)
        rates=np.array([model_fuel_rate(model,u) for u in result["commands"]])
        fuel=np.r_[0.,np.cumsum(rates*np.diff(result["times"]))]
        temp=assets/"reference_pending.npz";np.savez_compressed(temp,**cache)
        sources={name:sha256(folder/name) for name in ("math_core.py","flight.py","entry.py","accounting.py")}
        manifest=dict(schema_version=1,export_verified=True,design_signature=signature(s),trajectory_signature=trajectory_signature(s),model_sha256=sha256(model_path),
            mission_sha256=sha256(folder/"mission.json"),reference_sha256=sha256(temp),runtime_sources_sha256=sources,
            reference={k:v.tolist() if hasattr(v,"tolist") else v for k,v in result["reference"].items()},
            terminal_public_request=result["commands"][-1].tolist(),input_scales=result["scales"].tolist(),
            solver=result["solver"],validation=validation,model_fuel_kg=float(fuel[-1]),
            candidate_status="converged_local_solution" if stationary else "replay_verified_nonstationary_candidate",
            feedback="time-varying LQR about nonlinear trajectory; 12 physical errors + six previous-request deviations",
            feedback_weight_multipliers={key:s["feedback"][key] for key in ("position_weight_multiplier","rate_weight_multiplier")},
            feedback_input_derivative="central differences of original saturated allocator",
            design_used=s["design"],native_backend_sha256=sha256(args.root.resolve()/"system_identification/model.py"),
            native_qualified=False,obstacle_avoidance=False,official_score_captured=False)
        temp.replace(assets/"reference.npz")
        (assets/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        (assets/"unverified_candidate.npz").unlink(missing_ok=True)
        report.update(export_verified=True,model_fuel_kg=float(fuel[-1]),candidate_status=manifest["candidate_status"])
        (assets/"prepare_report.txt").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        make_plot(model,result,fuel,folder,args.show)
        print(json.dumps(report,indent=2))
        if args.select:
            select_policy(args.root)
        print("Prepared. Open the unchanged official simulation manually and select spectator.")
        return 0
    except (ValueError,ImportError,OSError,subprocess.CalledProcessError) as exc:
        print("PREPARATION FAILED: "+str(exc),file=sys.stderr);return 2


def make_plot(model,result,fuel,folder,show):
    import numpy as np
    import matplotlib
    if not show:matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from policy.nonlinear_handoff_staging.math_core import reported,matrix,direction_rotation
    t=result["times"];x=result["states"];ref=result["reference"]
    p=np.array([reported(a,model.definition) for a in x])
    angle=[np.degrees(direction_rotation(matrix(a[6:10])@(ref["xf"]-pr))[1]) for a,pr in zip(x,p)]
    fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for j,name in enumerate(("x","y","z")):
        axs[0,0].plot(t,p[:,j]-ref["pd"][j],label=name)
        axs[1,1].plot(t,x[:,10+j],label=name)
        axs[2,0].step(t[:-1],result["commands"][:,j],where="post",label=name)
        axs[2,1].step(t[:-1],result["commands"][:,3+j],where="post",label=name)
    axs[0,0].set_ylabel("Reported position - xd [m]");axs[0,0].legend()
    axs[0,1].plot(t,angle);axs[0,1].set_ylabel("+X pointing error [deg]")
    axs[1,0].plot(t,fuel);axs[1,0].set_ylabel("Model fuel [kg]")
    axs[1,1].set_ylabel("Body angular rates [rad/s]")
    axs[2,0].set_ylabel("Public translation requests [m/s²]")
    axs[2,1].set_ylabel("Public angular requests [rad/s²]")
    for ax in axs.flat:ax.grid(True,alpha=.3);ax.set_xlabel("Maneuver time [s]")
    fig.savefig(folder/"assets"/"prediction.png",dpi=150)
    if show:plt.show()
    plt.close(fig)


if __name__=="__main__":raise SystemExit(main())
