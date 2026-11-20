"""Offline public-stat accounting, per-stage diagnostics, and run comparison."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from policy.naive_handoff.accounting import ARRIVAL_RADIUS_M, DAMAGE_THRESHOLD_J, score_estimate
from policy.naive_handoff.control import sha256, matrix, pointing_error


def read_lines(path):
    if not Path(path).is_file():
        return []
    lines=Path(path).read_text(encoding="utf-8-sig").splitlines()
    rows=[]
    for i,line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise ValueError("Invalid interior log line %d: %s" % (i+1,path))
            # An interrupted final write is not silently treated as a complete run.
    return rows


def read_json(path,default):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).is_file() else default


def attempt_metrics(attempt, *, model=None):
    attempt=Path(attempt)
    rows=read_lines(attempt/"naive_handoff_telemetry.txt")
    settings=read_json(attempt/"settings_used.json",{})
    summary=read_json(attempt/"naive_handoff_summary.txt",{})
    result=read_json(attempt/"result.json",{})
    if not rows:
        return dict(attempt=str(attempt),label=settings.get("label","unknown"),outcome="no_policy_samples"),rows
    # Include the final public read at arrival/test completion. Telemetry rows
    # describe commands and therefore stop just before that final read.
    last=summary.get("last_state")
    if last:
        t=float(last["read_end_sim_s"])
        if t>rows[-1]["sim_s"]+1e-9:
            final=dict(rows[-1])
            final.update(sim_s=t,elapsed_s=rows[-1]["elapsed_s"]+t-rows[-1]["sim_s"],
                position_P_m=last["position_P_m"],velocity_P_m_s=last["velocity_P_m_s"],
                distance_m=float(np.linalg.norm(np.asarray(last["position_P_m"])-last["target_P_m"])),
                fuel_kg=last["fuel_used_kg"],damage_J=last["damage_J"],final_read_only=True)
            final["omega_B_rad_s"]=(matrix(last["q_PB_xyzw_reported"])@np.asarray(last["omega_relP_P_rad_s"])).tolist()
            final["pointing_error_deg"]=float(np.degrees(pointing_error(last["q_PB_xyzw_reported"],
                last["position_P_m"],rows[-1]["closest_reference_P_m"])))
            rows.append(final)
    times=np.asarray([r["sim_s"] for r in rows])
    if np.any(np.diff(times)<-1e-6):
        raise ValueError("Simulation clock reset inside attempt: "+str(attempt))
    fuel=np.asarray([r["fuel_kg"] for r in rows])
    distance=np.asarray([r["distance_m"] for r in rows])
    # Preserve the first observed arrival. No estimated score for a partial run.
    hits=np.flatnonzero(distance<=ARRIVAL_RADIUS_M)
    index=int(hits[0]) if len(hits) else len(rows)-1
    end=rows[index]
    damage=float(end["damage_J"])
    arrived=bool(len(hits) and damage<=DAMAGE_THRESHOLD_J)
    fuel_reset=bool(np.any(np.diff(fuel[:index+1]) < -1e-5))
    transitions=summary.get("stage_transitions",[])
    if not transitions:
        previous=None
        for r in rows:
            if r["stage"]!=previous:
                transitions.append(dict(stage=r["stage"],sim_s=r["sim_s"],elapsed_s=r["elapsed_s"],fuel_kg=r["fuel_kg"],
                    pointing_error_deg=r["pointing_error_deg"],omega_norm_rad_s=float(np.linalg.norm(r["omega_B_rad_s"])),
                    speed_m_s=float(np.linalg.norm(r["velocity_P_m_s"]))))
                previous=r["stage"]
    stage_metrics=[]
    for i,tr in enumerate(transitions):
        if tr["sim_s"]>end["sim_s"]:
            break
        after=transitions[i+1] if i+1<len(transitions) else None
        stop=min(float(after["sim_s"]) if after else end["sim_s"],end["sim_s"])
        f0=float(tr["fuel_kg"])
        f1=float(after["fuel_kg"]) if after and after["sim_s"]<=end["sim_s"] else float(end["fuel_kg"])
        seconds=max(0.0,stop-float(tr["sim_s"]))
        stage_metrics.append(dict(stage=tr["stage"],observed_duration_s=seconds,public_fuel_increment_kg=f1-f0,
            mean_public_fuel_rate_kg_s=(f1-f0)/seconds if seconds>0 else None,
            entry_pointing_error_deg=tr["pointing_error_deg"],entry_angular_rate_rad_s=tr["omega_norm_rad_s"],
            boundary_resolution="first read of the next stage; SDK read/send timing logged"))
    handoff=next((r for r in transitions if r["stage"]=="forward"),None)
    intervals=np.diff(times[:index+1])
    metrics=dict(attempt=str(attempt),label=settings.get("label","unknown"),
        outcome="observed_arrival" if arrived else summary.get("reason",result.get("reason","incomplete_or_running")),
        observed_arrival=arrived,official_score_captured=False,
        scorer_time_note="sim_s is the public runner clock; native scorer owns its actual start/poll instant",
        elapsed_observed_s=float(times[index]-times[0]),final_controller_elapsed_s=float(end["elapsed_s"]),
        final_sim_s=float(end["sim_s"]),first_policy_sim_s=float(times[0]),
        arrival_sim_s=float(end["sim_s"]) if arrived else None,
        public_fuel_total_kg=float(end["fuel_kg"]),public_fuel_increment_kg=float(end["fuel_kg"]-fuel[0]),
        fuel_counter_reset_detected=fuel_reset,damage_J=damage,
        score_formula_estimate=score_estimate(end["sim_s"],end["fuel_kg"],damage) if arrived and not fuel_reset else None,
        initial_distance_m=float(distance[0]),final_distance_m=float(distance[index]),minimum_sampled_distance_m=float(distance[:index+1].min()),
        distance_reduction_m=float(distance[0]-distance[index]),
        mean_progress_per_fuel_m_kg=float((distance[0]-distance[index])/(end["fuel_kg"]-fuel[0])) if end["fuel_kg"]>fuel[0] else None,
        max_control_observation_gap_s=float(intervals.max(initial=0.0)),
        median_control_observation_gap_s=float(np.median(intervals)) if len(intervals) else None,
        configured_boost_s=settings.get("boost_duration_s"),configured_slew_s=settings.get("slew_duration_s"),
        slew_weights=settings.get("slew_weights"),hold_weights=settings.get("hold_weights"),
        handoff=handoff,stages=stage_metrics,settings=settings,
        scoring_scope="Winning formula only after observed arrival; no model optimum or official score claimed")
    if model is not None:
        from policy.naive_handoff.accounting import model_fuel_rate
        commands=read_lines(attempt/"commands.txt")
        estimate=0.0
        for i,c in enumerate(commands):
            begin=float(c["send_end_sim_s"])
            finish=float(commands[i+1]["send_end_sim_s"]) if i+1<len(commands) else end["sim_s"]
            lo=max(begin,float(times[0]));hi=min(finish,float(end["sim_s"]))
            if hi>lo:
                u=np.r_[c["linear_body_request_m_s2"],c["angular_body_request_rad_s2"]]
                estimate+=model_fuel_rate(model,u)*(hi-lo)
        metrics["model_fuel_increment_estimate_kg"]=estimate
        metrics["model_fuel_note"]="Identified mean-period estimate over actual recorded send intervals; public Stats/FuelUsed is primary"
    return metrics,rows


def plot_attempt(rows,metrics,destination):
    import matplotlib.pyplot as plt
    if not rows:
        return
    t=np.asarray([r["elapsed_s"] for r in rows])
    fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    axs[0,0].plot(t,[r["distance_m"] for r in rows]);axs[0,0].axhline(ARRIVAL_RADIUS_M,color="k",linestyle=":")
    axs[0,0].set_ylabel("Distance to target center [m]")
    axs[0,1].plot(t,[r["fuel_kg"] for r in rows]);axs[0,1].set_ylabel("Public cumulative fuel [kg]")
    axs[1,0].plot(t,[r["pointing_error_deg"] for r in rows]);axs[1,0].set_ylabel("+X pointing error [deg]")
    axs[1,1].plot(t,[r["omega_B_rad_s"] for r in rows]);axs[1,1].set_ylabel("Body angular velocity [rad/s]")
    requests=np.asarray([r["public_request"] for r in rows])
    for j,name in enumerate(("x","y","z")):
        axs[2,0].step(t,requests[:,j],where="post",label=name)
        axs[2,1].step(t,requests[:,j+3],where="post",label=name)
    axs[2,0].set_ylabel("Public translation requests [m/s²]")
    axs[2,1].set_ylabel("Public angular requests [rad/s²]")
    axs[2,0].legend();axs[2,1].legend()
    for ax in axs.flat:
        ax.grid(True,alpha=.3);ax.set_xlabel("Controller elapsed [s]")
        for stage in metrics.get("stages",[])[1:]:
            tr=next(r for r in rows if r["stage"]==stage["stage"])
            ax.axvline(tr["elapsed_s"],color="gray",linestyle=":")
    fig.suptitle(metrics.get("label","")+" | "+metrics.get("outcome",""))
    fig.savefig(destination/"run_metrics.png",dpi=150)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session",type=Path)
    group.add_argument("--all",action="store_true",help="Rebuild comparison from this policy's run_logs")
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--model-fuel",action="store_true",help="Optional slower estimate; requires unchanged model hash")
    args=parser.parse_args(argv)
    root=args.root.resolve();folder=root/"policy"/"naive_handoff"
    base=folder/"run_logs" if args.all else args.session
    if not base.is_absolute():
        base=root/base
    if not base.exists():
        raise ValueError("Missing session/log folder: "+str(base))
    attempts=sorted({p.parent for p in base.rglob("naive_handoff_telemetry.txt")})
    if not attempts:
        raise ValueError("No naive_handoff_telemetry.txt below "+str(base))
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    all_metrics=[]
    output=folder/"analysis_results"
    for attempt in attempts:
        model=None
        if args.model_fuel:
            used=read_json(attempt/"reference_used.json",{})
            if sha256(root/"nominal_model.json")!=used.get("model_sha256"):
                raise ValueError("Model changed since this run; analyze public statistics without --model-fuel")
            from system_identification.model import Model
            model=Model.load(root/"nominal_model.json")
        metrics,rows=attempt_metrics(attempt,model=model)
        all_metrics.append(metrics)
        session=next((p for p in attempt.parents if (p/"plan.json").is_file()),attempt.parent)
        destination=output/session.name/attempt.parent.name/attempt.name
        destination.mkdir(parents=True,exist_ok=True)
        (destination/"metrics.txt").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
        plot_attempt(rows,metrics,destination)
        print(json.dumps({k:metrics.get(k) for k in ("label","outcome","final_controller_elapsed_s","public_fuel_total_kg","final_distance_m","score_formula_estimate")}))
    # --session updates that report; --all rebuilds the canonical comparison.
    output.mkdir(parents=True,exist_ok=True)
    if args.all:
        columns=["attempt","label","outcome","configured_boost_s","configured_slew_s","first_policy_sim_s",
            "final_controller_elapsed_s","arrival_sim_s","public_fuel_total_kg","public_fuel_increment_kg", "damage_J",
            "final_distance_m","distance_reduction_m","mean_progress_per_fuel_m_kg","score_formula_estimate",
            "handoff_pointing_error_deg","handoff_rate_rad_s","boost_fuel_kg","slew_fuel_kg","forward_fuel_kg"]
        with (output/"comparison.txt").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,delimiter="\t",extrasaction="ignore")
            writer.writeheader()
            for m in all_metrics:
                row=dict(m);h=m.get("handoff") or {}
                row.update(handoff_pointing_error_deg=h.get("pointing_error_deg"),handoff_rate_rad_s=h.get("omega_norm_rad_s"))
                for s in m.get("stages",[]):
                    row[s["stage"]+"_fuel_kg"]=s["public_fuel_increment_kg"]
                writer.writerow(row)
        (output/"comparison_details.json").write_text(json.dumps(all_metrics,indent=2)+"\n",encoding="utf-8")
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for m in all_metrics:
            if "public_fuel_total_kg" not in m:
                continue
            marker="o" if m["observed_arrival"] else "x"
            axs[0].scatter(m["public_fuel_total_kg"],m["final_controller_elapsed_s"],marker=marker)
            axs[0].annotate(m["label"],(m["public_fuel_total_kg"],m["final_controller_elapsed_s"]),fontsize=7)
            axs[1].scatter(m["public_fuel_increment_kg"],m["distance_reduction_m"],marker=marker)
        axs[0].set(xlabel="Public total fuel [kg]",ylabel="Observed controller elapsed [s]",title="Circle: arrival; cross: partial run")
        axs[1].set(xlabel="Public fuel since first sample [kg]",ylabel="Distance reduction [m]",title="Compare partial runs at equal duration")
        for ax in axs:
            ax.grid(True,alpha=.3)
        fig.savefig(output/"comparison.png",dpi=150)
        if args.show:
            plt.show()
        plt.close(fig)
    elif args.show:
        # Per-run figures were saved and closed; display saved reports explicitly.
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        for p in output.glob("*/"+attempts[-1].parent.name+"/"+attempts[-1].name+"/run_metrics.png"):
            fig,ax=plt.subplots(figsize=(12,10));ax.imshow(mpimg.imread(p));ax.axis("off")
        plt.show()
    print("Analysis: "+str(output))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
