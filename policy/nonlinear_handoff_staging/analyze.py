"""Local public-stat comparison for completed or interrupted staging runs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from policy.nonlinear_handoff_staging.accounting import ARRIVAL_RADIUS_M,DAMAGE_THRESHOLD_J,score_estimate


def read_json(path,default):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).is_file() else default


def read_lines(path):
    if not Path(path).is_file():return []
    lines=Path(path).read_text(encoding="utf-8-sig").splitlines();out=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:out.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise ValueError("Malformed interior log line: "+str(path))
    return out


def metrics(attempt):
    attempt=Path(attempt);rows=read_lines(attempt/"staging_telemetry.txt")
    s=read_json(attempt/"settings_used.json",{});summary=read_json(attempt/"staging_summary.txt",{})
    result=read_json(attempt/"result.json",{})
    if not rows:return dict(attempt=str(attempt),outcome=result.get("reason","no_policy_samples")),[]
    end=summary.get("last_state")
    if end and end["read_end_sim_s"]>rows[-1]["sim_s"]:
        r=dict(rows[-1]);t=end["read_end_sim_s"]
        r.update(sim_s=t,elapsed_s=r["elapsed_s"]+t-r["sim_s"],fuel_kg=end["fuel_used_kg"],damage_J=end["damage_J"],
            position_P_m=end["position_P_m"],velocity_P_m_s=end["velocity_P_m_s"],
            distance_m=float(np.linalg.norm(np.asarray(end["position_P_m"])-end["target_P_m"])),final_read_only=True)
        rows.append(r)
    hits=[i for i,r in enumerate(rows) if r["distance_m"]<=ARRIVAL_RADIUS_M]
    last=rows[hits[0]] if hits else rows[-1]
    arrived=bool(hits and last["damage_J"]<=DAMAGE_THRESHOLD_J)
    handoff=summary.get("handoff")
    if handoff is None:handoff=next((r for r in rows if r["elapsed_s"]>=s["design"]["duration_s"]),None)
    used=np.asarray([r["fuel_kg"] for r in rows]);reset=bool(np.any(np.diff(used)<-1e-5))
    duration=float(last["elapsed_s"]);increment=float(last["fuel_kg"]-rows[0]["fuel_kg"])
    planned=read_json(attempt/"reference_used.json",{})
    m=dict(attempt=str(attempt),label=s["execution"]["label"],
        outcome="observed_arrival" if arrived else summary.get("reason",result.get("reason","incomplete_or_running")),
        observed_arrival=arrived,official_score_captured=False,stop_sim_requested=bool(result.get("stop_sim_requested",False)),
        distance_setting_m=s["design"]["distance_m"],duration_setting_s=s["design"]["duration_s"],steps=s["design"]["steps"],
        first_policy_sim_s=float(rows[0]["sim_s"]),elapsed_controller_s=duration,last_sim_s=float(last["sim_s"]),
        fuel_total_kg=float(last["fuel_kg"]),fuel_increment_kg=increment,damage_J=float(last["damage_J"]),
        final_distance_m=float(last["distance_m"]),distance_reduction_m=float(rows[0]["distance_m"]-last["distance_m"]),
        handoff_pointing_deg=handoff.get("pointing_error_deg") if handoff else None,
        handoff_rate_rad_s=handoff.get("rate_norm_rad_s") if handoff else None,
        handoff_xd_error_m=handoff.get("xd_distance_m") if handoff else None,
        handoff_forward_velocity_m_s=handoff.get("forward_velocity_m_s") if handoff else None,
        handoff_fuel_total_kg=handoff.get("fuel_kg") if handoff else None,
        staging_fuel_increment_kg=float(handoff["fuel_kg"]-rows[0]["fuel_kg"]) if handoff else None,
        fuel_counter_reset=reset,maximum_sample_gap_s=float(np.diff([r["sim_s"] for r in rows]).max(initial=0.)),
        score_formula_estimate=score_estimate(last["sim_s"],last["fuel_kg"],last["damage_J"]) if arrived and not reset else None,
        model_endpoint=planned.get("validation",{}).get("endpoint"),model_staging_fuel_kg=planned.get("model_fuel_kg"),
        handoff=handoff,settings=s,score_note="Formula estimate uses public runner clock; native scorer owns start/poll timing and official result")
    if handoff:
        d=s["design"]
        angle_limit=planned.get("validation",{}).get("limits",{}).get("pointing_deg",d["terminal_pointing_deg"])
        m["measured_handoff_within_requested_tolerances"]=bool(m["handoff_pointing_deg"]<=angle_limit and
            m["handoff_rate_rad_s"]<=d["terminal_rate_rad_s"] and (d["terminal_position_tolerance_m"] is None or
                m["handoff_xd_error_m"]<=d["terminal_position_tolerance_m"]))
    return m,rows


def plot(rows,m,path):
    import matplotlib.pyplot as plt
    if not rows:return
    t=np.array([r["elapsed_s"] for r in rows]);u=np.array([r["public_request"] for r in rows])
    fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    axs[0,0].plot(t,[r["distance_m"] for r in rows]);axs[0,0].set_ylabel("Target center distance [m]")
    axs[0,1].plot(t,[r["fuel_kg"] for r in rows]);axs[0,1].set_ylabel("Public cumulative fuel [kg]")
    axs[1,0].plot(t,[r["pointing_error_deg"] for r in rows]);axs[1,0].set_ylabel("+X pointing error [deg]")
    axs[1,1].plot(t,[r["omega_B_rad_s"] for r in rows]);axs[1,1].set_ylabel("Body angular rates [rad/s]")
    for j,label in enumerate(("x","y","z")):
        axs[2,0].step(t,u[:,j],where="post",label=label);axs[2,1].step(t,u[:,j+3],where="post",label=label)
    axs[2,0].set_ylabel("Public translation requests [m/s²]");axs[2,1].set_ylabel("Public angular requests [rad/s²]")
    for ax in axs.flat:ax.set_xlabel("Controller elapsed [s]");ax.grid(True,alpha=.3);ax.axvline(m["duration_setting_s"],linestyle=":",color="gray")
    axs[2,0].legend();axs[2,1].legend();fig.suptitle(m["label"]+" | "+m["outcome"])
    fig.savefig(path/"run_metrics.png",dpi=150);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all",action="store_true");g.add_argument("--session",type=Path)
    p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--show",action="store_true");args=p.parse_args(argv)
    folder=args.root.resolve()/"policy"/"nonlinear_handoff_staging"
    base=folder/"run_logs" if args.all else args.session
    if not base.is_absolute():base=args.root.resolve()/base
    attempts=sorted({r.parent for r in base.rglob("staging_telemetry.txt")})
    if not attempts:raise ValueError("No staging telemetry below "+str(base))
    import matplotlib
    if not args.show:matplotlib.use("Agg")
    output=folder/"analysis_results";all_metrics=[]
    for attempt in attempts:
        m,rows=metrics(attempt);all_metrics.append(m)
        session=next((a for a in attempt.parents if (a/"plan.json").is_file()),attempt.parent)
        dest=output/session.name/attempt.parent.name/attempt.name;dest.mkdir(parents=True,exist_ok=True)
        (dest/"metrics.txt").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8");plot(rows,m,dest)
        print(json.dumps({k:m.get(k) for k in ("label","outcome","handoff_pointing_deg","handoff_rate_rad_s","staging_fuel_increment_kg","fuel_total_kg")}))
    if args.all:
        columns=["attempt","label","outcome","distance_setting_m","duration_setting_s","first_policy_sim_s","handoff_pointing_deg",
            "handoff_rate_rad_s","handoff_xd_error_m","handoff_forward_velocity_m_s","staging_fuel_increment_kg","elapsed_controller_s",
            "fuel_total_kg","damage_J","final_distance_m","score_formula_estimate","measured_handoff_within_requested_tolerances"]
        with (output/"comparison.txt").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,delimiter="\t",extrasaction="ignore");writer.writeheader();writer.writerows(all_metrics)
        (output/"comparison_details.json").write_text(json.dumps(all_metrics,indent=2)+"\n",encoding="utf-8")
    if args.show:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,5))
        for m in all_metrics:
            if m.get("staging_fuel_increment_kg") is None:continue
            ax.scatter(m["staging_fuel_increment_kg"],m["handoff_pointing_deg"]);ax.annotate(m["label"],(m["staging_fuel_increment_kg"],m["handoff_pointing_deg"]),fontsize=7)
        ax.set(xlabel="Measured staging fuel [kg]",ylabel="Measured handoff pointing error [deg]");ax.grid(True,alpha=.3)
        fig.savefig(output/"comparison.png",dpi=150);plt.show();plt.close(fig)
    print("Analysis: "+str(output));return 0


if __name__=="__main__":raise SystemExit(main())
