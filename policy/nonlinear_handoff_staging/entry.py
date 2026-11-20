"""Existing run(context) interface. Public SDK reads/writes stay on this thread."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from policy.nonlinear_handoff_staging.flight import Policy,packet_row,causal_state
from policy.nonlinear_handoff_staging.accounting import ARRIVAL_RADIUS_M,DAMAGE_THRESHOLD_J,score_estimate


def run(context):
    if context.parameters:raise ValueError("Tune this policy's settings.json; then run precompute.py --select")
    folder=Path(__file__).resolve().parent;out=Path(context.output_dir);out.mkdir(parents=True,exist_ok=True)
    print("[NONLINEAR STAGING] Loading verified offline trajectory",flush=True)
    policy=Policy(folder,context.model_path or Path(context.root)/"nominal_model.json")
    print("[NONLINEAR STAGING] Reference status: "+policy.manifest.get("candidate_status","unspecified"),flush=True)
    (out/"settings_used.json").write_text(json.dumps(policy.s,indent=2)+"\n",encoding="utf-8")
    (out/"reference_used.json").write_text(json.dumps(policy.manifest,indent=2)+"\n",encoding="utf-8")
    t0=None;first=last=None;last_stage=None;previous=np.zeros(6);times=[];rows=[]
    next_status=0.;handoff=None;count=0;reason="interrupted";metrics={}
    log=(out/"staging_telemetry.txt").open("w",encoding="utf-8",buffering=1)
    try:
        while True:
            wall=time.monotonic();context.checkpoint(sample=False)
            packet=context.state();now=float(packet["read_end_sim_s"]);last=packet
            times.append(now);rows.append(packet_row(packet))
            while len(times)>3 and times[1]<now-1.:times.pop(0);rows.pop(0)
            x=causal_state(times,rows,policy.parameters)
            if t0 is None:t0=float(context.clock());first=packet
            elapsed=max(0.,float(context.clock())-t0)
            distance=float(np.linalg.norm(np.asarray(packet["position_P_m"])-packet["target_P_m"]))
            arrived=distance<=ARRIVAL_RADIUS_M
            limit=policy.s["execution"]["test_duration_s"]
            if arrived or (limit is not None and elapsed>=limit):
                reason=("observed_arrival" if packet["damage_J"]<=DAMAGE_THRESHOLD_J else "observed_arrival_after_damage_limit") if arrived else "configured_test_duration"
                metrics=dict(sim_s=now,elapsed_controller_s=elapsed,distance_m=distance,
                    fuel_kg=packet["fuel_used_kg"],damage_J=packet["damage_J"],
                    score_formula_estimate=score_estimate(now,packet["fuel_used_kg"],packet["damage_J"]) if reason=="observed_arrival" else None)
                break
            u,detail=policy.control(elapsed,x,previous)
            receipt=context.command(u[:3],u[3:]);previous=u;count+=1
            if handoff is None and elapsed>=policy.duration:
                handoff=dict(elapsed_s=elapsed,sim_s=now,**detail,fuel_kg=packet["fuel_used_kg"],damage_J=packet["damage_J"],
                    position_P_m=packet["position_P_m"],velocity_P_m_s=x[3:6].tolist())
                context.event("measured_staging_handoff",**handoff)
            if detail["stage"]!=last_stage:
                context.event("staging_mode",elapsed_s=elapsed,sim_s=now,**detail)
                last_stage=detail["stage"]
            record=dict(elapsed_s=elapsed,sim_s=now,position_P_m=packet["position_P_m"],velocity_P_m_s=packet["velocity_P_m_s"],
                q_PB_xyzw=packet["q_PB_xyzw_reported"],omega_B_rad_s=x[10:13].tolist(),target_P_m=packet["target_P_m"],
                distance_m=distance,fuel_kg=packet["fuel_used_kg"],damage_J=packet["damage_J"],public_request=u.tolist(),**detail,
                read_span_s=packet["read_end_sim_s"]-packet["read_begin_sim_s"],**receipt)
            log.write(json.dumps(record,allow_nan=False)+"\n")
            if elapsed>=next_status:
                print("[NONLINEAR STAGING] t=%.2f %s angle=%.3f deg rate=%.4f fuel=%.3f kg" %
                    (elapsed,detail["stage"],detail["pointing_error_deg"],detail["rate_norm_rad_s"],packet["fuel_used_kg"]),flush=True)
                next_status=elapsed+policy.s["execution"]["status_period_s"]
            context.sleep(max(0.,policy.s["feedback"]["period_s"]-(time.monotonic()-wall)))
    finally:
        log.close()
        try:context.command(np.zeros(3),np.zeros(3))
        finally:
            summary=dict(reason=reason,commands=count,handoff=handoff,metrics=metrics,first_state=first,last_state=last,
                official_score_captured=False,stop_sim_requested=False)
            (out/"staging_summary.txt").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    context.finish(outcome="success" if reason=="observed_arrival" else ("failure" if reason.endswith("damage_limit") else "not_assessed"),
        reason=reason,stop_sim=False,official_score_captured=False,handoff=handoff,**metrics)
