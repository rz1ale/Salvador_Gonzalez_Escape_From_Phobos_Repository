"""Existing runner entry. No terminal connection, restart, or native stop."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from policy.naive_handoff.control import Policy, sha256
from policy.naive_handoff.accounting import ARRIVAL_RADIUS_M, DAMAGE_THRESHOLD_J, score_estimate
from policy.naive_handoff import VERSION


def run(context):
    if context.parameters:
        raise ValueError("Tune policy/naive_handoff/settings.json, then precompute; no hidden per-run overrides")
    folder = Path(__file__).resolve().parent
    model_path = context.model_path or Path(context.root)/"nominal_model.json"
    policy = Policy(folder,model_path)
    output = Path(context.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    (output/"settings_used.json").write_text(json.dumps(policy.settings,indent=2)+"\n",encoding="utf-8")
    (output/"reference_used.json").write_text(json.dumps(policy.manifest,indent=2)+"\n",encoding="utf-8")
    first = last = None
    t0 = None
    previous_stage = None
    count = 0
    next_status = 0.0
    reason = "interrupted"
    transitions = []
    result = {}
    log = (output/"naive_handoff_telemetry.txt").open("w",encoding="utf-8",buffering=1)
    try:
        while True:
            wall = time.monotonic()
            context.checkpoint(sample=False)
            packet = context.state()
            now = float(packet["read_end_sim_s"])
            if t0 is None:
                first = packet
                t0 = float(context.clock())
                context.event("naive_handoff_start",version=VERSION,sim_s=t0,
                    settings=policy.settings,model_sha256=sha256(model_path),
                    obstacle_avoidance=False,online_planner=False,stop_sim_requested=False)
            elapsed = max(0.0,now-t0)
            distance = float(np.linalg.norm(np.asarray(packet["position_P_m"])-packet["target_P_m"]))
            last = packet
            if distance <= ARRIVAL_RADIUS_M:
                reason = "observed_arrival" if packet["damage_J"]<=DAMAGE_THRESHOLD_J else "observed_arrival_after_damage_limit"
                result = dict(distance_m=distance,sim_s=now,elapsed_controller_s=elapsed,
                    fuel_kg=packet["fuel_used_kg"],damage_J=packet["damage_J"],
                    score_formula_estimate=score_estimate(now,packet["fuel_used_kg"],packet["damage_J"]) if reason=="observed_arrival" else None,
                    official_score_captured=False)
                break
            duration = policy.settings["test_duration_s"]
            if duration is not None and elapsed >= duration:
                reason = "configured_test_duration"
                result = dict(distance_m=distance,sim_s=now,elapsed_controller_s=elapsed,
                    fuel_kg=packet["fuel_used_kg"],damage_J=packet["damage_J"],
                    score_formula_estimate=None,official_score_captured=False)
                break
            # Use current SIMULATION phase; skip elapsed knots after a slow SDK read.
            elapsed = max(0.0,float(context.clock())-t0)
            u,detail = policy.command(elapsed,packet)
            receipt = context.command(u[:3],u[3:])
            count += 1
            row = dict(elapsed_s=elapsed,sim_s=now,position_P_m=packet["position_P_m"],
                velocity_P_m_s=packet["velocity_P_m_s"],q_PB_xyzw=packet["q_PB_xyzw_reported"],
                target_P_m=packet["target_P_m"],distance_m=distance,fuel_kg=packet["fuel_used_kg"],
                damage_J=packet["damage_J"],public_request=u.tolist(),**detail,
                read_span_s=packet["read_end_sim_s"]-packet["read_begin_sim_s"],
                send_begin_sim_s=receipt["send_begin_sim_s"],send_end_sim_s=receipt["send_end_sim_s"])
            log.write(json.dumps(row,allow_nan=False)+"\n")
            if detail["stage"] != previous_stage:
                transition = dict(stage=detail["stage"],elapsed_s=elapsed,sim_s=now,
                    fuel_kg=packet["fuel_used_kg"],pointing_error_deg=detail["pointing_error_deg"],
                    omega_norm_rad_s=float(np.linalg.norm(detail["omega_B_rad_s"])),
                    speed_m_s=float(np.linalg.norm(packet["velocity_P_m_s"])))
                transitions.append(transition)
                context.event("naive_handoff_stage",**transition)
                print("[NAIVE HANDOFF] "+json.dumps(transition),flush=True)
                previous_stage = detail["stage"]
            if elapsed >= next_status:
                print("[NAIVE HANDOFF] t=%.2f stage=%s distance=%.2f m fuel=%.3f kg pointing=%.2f deg" %
                    (elapsed,detail["stage"],distance,packet["fuel_used_kg"],detail["pointing_error_deg"]),flush=True)
                next_status = elapsed+policy.settings["status_period_s"]
            context.sleep(max(0.0,policy.settings["control_period_s"]-(time.monotonic()-wall)))
    finally:
        log.close()
        # Zero requests do not brake the vehicle. Native simulation remains open.
        try:
            context.command(np.zeros(3),np.zeros(3))
        finally:
            summary = dict(reason=reason,commands=count,stage_transitions=transitions,metrics=result,
                first_state=first,last_state=last,official_score_captured=False,stop_sim_requested=False)
            (output/"naive_handoff_summary.txt").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    outcome="success" if reason=="observed_arrival" else ("failure" if reason=="observed_arrival_after_damage_limit" else "not_assessed")
    context.finish(outcome=outcome,
        reason=reason,stop_sim=False,**result)
