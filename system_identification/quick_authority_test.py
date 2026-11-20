"""Manual authority test plus offline reanalysis. No model or simulator edits."""
import csv
import json
from pathlib import Path

import numpy as np

MODE = "x"               # "x" or "z"
HIGH = 10.0              # Raw public body acceleration request [m/s^2]
Z_BURN_S = 15.0          # Z-mode burn duration
X_LIFT_S = 5.0           # X-mode initial +Z burn
X_BURN_S = 10.0          # X-mode +X burn
COAST_S = 5.0            # Zero-command observation after either test
SAMPLE_SLEEP_S = 0.05
ANALYSIS_WINDOW_S = 2.0  # Early/late windows; shortened for short phases


def rotation_P_to_B(q):
    """Project convention: passive P->B quaternion, xyzw."""
    q = np.asarray(q, dtype=float)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def make_schedule(mode):
    if mode == "z":
        stages = [("z", Z_BURN_S, [0, 0, HIGH])]
    elif mode == "x":
        stages = [
            ("z", X_LIFT_S, [0, 0, HIGH]),
            ("x", X_BURN_S, [HIGH, 0, 0]),
        ]
    else:
        raise ValueError('MODE must be "x" or "z"')

    stages.append(("coast", COAST_S, [0, 0, 0]))
    schedule, clock = [], 0.0

    for phase, duration, command in stages:
        if duration < 0:
            raise ValueError("Durations cannot be negative")
        if duration == 0:
            continue
        schedule.append(dict(
            phase=phase,
            start_s=clock,
            end_s=clock+duration,
            command_B=command,
        ))
        clock += duration

    if not schedule:
        raise ValueError("The test duration must be positive")

    return schedule


def analyze_samples(samples, mode, high):
    # Segment boundaries come from recorded phases and commands.
    segments = []
    previous_key = None

    for sample in samples:
        key = (sample["phase"], tuple(sample["u"]))
        if key != previous_key:
            segments.append([])
        segments[-1].append(sample)
        previous_key = key

    # Background estimate: latter half of the final zero-command coast.
    background, background_window = None, None
    coasts = [
        seg for seg in segments
        if seg[0]["phase"] == "coast" and not np.any(seg[0]["u"])
    ]

    if coasts:
        seg = coasts[-1]
        middle = (seg[0]["t"] + seg[-1]["t"])/2
        tail = [s for s in seg if s["t"] >= middle]

        if len(tail) >= 2:
            t = np.array([s["t"] for s in tail])
            v = np.array([s["v"] for s in tail])

            if np.ptp(t) > 0:
                A = np.column_stack((t-t.mean(), np.ones(len(t))))
                background = np.linalg.lstsq(A, v, rcond=None)[0][0]
                background_window = [float(t[0]), float(t[-1])]

    def measure(seg, lo, hi):
        duration = 0.0
        net = np.zeros(3)
        corrected = np.zeros(3)

        for previous, current in zip(seg[:-1], seg[1:]):
            t0, t1 = previous["t"], current["t"]
            dt = t1-t0
            overlap = max(0.0, min(t1, hi)-max(t0, lo))

            if dt <= 0 or overlap <= 0:
                continue

            q0, q1 = previous["q"], current["q"]
            q0 = q0/np.linalg.norm(q0)
            q1 = q1/np.linalg.norm(q1)

            if np.dot(q0, q1) < 0:
                q1 = -q1

            B = rotation_P_to_B(q0+q1)

            # Differentiate P-frame velocity BEFORE rotating the increment.
            aP = (current["v"]-previous["v"])/dt
            net += overlap*(B @ aP)

            if background is not None:
                corrected += overlap*(B @ (aP-background))

            duration += overlap

        if duration <= 0:
            return None

        return dict(
            window_s=[float(lo), float(hi)],
            measurement_duration_s=float(duration),
            net_mean_acceleration_B_m_s2=(net/duration).tolist(),
            coast_subtracted_mean_acceleration_B_m_s2=(
                None if background is None
                else (corrected/duration).tolist()
            ),
        )

    result = dict(
        analysis_schema_version=2,
        mode=mode,
        high_public_request_m_s2=float(high),
        background_acceleration_P_m_s2=(
            None if background is None else background.tolist()
        ),
        background_source_window_s=background_window,
        phases={},
    )

    for number, seg in enumerate(segments, 1):
        phase = seg[0]["phase"]
        start, end = seg[0]["t"], seg[-1]["t"]
        width = min(ANALYSIS_WINDOW_S, (end-start)/2)

        key = (
            phase if phase not in result["phases"]
            else f"{phase}_{number}"
        )

        result["phases"][key] = dict(
            command_B_m_s2=np.asarray(seg[0]["u"]).tolist(),
            recorded_start_s=float(start),
            recorded_end_s=float(end),
            comparison_window_s=float(width),
            whole_phase=measure(seg, start, end),
            early=measure(seg, start, start+width),
            late=measure(seg, end-width, end),
        )

    return result


def print_summary(summary, path):
    print(
        f"\n[AUTHORITY TEST] mode={summary['mode']} | {path}",
        flush=True,
    )
    print(
        "Background P:",
        summary["background_acceleration_P_m_s2"],
        flush=True,
    )

    for phase, data in summary["phases"].items():
        print(
            f"{phase}: {data['recorded_start_s']:.3f}"
            f" to {data['recorded_end_s']:.3f} s",
            flush=True,
        )

        for name in ("whole_phase", "early", "late"):
            value = data[name]
            if value is not None:
                print(
                    f"  {name} {value['window_s']}: "
                    f"net B={value['net_mean_acceleration_B_m_s2']}; "
                    "coast-subtracted B="
                    f"{value['coast_subtracted_mean_acceleration_B_m_s2']}",
                    flush=True,
                )


def run(context):
    mode = MODE.lower()
    schedule = make_schedule(mode)
    output = Path(context.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    samples, zero = [], np.zeros(3)
    P = context.phobos_frame
    start = context.clock()

    try:
        with (output/"authority_measurements.txt").open(
            "w", encoding="utf-8", newline="", buffering=1
        ) as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow([
                "elapsed_sim_s", "sim_s", "read_span_s", "phase",
                "request_B_x", "request_B_y", "request_B_z",
                "position_P_x", "position_P_y", "position_P_z",
                "velocity_P_x", "velocity_P_y", "velocity_P_z",
                "q_PB_x", "q_PB_y", "q_PB_z", "q_PB_w",
            ])

            while True:
                elapsed = context.clock()-start
                if elapsed >= schedule[-1]["end_s"]:
                    break

                stage = next(
                    s for s in schedule if elapsed < s["end_s"]
                )
                command = np.asarray(
                    stage["command_B"], dtype=float
                )
                context.command(command, zero)

                before = context.clock()
                pv = context.shuttle.getLocVelAcc()
                if not hasattr(pv, "vel_WRT_ExprIn"):
                    pv = pv.locvel()

                position = np.asarray(
                    pv.loc_WRT_ExprIn(P), dtype=float
                )
                velocity = np.asarray(
                    pv.vel_WRT_ExprIn(P), dtype=float
                )
                q = np.asarray(
                    context.shuttle.getRotation().Quat_WRT(P),
                    dtype=float,
                )

                after = context.clock()
                timestamp = (before+after)/2

                samples.append(dict(
                    t=timestamp-start,
                    phase=stage["phase"],
                    u=command.copy(),
                    v=velocity.copy(),
                    q=q.copy(),
                ))

                writer.writerow([
                    timestamp-start, timestamp, after-before,
                    stage["phase"], *command,
                    *position, *velocity, *q,
                ])
                context.sleep(SAMPLE_SLEEP_S)

    finally:
        context.command(zero, zero)

    summary = analyze_samples(samples, mode, HIGH)
    summary["planned_schedule"] = schedule
    summary["damage_J"] = float(context.shuttle.GetParam(
        context.st.VarType.double, ["Stats", "Damage"]
    ))

    path = output/"authority_summary.txt"
    path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print_summary(summary, path)

    context.finish(
        outcome="success",
        reason="quick_authority_test_complete",
        stop_sim=False,
        authority_test=summary,
    )


def analyze_file(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    if not rows:
        raise ValueError(f"No recorded measurements: {path}")

    samples = [dict(
        t=float(r["elapsed_sim_s"]),
        phase=r["phase"],
        u=np.array([
            float(r[f"request_B_{a}"]) for a in "xyz"
        ]),
        v=np.array([
            float(r[f"velocity_P_{a}"]) for a in "xyz"
        ]),
        q=np.array([
            float(r[f"q_PB_{a}"]) for a in "xyzw"
        ]),
    ) for r in rows]

    mode = (
        "x" if any(s["phase"] == "x" for s in samples) else "z"
    )
    high = max(
        float(np.max(np.abs(s["u"]))) for s in samples
    )
    summary = analyze_samples(samples, mode, high)

    destination = path.with_name("authority_summary.txt")
    if destination.is_file():
        old = json.loads(
            destination.read_text(encoding="utf-8-sig")
        )
        for key in ("damage_J", "planned_schedule"):
            if key in old:
                summary[key] = old[key]

    destination.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print_summary(summary, destination)
    return summary


if __name__ == "__main__":
    # Offline only. This never launches or connects to a simulator.
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python quick_authority_test.py "
            "SESSION_DIR_OR_RAW_FILE"
        )

    source = Path(sys.argv[1])
    paths = (
        [source] if source.is_file()
        else sorted(source.rglob("authority_measurements.txt"))
    )

    if not paths:
        raise FileNotFoundError(
            f"No authority measurements under {source}"
        )

    for path in paths:
        analyze_file(path)