"""Invoked only below the official Controller.py bootstrap. SDK RPCs stay on its thread."""
from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import types

from .common import (Lock, begin_attempt, check_sources, commit_attempt, digest,
                     json_hash, local_path, new_session, read_json, result,
                     split_original, trial, utc, validate, write_json)

class Finish(BaseException):
    def __init__(self, value):
        self.value = value

class Abort(BaseException):
    def __init__(self, reason, details=None):
        self.reason, self.details = reason, details

def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, allow_nan=False, default=str) + "\n")

class Context:
    """Public inputs, cooperative cancellation, actual requests and local logs.

    A task exports run(context). It calls checkpoint/sleep regularly, and returns
    a result mapping or calls finish(). No simulator reset belongs in a task.
    """
    def __init__(self, root, session, attempt, assignment, config, namespace):
        self.root, self.session_dir, self.output_dir = Path(root), Path(session), Path(attempt)
        self.assignment, self.config = assignment, config
        self.parameters = assignment["parameters"]
        self.model_path = local_path(root, config["model"]) if config["model"] else None
        self._g = namespace
        self.st, self.shuttle, self.phobos, self.target = (namespace[k] for k in ("st", "shuttle", "phobos", "target"))
        self.shuttle_frame = namespace.get("shuttleFrame")
        self.phobos_frame = namespace.get("phobosFrame")
        self.top_camera = namespace.get("topCamera")
        self.bottom_camera = namespace.get("bottomCamera")
        self._command = namespace["SetCommand"]
        self._epoch = self.st.SimGlobals.GetSimEntity().GetParam(
            self.st.VarType.timestamp, ["Time", "StartDateTAI"]).as_datetime()
        self._start_wall = time.monotonic()
        self._last_poll = self._last_state = -math.inf
        self._last_sim = None
        self._last_advance = self._start_wall
        self._ready_sim = None
        self._finished = False
        self._stop_after_task = False

    def clock(self):
        return (self.st.SimGlobals.SimClock.GetTimeNow().as_datetime()-self._epoch).total_seconds()

    def event(self, name, **data):
        append(self.output_dir / "events.txt", {"utc": utc(), "event": name, **data})

    def heartbeat(self, phase="running"):
        write_json(self.output_dir / "heartbeat.json", {
            "token": self.assignment["token"], "phase": phase, "wall_unix_s": time.time(),
            "ready": self._ready_sim is not None,
            "sim_s": self._last_sim, "pid": os.getpid()})

    def state(self):
        import numpy as np
        start = self.clock()
        frame = self.phobos.GetBodyFixedFrame()
        pv = self.shuttle.getLocVelAcc()
        if not hasattr(pv, "loc_WRT_ExprIn"):
            pv = pv.locvel()
        p = np.asarray(pv.loc_WRT_ExprIn(frame), float)
        v = np.asarray(pv.vel_WRT_ExprIn(frame), float)
        q = np.asarray(self.shuttle.getRotation().Quat_WRT(frame), float)
        w = np.asarray(self.shuttle.getAngVelocity().WRT_ExprIn(frame), float)
        goal = self.target.getLocVelAcc()
        if not hasattr(goal, "loc_WRT_ExprIn"):
            goal = goal.locvel()
        goal = np.asarray(goal.loc_WRT_ExprIn(frame), float)
        values = np.r_[p, v, q, w, goal]
        if p.shape != (3,) or v.shape != (3,) or q.shape != (4,) or w.shape != (3,) or not np.isfinite(values).all() or np.linalg.norm(q) < .5:
            raise ValueError("Invalid own-state packet")
        record = {"read_begin_sim_s": start,
                  "position_P_m": p.tolist(), "velocity_P_m_s": v.tolist(),
                  "q_PB_xyzw_reported": q.tolist(), "omega_relP_P_rad_s": w.tolist(),
                  "target_P_m": goal.tolist(),
                  "fuel_used_kg": float(self.shuttle.GetParam(self.st.VarType.double, ["Stats", "FuelUsed"])),
                  "damage_J": float(self.shuttle.GetParam(self.st.VarType.double, ["Stats", "Damage"]))}
        record["read_end_sim_s"] = self.clock()
        append(self.output_dir / "states.txt", record)
        self._last_state = time.monotonic()
        return record

    def radar(self, count=500):
        if type(count) is not int or not 0 <= count <= 500:
            raise ValueError("radar count must be 0..500")
        rows = self._g["GetNearestDebrisLocations"](count)
        data = [x.tolist() if hasattr(x, "tolist") else x for x in rows]
        append(self.output_dir / "radar.txt", {"read_sim_s": self.clock(), "observations": data})
        return rows

    def point_camera(self, camera, yaw_deg, pitch_deg):
        return self._g["PanTiltCamera"](camera, yaw_deg, pitch_deg)

    def command(self, linear, angular):
        import numpy as np
        a, alpha = np.asarray(linear, float), np.asarray(angular, float)
        if a.shape != (3,) or alpha.shape != (3,) or not np.isfinite(np.r_[a, alpha]).all():
            raise ValueError("Commands must be finite 3-vectors")
        self.checkpoint()
        before = self.clock()
        self._command(a, alpha)
        record={"send_begin_sim_s": before,
               "send_end_sim_s": self.clock(), "linear_body_request_m_s2": a.tolist(),
               "angular_body_request_rad_s2": alpha.tolist()}
        append(self.output_dir / "commands.txt", record)
        return record

    def checkpoint(self, *, sample=True):
        now = time.monotonic()
        if (self.session_dir / "STOP_NOW").exists():
            raise Abort("operator_stop")
        limit = self.config["limits"]["attempt_wall_s"]
        if limit is not None and now-self._start_wall >= limit:
            raise Abort("attempt_wall_timeout")
        if now-self._last_poll < .1:
            return
        self._last_poll = now
        t = self.clock()
        if self._last_sim is None or t > self._last_sim + 1e-9:
            self._last_advance = now
        elif t < self._last_sim - 1e-6:
            raise Abort("controller_error", "Simulation clock moved backwards")
        self._last_sim = t
        if now-self._last_advance > self.config["limits"]["clock_stall_wall_s"]:
            raise Abort("simulation_clock_stall")
        duration = self.config["limits"]["task_sim_s"]
        if duration is not None and self._ready_sim is not None and t-self._ready_sim >= duration:
            self.finish(outcome="not_assessed", reason="sim_time_limit")
        self.heartbeat()
        if sample and not self._finished and now-self._last_state >= self.config["logging"]["state_period_s"]:
            self.state()

    def sleep(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("sleep duration must be finite and nonnegative")
        end = time.monotonic() + seconds
        while True:
            self.checkpoint()
            remaining = end-time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(.1, remaining))

    def ready(self):
        deadline = time.monotonic() + self.config["limits"]["startup_wall_s"]
        first = self.clock()
        last_error = None
        while time.monotonic() < deadline:
            self.checkpoint(sample=False)
            try:
                self.state()
                if self.clock() > first:
                    self._ready_sim = self.clock()
                    self.event("ready", sim_s=self._ready_sim)
                    self.heartbeat("ready")
                    return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(.05)
        raise Abort("startup_timeout", last_error)

    def finish(self, *, outcome="success", reason="task_completed", stop_sim=False, **metrics):
        if outcome not in ("success", "failure", "not_assessed"):
            raise ValueError("Unknown task outcome")
        if type(stop_sim) is not bool:
            raise ValueError("stop_sim must be a boolean")
        json_hash(metrics)
        raise Finish(result("completed", outcome, reason, stop_sim_requested=stop_sim, metrics=metrics))

    def request_stop_after_task(self):
        """Request native stop after result commit, including unexpected task errors."""
        self._stop_after_task = True

class OriginalTime:
    """Local alias for the preserved sample's time module; no global monkeypatch."""
    def __init__(self, context):
        self._ctx = context
    def sleep(self, seconds):
        self._ctx.sleep(seconds)
    def __getattr__(self, name):
        return getattr(time, name)

def execute_entry(ctx, namespace):
    path = local_path(ctx.root, ctx.config["entry"], must_exist=True)
    if path == ctx.root / "OriginalController.py":
        prefix, tail = split_original(path.read_bytes())
        active_prefix, _ = split_original((ctx.root / "Controller.py").read_bytes())
        if prefix != active_prefix:
            raise ValueError("Original/active protected bootstrap mismatch")
        scope = dict(namespace)
        scope.update(__name__="__main__", __file__=str(path), context=ctx,
                     SetCommand=ctx.command, GetNearestDebrisLocations=ctx.radar,
                     time=OriginalTime(ctx))
        # Only editable statements run; connection/bootstrap/helpers are already live.
        exec(compile(tail, str(path), "exec"), scope, scope)
        return None
    name = "_stu4_task_" + ctx.assignment["token"]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Allow task-local imports; all dependency paths remain in the cloned tree.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
        if not callable(getattr(module, "run", None)):
            raise ValueError("Entry file must export run(context)")
        return module.run(ctx)
    finally:
        sys.path.pop(0)
        sys.modules.pop(name, None)

def _manual_assignment(root):
    c = validate(root, read_json(root / "active_run.json"))
    if c["mode"] != "manual":
        raise ValueError("Automatic selection requires its terminal supervisor; no manual fallback")
    pointer = root / "sim_core" / "state" / "active_session.json"
    session = None
    if c["manual_session"] == "continue" and pointer.exists():
        session = local_path(root, read_json(pointer)["session"])
        plan, ledger = read_json(session / "plan.json"), read_json(session / "ledger.json")
        if plan["config_hash"] != json_hash(c):
            raise ValueError("Manual selection changed; use manual --new to start a new session")
        if ledger["state"] == "complete":
            raise ValueError("Manual set is complete; use manual --new for another set")
        if ledger["state"] != "pending":
            raise ValueError("Manual session is interrupted/paused; use resume or manual --new")
    if session is None:
        session = new_session(root, c)
        write_json(pointer, {"session": session.relative_to(root).as_posix()})
    plan, ledger = read_json(session / "plan.json"), read_json(session / "ledger.json")
    check_sources(root, plan)
    attempt, assignment = begin_attempt(session, c, ledger)
    return session, attempt, assignment, c, ledger

def dispatch(namespace, root):
    root = Path(root).resolve()
    # A second controller cannot steal an active assignment or create another run.
    with Lock(root / "sim_core" / "state" / "controller.lock"):
        token = os.environ.get("STU4_ATTEMPT_TOKEN")
        supervised = bool(token)
        if supervised:
            session = local_path(root, os.environ["STU4_SESSION"])
            attempt = local_path(session, os.environ["STU4_ATTEMPT"])
            assignment = read_json(attempt / "assignment.json")
            if assignment["token"] != token:
                raise ValueError("Supervisor assignment token mismatch")
            plan = read_json(session / "plan.json")
            c = plan["config"]
            if c["mode"] != "automatic":
                raise ValueError("Supervisor assignment is not automatic")
            check_sources(root, plan)
            ledger = None  # the supervisor is the sole ledger writer in automatic mode
        else:
            # Prevent an unrelated GUI from running while a supervisor owns this clone.
            with Lock(root / "sim_core" / "state" / "supervisor.lock"):
                session, attempt, assignment, c, ledger = _manual_assignment(root)
        ctx = None
        should_stop = supervised
        try:
            ctx = Context(root, session, attempt, assignment, c, namespace)
            ctx.heartbeat("starting")
            ctx.ready()
            ctx.event("entry_started", entry=c["entry"], parameters=assignment["parameters"])
            ctx.event("execution_environment", python=sys.version, executable=sys.executable)
            returned = execute_entry(ctx, namespace)
            if returned is not None and not isinstance(returned, dict):
                raise ValueError("run(context) returns a metrics dict, None, or calls finish()")
            json_hash(returned)
            value = result("completed", "not_assessed", "entry_returned", metrics=returned or {})
        except Finish as exc:
            value = exc.value
        except Abort as exc:
            value = result("interrupted" if exc.reason == "operator_stop" else "failed",
                           "not_assessed", exc.reason, details=exc.details)
            should_stop = should_stop or exc.reason == "operator_stop"
        except (KeyboardInterrupt, SystemExit) as exc:
            value = result("interrupted", "not_assessed", "operator_stop", details=str(exc))
            should_stop = True
        except BaseException as exc:
            # Preserve exact diagnostics; do not classify arbitrary bugs as retryable I/O.
            disconnected = isinstance(exc, (ConnectionError, EOFError, BrokenPipeError))
            value = result("failed", "not_assessed", "backend_disconnect" if disconnected else "controller_error",
                           details=traceback.format_exc())
        should_stop = should_stop or bool(value.get("stop_sim_requested")) or bool(ctx and ctx._stop_after_task)
        if ctx:
            ctx._finished = True
            try:
                import numpy as np
                ctx._command(np.zeros(3), np.zeros(3))
                ctx.event("final_zero_request", note="Zero thrust is not braking")
            except Exception as exc:
                value["final_zero_error"] = str(exc)
            value["sim_s"] = ctx._last_sim
        value.update(token=assignment["token"], trial_id=assignment["trial_id"], attempt_id=assignment["attempt_id"])
        for name in ("commands.txt", "states.txt", "radar.txt", "events.txt"):
            path = attempt / name
            if path.exists():
                with path.open("ab") as log:
                    os.fsync(log.fileno())
        write_json(attempt / "result.json", value)
        if not supervised:
            commit_attempt(c, ledger, value)
            if (session / "PAUSE").exists() and ledger["state"] == "pending":
                ledger.update(state="paused", reason="pause_after_attempt")
            write_json(session / "ledger.json", ledger)
        print("[STU4] " + json.dumps({"session": str(session), "result": value}), flush=True)
        if should_stop:
            namespace["st"].stop_sim()
            # Remain connected until native shutdown; -quitOnStop lets supervisor see exit.
            while True:
                time.sleep(.2)
                try:
                    namespace["st"].SimGlobals.SimClock.GetTimeNow()
                except Exception:
                    return
        # Manual mode never auto-restarts/stops. Keep the native System alive after return.
        print("[STU4] Manual attempt finished; close the SpaceCRAFT window when ready.", flush=True)
        while True:
            time.sleep(.2)
            if (session / "STOP_NOW").exists():
                namespace["st"].stop_sim()
                return
            try:
                namespace["st"].SimGlobals.SimClock.GetTimeNow()
            except Exception:
                return
