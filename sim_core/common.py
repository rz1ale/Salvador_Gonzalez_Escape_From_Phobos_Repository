"""Configuration, atomic state, process locks and deterministic trial scheduling."""
from __future__ import annotations
import ast
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

OFFICIAL_ASSET = "Local/Repos/STU4-Sim/Sims/COMPETITION_SIM.simconf"
DISPATCH_MARK = "# STU4_SHARED_RUNNER_V1"
FAILURES = {"startup_timeout", "backend_disconnect", "simulation_clock_stall",
            "controller_unresponsive", "attempt_wall_timeout", "process_exit",
            "shutdown_timeout", "controller_error", "task_failure"}

def default_config(entry="OriginalController.py", mode="manual"):
    """A fresh single-file selection must not inherit another task's case parameters."""
    return {"schema_version":1,"mode":mode,"manual_session":"new_on_open","entry":entry,
        "parameters":{},"model":None,"dependencies":[],"output_root":"run_logs",
        "schedule":{"cases":[{"id":"default","parameters":{}}],"repetitions_per_case":1,
                    "repeat_set":{"mode":"fixed","count":1},"order":"listed"},
        "retry":{"max_retries_per_trial":0,"retry_on":["startup_timeout","backend_disconnect","simulation_clock_stall"],"on_exhausted":"pause"},
        "on_task_failure":"continue","limits":{"startup_wall_s":150,"heartbeat_wall_s":45,
            "clock_stall_wall_s":90,"attempt_wall_s":900 if mode=="automatic" else None,"shutdown_wall_s":30,"task_sim_s":None},
        "logging":{"state_period_s":.2}}

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def read_json(path):
    # Windows rename/read sharing may transiently collide. Never delete a destination.
    for i in range(20):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except PermissionError:
            if i == 19:
                raise
            time.sleep(.02)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        for i in range(30):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if i == 29:
                    raise
                time.sleep(.02)
    finally:
        temp.unlink(missing_ok=True)

def local_path(root, value, *, must_exist=False):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Paths in JSON use nonempty solution-relative forward slashes")
    root = Path(root).resolve()
    p = (root / value).resolve()
    if Path(value).is_absolute() or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Absolute dependency/output paths are not supported")
    try:
        p.relative_to(root)
    except ValueError:
        raise ValueError("Path escapes the cloned solution: " + value) from None
    if must_exist and not p.is_file():
        raise ValueError("Missing file: " + str(p))
    return p

def keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(label + " has unknown fields: " + ", ".join(sorted(extra)))

def positive_int(value, label, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(label + " must be " + ("nonnegative" if zero else "positive") + " integer")

def validate(root, raw):
    c = copy.deepcopy(raw)
    keys(c, {"schema_version", "mode", "manual_session", "entry", "parameters", "model", "dependencies", "output_root",
             "schedule", "retry", "on_task_failure", "limits", "logging"}, "configuration")
    if type(c.get("schema_version")) is not int or c["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema_version")
    if c.get("mode") not in ("manual", "automatic"):
        raise ValueError("mode must be manual or automatic")
    c.setdefault("manual_session", "new_on_open")
    if c["manual_session"] not in ("new_on_open", "continue"):
        raise ValueError("manual_session must be new_on_open or continue")
    entry = local_path(root, c.get("entry"), must_exist=True)
    if entry.suffix != ".py" or entry == Path(root).resolve() / "Controller.py":
        raise ValueError("entry must be a .py file other than active Controller.py")
    if not isinstance(c.get("parameters", {}), dict):
        raise ValueError("parameters must be an object")
    c.setdefault("parameters", {})
    if c.get("model") is not None:
        local_path(root, c["model"], must_exist=True)
    c.setdefault("model", None)
    c.setdefault("dependencies", [])
    if not isinstance(c["dependencies"], list):
        raise ValueError("dependencies must be a list of solution-relative files")
    for dependency in c["dependencies"]:
        local_path(root, dependency, must_exist=True)
    c.setdefault("output_root", "run_logs")
    output = local_path(root, c["output_root"])
    if output == Path(root).resolve() or output.is_file():
        raise ValueError("output_root must be a subdirectory")
    s = c.get("schedule")
    keys(s, {"cases", "repetitions_per_case", "repeat_set", "order"}, "schedule")
    if s.get("order", "listed") != "listed":
        raise ValueError("Only explicitly listed order is implemented")
    s.setdefault("order", "listed")
    positive_int(s.get("repetitions_per_case", 1), "repetitions_per_case")
    s.setdefault("repetitions_per_case", 1)
    cases = s.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("schedule.cases must be a nonempty list")
    ids = set()
    for case in cases:
        keys(case, {"id", "parameters", "repetitions"}, "case")
        name = case.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name in ids:
            raise ValueError("Case IDs must be unique safe identifiers")
        ids.add(name)
        if not isinstance(case.get("parameters", {}), dict):
            raise ValueError("case.parameters must be an object")
        case.setdefault("parameters", {})
        positive_int(case.get("repetitions", s["repetitions_per_case"]), "case.repetitions")
    rep = s.get("repeat_set")
    keys(rep, {"mode", "count"}, "repeat_set")
    if rep.get("mode") == "fixed":
        positive_int(rep.get("count"), "repeat_set.count")
    elif rep.get("mode") != "until_stopped" or "count" in rep:
        raise ValueError("repeat_set is fixed/count or until_stopped without count")
    if c["mode"] == "manual" and c["manual_session"] == "new_on_open":
        if len(cases) != 1 or cases[0].get("repetitions", s["repetitions_per_case"]) != 1 or rep != {"mode": "fixed", "count": 1}:
            raise ValueError("Manual sets/repetitions require manual_session=continue; use manual --new")
    retry = c.get("retry")
    keys(retry, {"max_retries_per_trial", "retry_on", "on_exhausted"}, "retry")
    positive_int(retry.get("max_retries_per_trial"), "max_retries_per_trial", zero=True)
    if not isinstance(retry.get("retry_on"), list) or any(x not in FAILURES for x in retry["retry_on"]):
        raise ValueError("Unknown retry_on reason; operator_stop is never retryable")
    if retry.get("on_exhausted") not in ("pause", "continue"):
        raise ValueError("on_exhausted must be pause or continue")
    if c.get("on_task_failure") not in ("pause", "continue"):
        raise ValueError("on_task_failure must be pause or continue")
    limits = c.setdefault("limits", {})
    keys(limits, {"startup_wall_s", "heartbeat_wall_s", "clock_stall_wall_s",
                  "attempt_wall_s", "shutdown_wall_s", "task_sim_s"}, "limits")
    defaults = dict(startup_wall_s=150, heartbeat_wall_s=45, clock_stall_wall_s=90,
                    attempt_wall_s=None, shutdown_wall_s=30, task_sim_s=None)
    for k, v in defaults.items():
        limits.setdefault(k, v)
        v = limits[k]
        if v is None and k in ("attempt_wall_s", "task_sim_s"):
            continue
        if type(v) not in (float, int) or not math.isfinite(v) or v <= 0:
            raise ValueError("Invalid positive seconds: limits." + k)
    if c["mode"] == "automatic" and limits["attempt_wall_s"] is None:
        raise ValueError("Automatic mode requires a finite attempt_wall_s")
    logging = c.setdefault("logging", {"state_period_s": .2})
    keys(logging, {"state_period_s"}, "logging")
    v = logging.setdefault("state_period_s", .2)
    if type(v) not in (float, int) or not math.isfinite(v) or v < .02:
        raise ValueError("logging.state_period_s must be >= .02 seconds")
    # Confirm JSON-serializable parameters and reject nonfinite constants.
    json_hash(c)
    return c

class Lock:
    """Kernel-released lock; an old lock filename never blocks recovery."""
    def __init__(self, path):
        self.path, self.f = Path(path), None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a+b")
        self.f.seek(0, 2)
        if not self.f.tell():
            self.f.write(b"0")
            self.f.flush()
        self.f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            self.f = None
            raise RuntimeError("Another owner is active: " + str(self.path)) from None
        return self
    def __exit__(self, *args):
        if self.f:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            self.f.close()
            self.f = None

def split_original(data):
    """Return byte-exact protected prefix and padded editable source."""
    text = data.decode("utf-8-sig")
    lines = text.splitlines(keepends=True)
    markers = [i for i, line in enumerate(lines) if "DON'T CHANGE ANY OF THE ABOVE" in line]
    if len(markers) != 1:
        raise ValueError("Expected one official ABOVE protection marker; no file changed")
    end = markers[0] + 1
    if end >= len(lines) or not re.fullmatch(r"\s*#{10,}\s*", lines[end]):
        raise ValueError("Missing closing protection banner; no file changed")
    end += 1
    # Byte slicing rather than re-encoding preserves BOM and line endings.
    offset = len("".join(lines[:end]).encode("utf-8")) + (3 if data.startswith(b"\xef\xbb\xbf") else 0)
    prefix = data[:offset]
    tail = "\n" * end + "".join(lines[end:])
    ast.parse(tail)
    return prefix, tail

def install(root):
    root = Path(root).resolve()
    original, controller = root / "OriginalController.py", root / "Controller.py"
    if original.exists():
        data = original.read_bytes()
    elif controller.exists():
        data = controller.read_bytes()
        if DISPATCH_MARK.encode() in data:
            raise ValueError("OriginalController.py missing; cannot reconstruct the baseline")
    else:
        raise ValueError("Place this package in the fresh STU4-Solution root")
    prefix, _ = split_original(data)
    if controller.exists():
        current = controller.read_bytes()
        current_prefix, _ = split_original(current)
        if current_prefix != prefix:
            raise ValueError("Protected sections differ; refusing to replace Controller.py")
        if current != data and DISPATCH_MARK.encode() not in current:
            raise ValueError("Controller.py has other user changes; preserve/review them before installing")
    body = ("\n\n" + DISPATCH_MARK + "\n"
            "from pathlib import Path as _RunnerPath\n"
            "_runner_root = _RunnerPath(__file__).resolve().parent\n"
            "if str(_runner_root) not in sys.path:\n"
            "    sys.path.insert(0, str(_runner_root))\n"
            "from sim_core.runtime import dispatch as _run_selected\n"
            "_run_selected(globals(), _runner_root)\n")
    new = prefix + body.encode()
    compile(new, str(controller), "exec")
    if not original.exists():
        with original.open("xb") as f:
            f.write(data)
    temp = controller.with_suffix(".runner.tmp")
    temp.write_bytes(new)
    os.replace(temp, controller)
    return {"original_sha256": digest(original), "protected_sha256": hashlib.sha256(prefix).hexdigest()}

def snapshot_sources(root, c):
    root = Path(root)
    paths = {c["entry"], "Controller.py", "OriginalController.py", "run_simulation.py"}
    paths.update(p.relative_to(root).as_posix() for p in (root / "sim_core").rglob("*.py"))
    if c["model"]:
        paths.add(c["model"])
    paths.update(c["dependencies"])
    return {p: digest(local_path(root, p, must_exist=True)) for p in sorted(paths)}

def check_sources(root, plan):
    for p, h in plan["source_hashes"].items():
        if digest(local_path(root, p, must_exist=True)) != h:
            raise ValueError("Session source/model changed: " + p + "; start a new session")
    official = Path(root).parent / "STU4-Sim" / "Sims" / "COMPETITION_SIM.simconf"
    if plan.get("official_config_sha256") and digest(official) != plan["official_config_sha256"]:
        raise ValueError("Official configuration changed; start/revalidate a new session")

def new_session(root, c):
    root = Path(root).resolve()
    official = root.parent / "STU4-Sim" / "Sims" / "COMPETITION_SIM.simconf"
    if not official.is_file():
        raise ValueError("Missing sibling official COMPETITION_SIM.simconf")
    name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    session = local_path(root, c["output_root"]) / name
    session.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": 1, "created_utc": utc(), "config": c,
            "python": {"version": sys.version, "executable": sys.executable},
            "config_hash": json_hash(c), "source_hashes": snapshot_sources(root, c),
            "official_asset": OFFICIAL_ASSET,
            "official_config_sha256": digest(official)}
    write_json(session / "plan.json", plan)
    write_json(session / "ledger.json", {"state": "pending", "pass": 0, "case": 0,
               "repetition": 0, "next_attempt": 1, "trial_failures": 0,
               "attempts": [], "trial_results": [], "reason": None, "active": None})
    return session

def trial(config, ledger):
    if ledger["state"] == "complete":
        return None
    s = config["schedule"]
    case = s["cases"][ledger["case"]]
    params = copy.deepcopy(config["parameters"])
    params.update(case["parameters"])  # documented shallow top-level override
    return {"trial_id": f"p{ledger['pass']+1:04d}_{case['id']}_r{ledger['repetition']+1:04d}",
            "case_id": case["id"], "pass_index": ledger["pass"]+1,
            "repetition_index": ledger["repetition"]+1, "parameters": params}

def advance(c, ledger):
    s = c["schedule"]
    case = s["cases"][ledger["case"]]
    ledger["repetition"] += 1
    if ledger["repetition"] >= case.get("repetitions", s["repetitions_per_case"]):
        ledger["repetition"] = 0
        ledger["case"] += 1
        if ledger["case"] >= len(s["cases"]):
            ledger["case"] = 0
            ledger["pass"] += 1
    rep = s["repeat_set"]
    ledger.update(next_attempt=1, trial_failures=0, reason=None, active=None)
    ledger["state"] = "complete" if rep["mode"] == "fixed" and ledger["pass"] >= rep["count"] else "pending"

def begin_attempt(session, c, ledger):
    t = trial(c, ledger)
    attempt_id = f"attempt_{ledger['next_attempt']:03d}_{uuid.uuid4().hex[:8]}"
    directory = Path(session) / "trials" / t["trial_id"] / attempt_id
    directory.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    record = {**t, "attempt_id": attempt_id, "directory": directory.relative_to(session).as_posix(),
              "token": token, "started_utc": utc(), "result": None}
    ledger["attempts"].append(record)
    ledger.update(state="running", active=record["directory"], reason=None)
    write_json(Path(session) / "ledger.json", ledger)
    write_json(directory / "assignment.json", record)
    return directory, record

def commit_attempt(c, ledger, result):
    record = ledger["attempts"][-1]
    if record["result"] is not None:
        raise ValueError("Attempt already committed")
    record["result"] = result
    ledger["active"] = None
    ledger["next_attempt"] += 1
    reason = result["reason"]
    if reason == "operator_stop":
        ledger.update(state="paused", reason=reason)
        return
    failure = result["execution_status"] != "completed" or result["task_outcome"] == "failure"
    eligible = reason in c["retry"]["retry_on"]
    if result["task_outcome"] == "failure" and "task_failure" in c["retry"]["retry_on"]:
        eligible = True
    if failure and eligible:
        ledger["trial_failures"] += 1
        if ledger["trial_failures"] <= c["retry"]["max_retries_per_trial"]:
            ledger.update(state="pending", reason="retry: " + reason)
            return
        if c["retry"]["on_exhausted"] == "pause":
            ledger.update(state="paused", reason="retry_exhausted: " + reason)
            return
    elif failure:
        if result["execution_status"] != "completed" or c["on_task_failure"] == "pause":
            ledger.update(state="paused", reason=reason)
            return
    ledger.setdefault("trial_results", []).append({"trial_id": record["trial_id"],
        "final_attempt": record["attempt_id"], "result": result})
    advance(c, ledger)

def result(status, outcome, reason, **extra):
    return {"execution_status": status, "task_outcome": outcome, "reason": reason,
            "completed_utc": utc(), "official_score_captured": False, **extra}
