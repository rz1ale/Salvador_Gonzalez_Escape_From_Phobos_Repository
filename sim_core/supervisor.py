"""Fresh-process official-simulation supervisor. No native files are generated/edited."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import (Lock, OFFICIAL_ASSET, advance, begin_attempt, check_sources, commit_attempt,
                     local_path, read_json, result, trial, utc, write_json)

def retry_schedule(config, initial_attempts, total_attempts):
    """Launcher policy only; public-I/O/flight code and smoke signatures stay fixed."""
    if (type(initial_attempts) is not int or type(total_attempts) is not int or
            not 1 <= initial_attempts <= total_attempts):
        raise ValueError("Retry schedule requires 1 <= main attempts <= total attempts")
    if config['mode'] != 'automatic' or config['schedule']['repeat_set']['mode'] != 'fixed':
        raise ValueError("Retry catch-up requires a finite automatic schedule")
    c=copy.deepcopy(config)
    c['retry'].update(max_retries_per_trial=initial_attempts-1,on_exhausted='continue')
    c['retry']['retry_on']=list(dict.fromkeys(c['retry']['retry_on']+[
        'startup_timeout','backend_disconnect','simulation_clock_stall',
        'controller_unresponsive','attempt_wall_timeout','process_exit',
        'shutdown_timeout','task_failure']))
    c['retry']['retry_on']=[r for r in c['retry']['retry_on'] if r!='controller_error']
    c['on_task_failure']='pause'
    return c,dict(schema_version=1,initial_attempts=initial_attempts,total_attempts=total_attempts)

def get_retry_schedule(plan):
    policy=plan.get('retry_schedule')
    if policy is None: return None
    if (not isinstance(policy,dict) or set(policy)!={'schema_version','initial_attempts','total_attempts'} or
            type(policy['schema_version']) is not int or policy['schema_version']!=1):
        raise ValueError('Invalid retry_schedule in plan')
    retry_schedule(plan['config'],policy['initial_attempts'],policy['total_attempts'])
    return policy

def attempt_count(ledger,trial_id):
    # Absolute physical-attempt reservations; resume/reconfiguration never resets it.
    return sum(r['trial_id']==trial_id for r in ledger['attempts'])

def retryable(config,value):
    if value['reason'] in ('operator_stop','controller_error','smoke_tests_required'):
        return False
    return (value['execution_status']!='completed' or value['task_outcome']=='failure') and (
        value['reason'] in config['retry']['retry_on'] or
        value['task_outcome']=='failure' and 'task_failure' in config['retry']['retry_on'])

def queue_trial(config,ledger,record):
    if any(q['trial_id']==record['trial_id'] for q in ledger['retry_queue']): return
    index=next(i for i,c in enumerate(config['schedule']['cases']) if c['id']==record['case_id'])
    ledger['retry_queue'].append(dict(trial_id=record['trial_id'],case=index,
        **{'pass':record['pass_index']-1,'repetition':record['repetition_index']-1},state='waiting'))

def select_catch_up(ledger):
    remaining=[q for q in ledger['retry_queue'] if q['state']=='waiting']
    if not remaining:
        if 'main_end_cursor' in ledger: ledger.update(ledger['main_end_cursor'])
        ledger.update(state='complete',retry_phase='complete',active=None,reason=None)
        return
    q=remaining[0]
    ledger.update({k:q[k] for k in ('pass','case','repetition')})
    previous=[r for r in ledger['attempts'] if r['trial_id']==q['trial_id']]
    # Numeric directory sequence remains unique, including pre-policy attempts.
    next_number=max([int(r['attempt_id'].split('_')[1]) for r in previous],default=0)+1
    ledger.update(state='pending',retry_phase='catch_up',next_attempt=next_number,
                  trial_failures=0,active=None,reason='catch_up: '+q['trial_id'])

def finish_phase(ledger):
    if ledger['retry_phase']=='main' and ledger['state']=='complete':
        ledger['main_end_cursor']={k:ledger[k] for k in ('pass','case','repetition')}
        select_catch_up(ledger)

def initialize_retry_schedule(config,ledger,plan):
    policy=get_retry_schedule(plan)
    if not policy: return None
    if 'retry_phase' not in ledger:
        ledger.update(retry_phase='main',retry_queue=[])
        # Existing finalized failures can be revisited; completed successes never are.
        kept=[]
        for finalized in ledger.get('trial_results',[]):
            records=[r for r in ledger['attempts'] if r['trial_id']==finalized['trial_id']]
            if (records and retryable(config,finalized['result']) and
                    len(records)<policy['total_attempts']):
                queue_trial(config,ledger,records[-1])
            else: kept.append(finalized)
        ledger['trial_results']=kept
    finish_phase(ledger)
    return policy

def finalize_without_launch(config,ledger,record,policy):
    value=record['result']
    phase=ledger['retry_phase']
    deferred=(phase=='main' and retryable(config,value) and
              attempt_count(ledger,record['trial_id'])<policy['total_attempts'])
    if deferred:
        queue_trial(config,ledger,record)
    elif not any(r['trial_id']==record['trial_id'] for r in ledger['trial_results']):
        ledger['trial_results'].append(dict(trial_id=record['trial_id'],
            final_attempt=record['attempt_id'],result=value))
    if phase=='catch_up':
        next(q for q in ledger['retry_queue'] if q['trial_id']==record['trial_id'])['state']='done'
        select_catch_up(ledger)
    else:
        advance(config,ledger)
        finish_phase(ledger)

def prepare_retry_schedule(config,ledger,plan):
    policy=initialize_retry_schedule(config,ledger,plan)
    if not policy: return
    while ledger['state']=='pending':
        t=trial(config,ledger)
        used=attempt_count(ledger,t['trial_id'])
        limit=policy['initial_attempts'] if ledger['retry_phase']=='main' else policy['total_attempts']
        if used<limit: return
        records=[r for r in ledger['attempts'] if r['trial_id']==t['trial_id']]
        latest=records[-1]
        if latest['result'] is None:
            raise RuntimeError('Uncommitted attempt requires reconciliation before retry scheduling')
        value=latest['result']
        success=value['execution_status']=='completed' and value['task_outcome']=='success'
        if not success and not retryable(config,value):
            ledger.update(state='paused',reason='attempt_limit_requires_review: '+value['reason'])
            return
        finalize_without_launch(config,ledger,latest,policy)

def commit_scheduled_attempt(config,ledger,value,plan):
    policy=initialize_retry_schedule(config,ledger,plan)
    if not policy:
        commit_attempt(config,ledger,value)
        return
    record=ledger['attempts'][-1]
    phase=ledger['retry_phase']
    used=attempt_count(ledger,record['trial_id'])
    limit=policy['initial_attempts'] if phase=='main' else policy['total_attempts']
    eligible=retryable(config,value)
    work=copy.deepcopy(config)
    # Derive remaining budget from immutable attempt history, not a resettable counter.
    ledger['trial_failures']=0
    work['retry'].update(max_retries_per_trial=1 if used<limit else 0,on_exhausted='continue')
    if not eligible:
        work['retry']['retry_on']=[]
        work['on_task_failure']='pause'
    commit_attempt(work,ledger,value)
    final=bool(ledger['trial_results'] and ledger['trial_results'][-1]['final_attempt']==record['attempt_id'])
    if final:
        if phase=='main' and eligible and used<policy['total_attempts']:
            ledger['trial_results'].pop()
            queue_trial(config,ledger,record)
        elif phase=='catch_up':
            next(q for q in ledger['retry_queue'] if q['trial_id']==record['trial_id'])['state']='done'
        if phase=='catch_up': select_catch_up(ledger)
    finish_phase(ledger)

def active_simulators():
    if os.name != "nt":
        return []
    query = ("Get-CimInstance Win32_Process -ErrorAction Stop | "
             "Where-Object {$_.Name -in @('SpaceCRAFT.exe','run_Server.exe')} | "
             "Select-Object ProcessId,Name | ConvertTo-Json -Compress")
    p = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", query],
                       capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise RuntimeError("Cannot inspect existing simulator processes: " + p.stderr)
    data = json.loads(p.stdout) if p.stdout.strip() else []
    return data if isinstance(data, list) else [data]

class WindowsJob:
    """Kernel-owned descendant cleanup, with explicit 64-bit-safe ctypes signatures."""
    def __init__(self):
        import ctypes as ct
        from ctypes import wintypes as wt
        self.ct = ct
        self.k = ct.WinDLL("kernel32", use_last_error=True)
        self.k.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
        self.k.CreateJobObjectW.restype = wt.HANDLE
        self.k.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
        self.k.SetInformationJobObject.restype = wt.BOOL
        self.k.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        self.k.AssignProcessToJobObject.restype = wt.BOOL
        self.k.QueryInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p]
        self.k.QueryInformationJobObject.restype = wt.BOOL
        self.k.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
        self.k.TerminateJobObject.restype = wt.BOOL
        self.k.CloseHandle.argtypes = [wt.HANDLE]
        self.k.CloseHandle.restype = wt.BOOL
        class BASIC(ct.Structure):
            _fields_ = [("ProcessTime", ct.c_int64), ("JobTime", ct.c_int64),
                        ("Flags", wt.DWORD), ("MinWorking", ct.c_size_t),
                        ("MaxWorking", ct.c_size_t), ("ActiveLimit", wt.DWORD),
                        ("Affinity", ct.c_size_t), ("Priority", wt.DWORD), ("Scheduling", wt.DWORD)]
        class IO(ct.Structure):
            _fields_ = [(name, ct.c_uint64) for name in ("r", "w", "o", "rb", "wb", "ob")]
        class EXT(ct.Structure):
            _fields_ = [("Basic", BASIC), ("IO", IO), ("ProcessMemory", ct.c_size_t),
                        ("JobMemory", ct.c_size_t), ("PeakProcess", ct.c_size_t), ("PeakJob", ct.c_size_t)]
        class ACCOUNT(ct.Structure):
            _fields_ = [("User", ct.c_int64), ("Kernel", ct.c_int64), ("UserPeriod", ct.c_int64),
                        ("KernelPeriod", ct.c_int64), ("Faults", wt.DWORD), ("Total", wt.DWORD),
                        ("Active", wt.DWORD), ("Terminated", wt.DWORD)]
        self.account_type = ACCOUNT
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())
        info = EXT()
        info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.k.SetInformationJobObject(self.handle, 9, ct.byref(info), ct.sizeof(info)):
            self.close()
            raise ct.WinError(ct.get_last_error())
    def attach(self, child):
        if not self.k.AssignProcessToJobObject(self.handle, int(child._handle)):
            raise self.ct.WinError(self.ct.get_last_error())
    def count(self):
        info = self.account_type()
        if not self.k.QueryInformationJobObject(self.handle, 1, self.ct.byref(info), self.ct.sizeof(info), None):
            raise self.ct.WinError(self.ct.get_last_error())
        return info.Active
    def terminate(self):
        if not self.k.TerminateJobObject(self.handle, 1):
            raise self.ct.WinError(self.ct.get_last_error())
    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None

class OwnedProcess:
    def __init__(self, command, cwd, env, output):
        self.job = WindowsJob() if os.name == "nt" else None
        self.output = Path(output).open("ab", buffering=0)
        self.child = None
        try:
            self.child = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=self.output,
                                          stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
            if self.job:
                self.job.attach(self.child)
        except BaseException:
            if self.child and self.child.poll() is None:
                self.child.terminate()
                self.child.wait(timeout=10)
            if self.job:
                self.job.close()
            self.output.close()
            raise
    def poll(self):
        return self.child.poll()
    def close(self):
        # Only processes launched into this owned job/group are eligible for cleanup.
        try:
            if self.job:
                if self.job.count():
                    self.job.terminate()
                deadline = time.monotonic() + 10
                while self.job.count() and time.monotonic() < deadline:
                    time.sleep(.1)
                if self.job.count():
                    raise RuntimeError("Owned simulator descendants did not exit")
            else:
                try:
                    os.killpg(self.child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait(timeout=10)
        finally:
            if self.job:
                self.job.close()
            self.output.close()

def load_matching(path, token):
    if not Path(path).exists():
        return None
    data = read_json(path)
    if data.get("token") != token:
        raise RuntimeError("Mismatched/stale attempt message: " + str(path))
    return data

def execute_attempt(root, session, attempt, assignment, config, executable, first_role=False):
    env = os.environ.copy()
    env.update(STU4_SESSION=Path(session).relative_to(root).as_posix(),
               STU4_ATTEMPT=Path(attempt).relative_to(session).as_posix(),
               STU4_ATTEMPT_TOKEN=assignment["token"])
    command = [str(executable), "--sim="+OFFICIAL_ASSET, "-quitOnStop"]
    if first_role:
        command.append("-firstRole")
    # The caller validates SpaceCRAFT.exe and exact official file before any spawn.
    write_json(Path(attempt) / "launch.json", {"command": command, "created_utc": utc()})
    child = OwnedProcess(command, Path(executable).parent, env, Path(attempt) / "launcher.txt")
    start = time.monotonic()
    ready = False
    last_heartbeat = last_clock_change = start
    last_clock = None
    shutdown_at = None
    failure = None
    interrupts = 0
    limits = config["limits"]
    try:
        while True:
            try:
                now = time.monotonic()
                hb = load_matching(Path(attempt) / "heartbeat.json", assignment["token"])
                if hb:
                    age = max(0., time.time()-float(hb["wall_unix_s"]))
                    last_heartbeat = now-age
                    ready = ready or bool(hb.get("ready"))
                    clock = hb.get("sim_s")
                    if clock is not None and (last_clock is None or clock > last_clock):
                        last_clock_change, last_clock = now, clock
                final = load_matching(Path(attempt) / "result.json", assignment["token"])
                if (Path(session) / "STOP_NOW").exists() and shutdown_at is None:
                    failure, shutdown_at = "operator_stop", now
                if final and shutdown_at is None:
                    shutdown_at = now
                if child.poll() is not None:
                    if failure:
                        return result("interrupted" if failure == "operator_stop" else "failed",
                                      "not_assessed", failure, controller_result=final)
                    return final or result("failed", "not_assessed", "process_exit",
                                           exit_code=child.poll())
                if shutdown_at is not None:
                    if now-shutdown_at > limits["shutdown_wall_s"]:
                        return result("interrupted" if failure == "operator_stop" else "failed", "not_assessed",
                                      failure or "shutdown_timeout", controller_result=final)
                elif not ready and now-start > limits["startup_wall_s"]:
                    failure, shutdown_at = "startup_timeout", now
                    (Path(session) / "STOP_NOW").touch()
                elif hb and now-last_heartbeat > limits["heartbeat_wall_s"]:
                    failure, shutdown_at = "controller_unresponsive", now
                    (Path(session) / "STOP_NOW").touch()
                elif ready and now-last_clock_change > limits["clock_stall_wall_s"]:
                    failure, shutdown_at = "simulation_clock_stall", now
                    (Path(session) / "STOP_NOW").touch()
                elif now-start > limits["attempt_wall_s"]:
                    failure, shutdown_at = "attempt_wall_timeout", now
                    (Path(session) / "STOP_NOW").touch()
                time.sleep(.2)
            except KeyboardInterrupt:
                interrupts += 1
                (Path(session) / "PAUSE").touch()
                if interrupts >= 2:
                    (Path(session) / "STOP_NOW").touch()
                print("Pause after current attempt requested" if interrupts == 1 else "Stop current attempt requested", flush=True)
    finally:
        child.close()

def reconcile(session, config, ledger, plan=None):
    if ledger["state"] != "running":
        return
    attempt = Path(session) / ledger["active"]
    record = ledger["attempts"][-1]
    old = load_matching(attempt / "supervisor_result.json", record["token"])
    if old is None:
        old = load_matching(attempt / "result.json", record["token"])
    value = old or result("failed", "not_assessed", "process_exit", note="Interrupted supervisor; no completion proof")
    write_json(attempt / "supervisor_result.json", value)
    plan=read_json(Path(session)/'plan.json') if plan is None else plan
    commit_scheduled_attempt(config, ledger, value, plan)
    write_json(Path(session) / "ledger.json", ledger)

def supervise(root, session, executable, *, first_role=False, attempt_runner=execute_attempt):
    root, session = Path(root).resolve(), Path(session).resolve()
    executable = Path(executable).expanduser().resolve()
    if not executable.is_file() or executable.name.lower() != "spacecraft.exe":
        raise ValueError("Supply the installed SpaceCRAFT.exe; backend-only mode is not supported")
    official = root.parent / "STU4-Sim" / "Sims" / "COMPETITION_SIM.simconf"
    if not official.is_file():
        raise ValueError("Missing sibling official COMPETITION_SIM.simconf")
    with Lock(root / "sim_core" / "state" / "supervisor.lock"):
        if active_simulators():
            raise RuntimeError("Close existing SpaceCRAFT/run_Server processes before automatic startup")
        # A manual controller can outlive its GUI briefly. Never take it over.
        with Lock(root / "sim_core" / "state" / "controller.lock"):
            pass
        plan = read_json(session / "plan.json")
        config = plan["config"]
        if config["mode"] != "automatic":
            raise ValueError("Cannot supervise a manual session")
        check_sources(root, plan)
        ledger = read_json(session / "ledger.json")
        reconcile(session, config, ledger, plan)
        prepare_retry_schedule(config,ledger,plan)
        write_json(root / "sim_core" / "state" / "active_session.json", {"session": session.relative_to(root).as_posix()})
        while ledger["state"] == "pending":
            if (session / "PAUSE").exists():
                ledger.update(state="paused", reason="pause_after_attempt")
                break
            check_sources(root, plan)
            prepare_retry_schedule(config,ledger,plan)
            write_json(session/'ledger.json',ledger)
            if ledger['state']!='pending': break
            attempt, assignment = begin_attempt(session, config, ledger)
            if 'retry_phase' in ledger:
                assignment['retry_phase']=ledger['retry_phase']
                assignment['attempt_number_for_trial']=attempt_count(ledger,assignment['trial_id'])
                write_json(attempt/'assignment.json',assignment)
                write_json(session/'ledger.json',ledger)
            try:
                value = attempt_runner(root, session, attempt, assignment, config, executable, first_role)
            except KeyboardInterrupt:
                (session / "PAUSE").touch()
                value = result("interrupted", "not_assessed", "operator_stop")
            except Exception as exc:
                value = result("failed", "not_assessed", "controller_error", details=str(exc))
            value.update(token=assignment["token"], trial_id=assignment["trial_id"], attempt_id=assignment["attempt_id"])
            write_json(attempt / "supervisor_result.json", value)
            commit_scheduled_attempt(config, ledger, value, plan)
            # A watchdog's STOP_NOW is attempt-scoped. Operator PAUSE persists.
            if not (session / "PAUSE").exists():
                (session / "STOP_NOW").unlink(missing_ok=True)
            if (session / "PAUSE").exists() and ledger["state"] == "pending":
                ledger.update(state="paused", reason="pause_after_attempt")
            write_json(session / "ledger.json", ledger)
            print(json.dumps({"trial": assignment["trial_id"], "attempt": assignment["attempt_id"],
                              "reason": value["reason"], "session_state": ledger["state"],
                              **({"phase":ledger['retry_phase'],
                                  "attempts_for_trial":attempt_count(ledger,assignment['trial_id']),
                                  "deferred_trials":sum(q['state']=='waiting' for q in ledger['retry_queue'])}
                                 if 'retry_phase' in ledger else {})}), flush=True)
            if ledger["state"] == "pending":
                time.sleep(.2)
        write_json(session / "ledger.json", ledger)
        return 0 if ledger["state"] == "complete" else 2
