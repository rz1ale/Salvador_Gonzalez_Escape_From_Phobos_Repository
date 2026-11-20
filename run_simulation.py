"""Operator CLI. Run this from a terminal; Controller.py is started by STPro only."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from sim_core.common import (DISPATCH_MARK, Lock, check_sources, default_config, digest, install, json_hash,
                             local_path, new_session, read_json, snapshot_sources, split_original,
                             utc, validate, write_json)

ROOT = Path(__file__).resolve().parent

def analysis_spec(root,value):
    name=value.replace('\\','/')
    path=local_path(root,name,must_exist=True)
    if path.suffix!='.py' or path.name.lower() in ('controller.py','originalcontroller.py','run_simulation.py'):
        raise ValueError('Analysis must be a solution-local offline .py script, not a controller/launcher')
    return dict(entry=name,sha256=digest(path))

def run_post_analysis(root,session,supervisor_code):
    if supervisor_code: return supervisor_code
    plan=read_json(Path(session)/'plan.json')
    spec=plan.get('post_analysis')
    if not spec: return supervisor_code
    if (not isinstance(spec,dict) or set(spec)!={'entry','sha256'} or
            not isinstance(spec['entry'],str) or not isinstance(spec['sha256'],str)):
        raise ValueError('Invalid post_analysis metadata')
    checked=analysis_spec(root,spec['entry'])
    if checked!=spec: raise ValueError('Analysis source changed; run the intended analyzer explicitly on the completed data')
    if spec['entry']==plan['config']['entry']: raise ValueError('Flight entry cannot be its own offline analyzer')
    if read_json(Path(session)/'ledger.json')['state']!='complete': return supervisor_code
    print('Campaign finished; running offline analysis.',flush=True)
    code=subprocess.call([sys.executable,str(local_path(root,spec['entry'])),
        '--session',Path(session).relative_to(root).as_posix(),'--show'],cwd=root)
    if code: print('Offline analysis failed; recorded campaign results are unchanged.',flush=True)
    return code

def selected_session(root, value=None):
    if value:
        p = local_path(root, value)
    else:
        pointer = Path(root) / "sim_core" / "state" / "active_session.json"
        if not pointer.exists():
            raise ValueError("No active session")
        p = local_path(root, read_json(pointer)["session"])
    if not (p / "plan.json").is_file() or not (p / "ledger.json").is_file():
        raise ValueError("Not a runner session: " + str(p))
    return p

def validate_install(root):
    original = Path(root) / "OriginalController.py"
    controller = Path(root) / "Controller.py"
    if not original.is_file() or not controller.is_file():
        raise ValueError("Run: python run_simulation.py install")
    a, _ = split_original(original.read_bytes())
    b, _ = split_original(controller.read_bytes())
    if a != b or DISPATCH_MARK.encode() not in controller.read_bytes():
        raise ValueError("Shared dispatch is missing or protected bootstrap differs; run install/review")

def update_selection(root, args, mode):
    source = getattr(args, "config", None)
    if source:
        c = read_json(local_path(root, source.replace("\\", "/"), must_exist=True))
    elif getattr(args,"entry",None):
        c = default_config(args.entry.replace("\\","/"),mode)
    else:
        c = read_json(Path(root) / "active_run.json")
    if source:
        source = source.replace("\\", "/")
        c.setdefault("dependencies", [])
        if source not in c["dependencies"]:
            c["dependencies"].append(source)
    case_id = getattr(args, "case", None)
    if case_id:
        selected = [x for x in c["schedule"]["cases"] if x["id"] == case_id]
        if len(selected) != 1:
            raise ValueError("Case ID not present exactly once: " + case_id)
        c["schedule"]["cases"] = selected
    c["mode"] = mode
    if getattr(args, "entry", None):
        c["entry"] = args.entry.replace("\\", "/")
    if getattr(args, "model", None):
        c["model"] = args.model.replace("\\", "/")
    if getattr(args, "parameters_file", None):
        c["parameters"] = read_json(local_path(root, args.parameters_file.replace("\\", "/"), must_exist=True))
    runs = getattr(args, "runs", None)
    if runs is not None:
        if len(c["schedule"]["cases"]) != 1:
            raise ValueError("--runs requires one case; edit the schedule for sets")
        c["schedule"]["cases"][0].pop("repetitions", None)
        c["schedule"]["repetitions_per_case"] = runs
        c["schedule"]["repeat_set"] = {"mode": "fixed", "count": 1}
    retries = getattr(args, "retries", None)
    if retries is not None and getattr(args,'retry_schedule',None):
        raise ValueError('Use either --retries or --retry-schedule, not both')
    if retries is not None:
        c["retry"]["max_retries_per_trial"] = retries
    if mode == "automatic":
        if getattr(args, "attempt_wall_s", None) is not None:
            c["limits"]["attempt_wall_s"] = args.attempt_wall_s
        elif c["limits"]["attempt_wall_s"] is None:
            c["limits"]["attempt_wall_s"] = 900
        if getattr(args,'retry_schedule',None):
            from sim_core.supervisor import retry_schedule
            c,_=retry_schedule(c,*args.retry_schedule)
    elif getattr(args, "command", "") == "manual":
        c["manual_session"] = "continue"
    elif getattr(args, "command", "") == "select":
        c["manual_session"] = args.manual_session
    c = validate(root, c)
    write_json(Path(root) / "active_run.json", c)
    return c

def set_retry_schedule(root,session,initial_attempts,total_attempts):
    """Explicit paused-session revision. History stays in the same session, no data copies."""
    from sim_core.supervisor import (initialize_retry_schedule, reconcile, retry_schedule)
    plan=read_json(session/'plan.json')
    if json_hash(plan['config'])!=plan['config_hash']:
        raise ValueError('Stored configuration hash differs; retry-policy revision refused')
    # Only reviewed launcher/test updates can be adopted. Never waive flight/model,
    # runtime, common, task, dependency or official-configuration hash checks.
    launcher_files={'run_simulation.py','sim_core/supervisor.py','sim_core/tests/test_runner.py'}
    protected=copy.deepcopy(plan)
    protected['source_hashes']={p:h for p,h in plan['source_hashes'].items() if p not in launcher_files}
    check_sources(root,protected)
    c,policy=retry_schedule(validate(root,plan['config']),initial_attempts,total_attempts)
    c=validate(root,c)
    ledger=read_json(session/'ledger.json')
    reconcile(session,plan['config'],ledger,plan)
    if ledger.get('active') is not None:
        raise ValueError('Retry policy requires no active attempt')
    # Keep original config/source provenance inline; no variant files or log copies.
    previous={k:copy.deepcopy(plan.get(k)) for k in
              ('config','config_hash','source_hashes','retry_schedule','revision')}
    previous.update(revised_utc=utc(),reason='explicit_set_retries')
    plan.setdefault('revisions',[]).append(previous)
    plan.update(config=c,config_hash=json_hash(c),source_hashes=snapshot_sources(root,c),
                retry_schedule=policy,revision=plan.get('revision',0)+1)
    write_json(session/'plan.json',plan)
    # If interrupted between these two atomic writes, resume repeats this idempotent
    # ledger preparation from the now-committed plan; it never launches on revision.
    initialize_retry_schedule(c,ledger,plan)
    write_json(session/'ledger.json',ledger)
    return plan

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("install", help="Preserve OriginalController; change only active controller user section")
    sub.add_parser("check", help="Validate controller dispatch, paths and configuration")
    sub.add_parser("selftest", help="Offline tests; does not start STPro")
    for name in ("select", "manual", "auto"):
        s = sub.add_parser(name)
        s.add_argument("--entry")
        s.add_argument("--config", help="Load a solution-local runner JSON plan, then apply explicit CLI overrides")
        s.add_argument("--case", help="Select one exact case ID from the loaded plan")
        s.add_argument("--model")
        s.add_argument("--parameters-file")
        s.add_argument("--runs", type=int)
        s.add_argument("--retries", type=int)
        if name == "select":
            s.add_argument("--manual-session", choices=("new_on_open", "continue"), default="new_on_open")
        if name == "manual":
            s.add_argument("--new", action="store_true", required=True, help="Arm a new manual suite; does not open GUI")
        if name == "auto":
            s.add_argument("--executable", required=True)
            s.add_argument("--first-role", action="store_true", help="Only after verifying the first role is acceptable")
            s.add_argument("--attempt-wall-s", type=float)
            s.add_argument('--retry-schedule',nargs=2,type=int,metavar=('MAIN_ATTEMPTS','TOTAL_ATTEMPTS'),
                           help='Bounded main-pass attempts then failed-only catch-up, e.g. 3 5; counts attempts, not extra retries')
            s.add_argument('--analysis',help='Solution-local offline analyzer accepting --session and --show; runs after completed campaign')
    s = sub.add_parser("resume")
    s.add_argument("--session")
    s.add_argument("--retry", action="store_true", help="Explicitly renew current trial retry budget; retain old attempts")
    s.add_argument("--executable")
    s.add_argument("--first-role", action="store_true")
    s.add_argument('--analysis',help='Set/replace the stored offline analyzer for this resumed session')
    s=sub.add_parser('set-retries',help='Revise a stopped automatic session; preserve history and flight-source hashes; starts nothing')
    s.add_argument('--session')
    s.add_argument('--initial-attempts',type=int,default=3)
    s.add_argument('--max-total-attempts',type=int,default=5)
    for name in ("status", "stop"):
        s = sub.add_parser(name)
        s.add_argument("--session")
        if name == "stop":
            s.add_argument("--now", action="store_true")
    return p

def main(argv=None, root=ROOT):
    root = Path(root).resolve()
    args = parser().parse_args(argv)
    try:
        if args.command == "selftest":
            for folder in ("sim_core/tests", "system_identification/tests"):
                if (root / folder).is_dir():
                    code = subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", folder, "-v"], cwd=root)
                    if code:
                        return code
            return 0
        if args.command == "status":
            session = selected_session(root, args.session)
            ledger = read_json(session / "ledger.json")
            print(json.dumps({"session": str(session), "state": ledger["state"], "reason": ledger["reason"],
                              "attempts": len(ledger["attempts"]), "active": ledger["active"],
                              "trials_finalized": len(ledger.get("trial_results", [])),
                              "task_failures": sum(x["result"]["task_outcome"] == "failure" for x in ledger.get("trial_results", [])),
                              "failed_attempts":sum(bool(r.get('result')) and
                                  (r['result']['execution_status']=='failed' or r['result']['task_outcome']=='failure')
                                  for r in ledger['attempts']),
                              "retry_phase":ledger.get('retry_phase','main'),
                              "deferred_trials":sum(q['state']=='waiting' for q in ledger.get('retry_queue',[])),
                              "retry_schedule":read_json(session/'plan.json').get('retry_schedule'),
                              "unresolved_trials":[r['trial_id'] for r in ledger.get('trial_results',[])
                                  if r['result']['execution_status']!='completed' or r['result']['task_outcome']=='failure'],
                              "cursor": {k: ledger[k] for k in ("pass", "case", "repetition")}}, indent=2))
            return 0
        if args.command == "stop":
            session = selected_session(root, args.session)
            (session / "PAUSE").touch()
            if args.now:
                (session / "STOP_NOW").touch()
            print("Stop current attempt requested" if args.now else "Pause after current attempt requested")
            return 0
        if args.command in ("install", "select", "manual", "auto", "resume",'set-retries'):
            # Setup/selection cannot race a live controller or supervisor.
            with Lock(root / "sim_core" / "state" / "supervisor.lock"), Lock(root / "sim_core" / "state" / "controller.lock"):
                from sim_core.supervisor import active_simulators
                if active_simulators():
                    raise ValueError("Close existing SpaceCRAFT/run_Server processes before setup or selection")
                if args.command == "install":
                    info = install(root)
                    c = validate(root, read_json(root / "active_run.json"))
                    write_json(root / "sim_core" / "state" / "installation.json", info)
                    print("Installed. Original preserved; protected prefix unchanged. Selected: " + c["entry"])
                    return 0
                validate_install(root)
                if args.command=='set-retries':
                    session=selected_session(root,args.session)
                    plan=set_retry_schedule(root,session,args.initial_attempts,args.max_total_attempts)
                    print(json.dumps(dict(session=str(session),retry_schedule=plan['retry_schedule'],
                        revision=plan['revision'],note='No simulation started; original plan provenance retained in revisions'),indent=2))
                    return 0
                if args.command == "resume":
                    from sim_core.supervisor import active_simulators, prepare_retry_schedule, reconcile
                    if active_simulators():
                        raise ValueError("Close existing native simulations before resuming")
                    session = selected_session(root, args.session)
                    plan = read_json(session / "plan.json")
                    c = plan["config"]
                    if c["mode"] == "manual" and c["manual_session"] != "continue":
                        raise ValueError("new_on_open runs are independent; start another GUI run instead of resume")
                    if c["mode"] == "automatic" and not args.executable:
                        raise ValueError("Automatic resume requires --executable PATH_TO_SpaceCRAFT.exe")
                    check_sources(root, plan)
                    if args.analysis:
                        spec=analysis_spec(root,args.analysis)
                        if spec['entry']==c['entry']: raise ValueError('Flight entry cannot be its own offline analyzer')
                        plan.setdefault('analysis_revisions',[]).append(dict(previous=plan.get('post_analysis'),changed_utc=utc()))
                        plan['post_analysis']=spec
                        write_json(session/'plan.json',plan)
                    ledger = read_json(session / "ledger.json")
                    reconcile(session, c, ledger,plan)
                    prepare_retry_schedule(c,ledger,plan)
                    if ledger["state"] == "complete":
                        raise ValueError("Session already complete; start a new session")
                    if (str(ledger.get("reason", "")).startswith("retry_exhausted") and
                            not args.retry and not plan.get('retry_schedule')):
                        raise ValueError("Retry budget exhausted; use resume --retry to renew it explicitly")
                    if args.retry:
                        if plan.get('retry_schedule'):
                            raise ValueError('Fixed total-attempt caps cannot be reset with --retry')
                        ledger["trial_failures"] = 0
                    ledger.update(state="pending", reason="operator_resume")
                    write_json(session / "ledger.json", ledger)
                    for flag in ("PAUSE", "STOP_NOW"):
                        (session / flag).unlink(missing_ok=True)
                    if c["mode"] == "manual":
                        write_json(root / "active_run.json", c)
                    write_json(root / "sim_core" / "state" / "active_session.json", {"session": session.relative_to(root).as_posix()})
                else:
                    post=analysis_spec(root,args.analysis) if getattr(args,'analysis',None) else None
                    c = update_selection(root, args, "automatic" if args.command == "auto" else "manual")
                    if post and post['entry']==c['entry']: raise ValueError('Flight entry cannot be its own offline analyzer')
                    if args.command == "select":
                        print("Selected for manual launch: " + c["entry"])
                        return 0
                    session = new_session(root, c)
                    if getattr(args,'retry_schedule',None) or post:
                        from sim_core.supervisor import retry_schedule
                        plan=read_json(session/'plan.json')
                        if getattr(args,'retry_schedule',None):
                            _,plan['retry_schedule']=retry_schedule(c,*args.retry_schedule)
                        if post: plan['post_analysis']=post
                        write_json(session/'plan.json',plan)
                    write_json(root / "sim_core" / "state" / "active_session.json", {"session": session.relative_to(root).as_posix()})
            if c["mode"] == "manual":
                print("Manual suite armed: " + str(session) + "; open the official simulation yourself")
                return 0
            if not args.executable:
                raise ValueError("Automatic resume requires --executable PATH_TO_SpaceCRAFT.exe")
            from sim_core.supervisor import supervise
            return run_post_analysis(root,session,supervise(root, session, args.executable, first_role=args.first_role))
        if args.command == "check":
            validate_install(root)
            c = validate(root, read_json(root / "active_run.json"))
            print(json.dumps({"valid": True, "entry": c["entry"], "mode": c["mode"],
                              "note": "No native simulation started; organizer authenticity is not established by this check"}, indent=2))
            return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
