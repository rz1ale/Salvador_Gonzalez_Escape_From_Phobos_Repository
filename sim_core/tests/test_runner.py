"""Meaningful offline checks. No Space Teams installation or simulator is used."""
from __future__ import annotations
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

from sim_core.common import (DISPATCH_MARK, Lock, advance, begin_attempt, check_sources,
    commit_attempt, install, json_hash, local_path, new_session, read_json, result,
    split_original, trial, validate, write_json)
from sim_core.runtime import execute_entry, _manual_assignment, dispatch
from sim_core.supervisor import (reconcile, supervise, execute_attempt, retry_schedule,
    get_retry_schedule, commit_scheduled_attempt, prepare_retry_schedule)
from run_simulation import main, analysis_spec, run_post_analysis

PACKAGE = Path(__file__).resolve().parents[2]
PREFIX = ("# DON'T CHANGE ANY OF THE BELOW.\n"
          "BOOT_COUNT = globals().get('BOOT_COUNT', 0) + 1\n"
          "def SetCommand(a,b):\n    pass\n"
          "# DON'T CHANGE ANY OF THE ABOVE.\n"
          "########################################\n")
TAIL = "\nassert BOOT_COUNT == 1\nSetCommand([0,0,0],[0,0,0])\n"

def copy_test_core(source, destination):
    """Copy source only; installed session pointers/locks are never test inputs."""
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns("__pycache__","state"))

class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "STU4-Solution"
        self.root.mkdir()
        copy_test_core(PACKAGE / "sim_core", self.root / "sim_core")
        shutil.copy2(PACKAGE / "run_simulation.py", self.root)
        # Tests must not depend on the operator's currently selected policy/plan.
        self.raw = {
            "schema_version":1,"mode":"manual","manual_session":"new_on_open",
            "entry":"OriginalController.py","parameters":{},"model":None,"dependencies":[],
            "output_root":"run_logs","schedule":{"cases":[{"id":"default","parameters":{}}],
                "repetitions_per_case":1,"repeat_set":{"mode":"fixed","count":1},"order":"listed"},
            "retry":{"max_retries_per_trial":0,"retry_on":["startup_timeout","backend_disconnect","simulation_clock_stall"],"on_exhausted":"pause"},
            "on_task_failure":"continue","limits":{"startup_wall_s":150,"heartbeat_wall_s":45,
                "clock_stall_wall_s":90,"attempt_wall_s":None,"shutdown_wall_s":30,"task_sim_s":None},
            "logging":{"state_period_s":.2}}
        write_json(self.root / "active_run.json", self.raw)
        (self.root / "OriginalController.py").write_text(PREFIX + TAIL)
        install(self.root)
        native = self.root.parent / "STU4-Sim" / "Sims"
        native.mkdir(parents=True)
        (native / "COMPETITION_SIM.simconf").write_text('{"offline_test_fixture":true}')
    def tearDown(self):
        self.temp.cleanup()
    def config(self, **updates):
        raw = copy.deepcopy(self.raw)
        raw.update(updates)
        return validate(self.root, raw)
    def session(self, c=None):
        c = c or self.config()
        s = new_session(self.root, c)
        return c, s, read_json(s / "ledger.json")
    def test_original_is_preserved_and_prefix_is_exact(self):
        before = (self.root / "OriginalController.py").read_bytes()
        install(self.root)
        self.assertEqual(before, (self.root / "OriginalController.py").read_bytes())
        p, _ = split_original(before)
        q, _ = split_original((self.root / "Controller.py").read_bytes())
        self.assertEqual(p, q)
        self.assertIn(DISPATCH_MARK, (self.root / "Controller.py").read_text())
    def test_install_preserves_bom_crlf_and_nonascii(self):
        original = ("# café\n" + PREFIX + TAIL).replace("\n", "\r\n").encode("utf-8")
        original = b"\xef\xbb\xbf" + original
        (self.root / "OriginalController.py").write_bytes(original)
        (self.root / "Controller.py").write_bytes(original)
        install(self.root)
        self.assertEqual(split_original(original)[0], split_original((self.root / "Controller.py").read_bytes())[0])
    def test_install_refuses_other_changes_without_overwrite(self):
        path = self.root / "Controller.py"
        path.write_text(PREFIX + "\nprint('important other user code')\n")
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            install(self.root)
        self.assertEqual(before, path.read_bytes())
    def test_only_original_editable_section_executes(self):
        c = self.config()
        calls = []
        scope = {}
        exec(PREFIX, scope, scope)
        ctx = types.SimpleNamespace(root=self.root, config=c, assignment={"token":"t"},
                                    command=lambda a,b:calls.append((a,b)), radar=lambda n:[], sleep=lambda t:None)
        execute_entry(ctx, scope)
        self.assertEqual(scope["BOOT_COUNT"], 1)
        self.assertEqual(len(calls), 1)
    def test_official_config_never_changes(self):
        path = self.root.parent / "STU4-Sim/Sims/COMPETITION_SIM.simconf"
        before = path.read_bytes()
        c,s,l = self.session()
        begin_attempt(s,c,l)
        self.assertEqual(before, path.read_bytes())
    def test_paths_unknown_fields_and_bad_counts_rejected(self):
        for change in ({"entry":"../outside.py"}, {"entry":"Controller.py"},
                       {"sim":"custom.simconf"}, {"parameters":[]}, {"schema_version":True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config(**change)
        c = copy.deepcopy(self.raw)
        c["schedule"]["repetitions_per_case"] = True
        with self.assertRaises(ValueError):
            validate(self.root,c)
        c = copy.deepcopy(self.raw)
        c["retry"]["retry_on"] = ["operator_stop"]
        with self.assertRaises(ValueError):
            validate(self.root,c)
        with self.assertRaises(ValueError):
            local_path(self.root,"C:/external/file.py")
    def test_order_repetitions_passes_and_parameters(self):
        c = self.config(parameters={"gain":1,"fixed":7})
        c["schedule"].update(cases=[{"id":"A","parameters":{"gain":2}}, {"id":"B","parameters":{}}],
                             repetitions_per_case=2, repeat_set={"mode":"fixed","count":2})
        _,s,l = self.session(c)
        ids = []
        while l["state"] != "complete":
            t = trial(c,l)
            ids.append(t["case_id"])
            self.assertEqual(t["parameters"]["fixed"],7)
            if t["case_id"] == "A": self.assertEqual(t["parameters"]["gain"],2)
            advance(c,l)
        self.assertEqual(ids,["A","A","B","B","A","A","B","B"])
        self.assertIsNone(trial(c,l))
    def test_two_retries_mean_three_attempts_and_no_advance(self):
        c,s,l = self.session()
        c["retry"].update(max_retries_per_trial=2)
        ids = []
        for i in range(3):
            _,assignment = begin_attempt(s,c,l)
            ids.append(assignment["trial_id"])
            commit_attempt(c,l,result("failed","not_assessed","startup_timeout"))
            self.assertEqual(l["state"], "pending" if i<2 else "paused")
        self.assertEqual(len(set(ids)),1)
        self.assertEqual(len(l["attempts"]),3)
        self.assertEqual(l["pass"],0)
    def test_task_failure_is_recorded_without_success_reroll(self):
        c,s,l = self.session()
        c["retry"]["max_retries_per_trial"] = 5
        begin_attempt(s,c,l)
        commit_attempt(c,l,result("completed","failure","collision"))
        self.assertEqual(l["state"],"complete")
        self.assertEqual(len(l["attempts"]),1)
        self.assertEqual(l["attempts"][0]["result"]["task_outcome"],"failure")
    def test_operator_stop_pauses_and_does_not_consume_retry_budget(self):
        c,s,l = self.session()
        begin_attempt(s,c,l)
        commit_attempt(c,l,result("interrupted","not_assessed","operator_stop"))
        self.assertEqual(l["state"],"paused")
        self.assertEqual(l["trial_failures"],0)
    def test_interrupted_supervisor_reconciles_saved_completion_once(self):
        c,s,l = self.session()
        a,record = begin_attempt(s,c,l)
        write_json(a/"result.json",result("completed","success","done",token=record["token"]))
        reconcile(s,c,l)
        self.assertEqual(l["state"],"complete")
        reconcile(s,c,l)
        self.assertEqual(len(l["attempts"]),1)
    def test_hash_change_prevents_resume(self):
        c,s,l = self.session()
        plan = read_json(s/"plan.json")
        with (self.root/"OriginalController.py").open("a") as f: f.write("# changed\n")
        with self.assertRaises(ValueError): check_sources(self.root,plan)
    def test_reconcile_preserves_supervisor_failure_over_controller_stop(self):
        c,s,l=self.session()
        a,record=begin_attempt(s,c,l)
        write_json(a/"result.json",result("interrupted","not_assessed","operator_stop",token=record["token"]))
        write_json(a/"supervisor_result.json",result("failed","not_assessed","startup_timeout",token=record["token"]))
        reconcile(s,c,l)
        self.assertEqual(l["attempts"][0]["result"]["reason"],"startup_timeout")
        self.assertTrue(l["reason"].startswith("retry_exhausted"))
    def test_kernel_lock_prevents_duplicate_owner(self):
        with Lock(self.root/"test.lock"):
            with self.assertRaises(RuntimeError):
                with Lock(self.root/"test.lock"): pass
        with Lock(self.root/"test.lock"): pass
    def test_default_manual_creates_independent_session_each_open(self):
        a = _manual_assignment(self.root)[0]
        b = _manual_assignment(self.root)[0]
        self.assertNotEqual(a,b)
    def test_manual_suite_advances_one_trial_per_gui_open(self):
        c = self.config(manual_session="continue")
        c["schedule"]["cases"] = [{"id":"A","parameters":{}},{"id":"B","parameters":{}}]
        write_json(self.root/"active_run.json",c)
        s,a,rec,c,l = _manual_assignment(self.root)
        commit_attempt(c,l,result("completed","success","done"))
        write_json(s/"ledger.json",l)
        s2,a2,rec2,c2,l2 = _manual_assignment(self.root)
        self.assertEqual(s,s2)
        self.assertEqual(rec2["case_id"],"B")
    def test_installed_session_state_is_not_imported_into_fixture(self):
        # Reproduce an installed source tree whose pointer names real logs absent
        # from a temporary repo. Never create/remove state in the operator's repo.
        stale=self.root/'sim_core/state/active_session.json'
        write_json(stale,{"session":"run_logs/real_session_not_in_test_fixture"})
        before=stale.read_bytes()
        clean=Path(self.temp.name)/'clean'/'STU4-Solution'
        clean.mkdir(parents=True)
        copy_test_core(self.root/'sim_core',clean/'sim_core')
        self.assertFalse((clean/'sim_core/state').exists())
        for name in ('run_simulation.py','Controller.py','OriginalController.py'):
            shutil.copyfile(self.root/name,clean/name)
        native=clean.parent/'STU4-Sim/Sims'
        native.mkdir(parents=True)
        (native/'COMPETITION_SIM.simconf').write_text('{"offline_test_fixture":true}')
        c=copy.deepcopy(self.raw)
        c['manual_session']='continue'
        c['schedule']['cases']=[{"id":"A","parameters":{}},{"id":"B","parameters":{}}]
        write_json(clean/'active_run.json',c)
        s,a,record,c,ledger=_manual_assignment(clean)
        self.assertEqual(record['case_id'],'A')
        commit_attempt(c,ledger,result('completed','success','done'))
        write_json(s/'ledger.json',ledger)
        self.assertEqual(_manual_assignment(clean)[2]['case_id'],'B')
        self.assertEqual(stale.read_bytes(),before)
    def test_until_stopped_never_marks_schedule_complete(self):
        c,s,l = self.session()
        c["schedule"]["repeat_set"] = {"mode":"until_stopped"}
        for i in range(15): advance(c,l)
        self.assertEqual(l["state"],"pending")
        self.assertEqual(l["pass"],15)
    def test_manual_multi_trial_requires_continue(self):
        c=copy.deepcopy(self.raw)
        c["schedule"]["repetitions_per_case"]=2
        with self.assertRaises(ValueError): validate(self.root,c)
        c["manual_session"]="continue"
        self.assertEqual(validate(self.root,c)["schedule"]["repetitions_per_case"],2)
    def test_runtime_public_sdk_flow_commands_logs_and_completion(self):
        entry=self.root/"fixture_task.py"
        entry.write_text("def run(context):\n    context.state()\n    context.command([0,0,0],[0,0,0])\n    context.finish(samples=1)\n")
        c=self.config(entry="fixture_task.py")
        c["mode"]="automatic"
        c["limits"]["attempt_wall_s"]=10
        c=validate(self.root,c)
        s=new_session(self.root,c)
        l=read_json(s/"ledger.json")
        a,assignment=begin_attempt(s,c,l)
        epoch=datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc)
        class Stamp:
            def __init__(self,t): self.t=t
            def as_datetime(self): return epoch+datetime.timedelta(seconds=self.t)
        class PV:
            def loc_WRT_ExprIn(self,f): return [0,0,100]
            def vel_WRT_ExprIn(self,f): return [0,0,0]
        class Body:
            def GetBodyFixedFrame(self): return "frame"
            def getLocVelAcc(self): return PV()
            def getRotation(self): return types.SimpleNamespace(Quat_WRT=lambda f:[0,0,0,1])
            def getAngVelocity(self): return types.SimpleNamespace(WRT_ExprIn=lambda f:[0,0,0])
            def GetParam(self,kind,key): return 0
        class SDK:
            def __init__(self):
                self.t=0
                self.stopped=False
                self.VarType=types.SimpleNamespace(timestamp="timestamp",double="double")
                self.SimGlobals=types.SimpleNamespace(GetSimEntity=lambda:types.SimpleNamespace(GetParam=lambda *a:Stamp(0)),
                                                    SimClock=types.SimpleNamespace(GetTimeNow=self.now))
            def now(self):
                if self.stopped: raise EOFError("fixture stopped")
                self.t+=.05
                return Stamp(self.t)
            def stop_sim(self): self.stopped=True
        sdk=SDK()
        calls=[]
        ns=dict(st=sdk,shuttle=Body(),phobos=Body(),target=Body(),SetCommand=lambda x,y:calls.append((x,y)))
        env=dict(STU4_SESSION=s.relative_to(self.root).as_posix(),STU4_ATTEMPT=a.relative_to(s).as_posix(),STU4_ATTEMPT_TOKEN=assignment["token"])
        with patch.dict(os.environ,env): dispatch(ns,self.root)
        value=read_json(a/"result.json")
        self.assertEqual(value["task_outcome"],"success")
        self.assertEqual(len(calls),2)  # task command, then final zero request
        self.assertTrue((a/"states.txt").is_file())
        self.assertTrue((a/"commands.txt").is_file())
        self.assertEqual(read_json(s/"ledger.json")["state"],"running")  # supervisor alone commits
        reconcile(s,c,l)
        self.assertEqual(l["state"],"complete")
    def test_cli_install_and_check(self):
        with patch("sim_core.supervisor.active_simulators",return_value=[]):
            self.assertEqual(main(["install"],self.root),0)
            self.assertEqual(main(["check"],self.root),0)
    def test_supervisor_retry_ledger_and_no_duplicate_trial(self):
        c = self.config()
        c["mode"]="automatic"
        c["limits"]["attempt_wall_s"]=10
        c["retry"]["max_retries_per_trial"]=1
        c=validate(self.root,c)
        s=new_session(self.root,c)
        exe=self.root/"SpaceCRAFT.exe"
        exe.write_text("offline fixture")
        calls=[]
        def fake(root,session,attempt,assignment,config,executable,first_role):
            calls.append(assignment["trial_id"])
            return result("failed","not_assessed","startup_timeout") if len(calls)==1 else result("completed","success","done")
        with patch("sim_core.supervisor.active_simulators",return_value=[]):
            self.assertEqual(supervise(self.root,s,exe,attempt_runner=fake),0)
        ledger=read_json(s/"ledger.json")
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],calls[1])
        self.assertEqual(ledger["state"],"complete")
        self.assertTrue(all((s/x["directory"]/"supervisor_result.json").is_file() for x in ledger["attempts"]))
    def retry_session(self,cases=('A','B','C'),main_attempts=3,total_attempts=5):
        c=self.config();c['mode']='automatic'
        c['limits']['attempt_wall_s']=10
        c['schedule']['cases']=[dict(id=name,parameters={}) for name in cases]
        c,policy=retry_schedule(c,main_attempts,total_attempts)
        c=validate(self.root,c)
        s=new_session(self.root,c)
        plan=read_json(s/'plan.json');plan['retry_schedule']=policy
        write_json(s/'plan.json',plan)
        exe=self.root/'SpaceCRAFT.exe';exe.write_text('offline process fixture')
        return c,s,plan,exe
    def test_retry_main_pass_then_failed_only_catch_up_and_hard_cap(self):
        c,s,plan,exe=self.retry_session()
        calls=[];counts={}
        def fake(root,session,attempt,a,config,executable,first_role):
            name=a['case_id'];counts[name]=counts.get(name,0)+1;calls.append(name)
            success=name=='B' or name=='A' and counts[name]==5
            return result('completed','success' if success else 'failure','done' if success else 'radar_clearance_guard')
        with patch('sim_core.supervisor.active_simulators',return_value=[]):
            self.assertEqual(supervise(self.root,s,exe,attempt_runner=fake),0)
        l=read_json(s/'ledger.json')
        self.assertEqual(calls,['A','A','A','B','C','C','C','A','A','C','C'])
        self.assertEqual(counts,dict(A=5,B=1,C=5))
        self.assertEqual(l['state'],'complete');self.assertEqual(l['retry_phase'],'complete')
        self.assertEqual(len(l['trial_results']),3)
        self.assertEqual(sum(r['result']['task_outcome']=='failure' for r in l['trial_results']),1)
        self.assertEqual(len({r['trial_id'] for r in l['trial_results']}),3)
        self.assertEqual(l['pass'],1);self.assertEqual(l['case'],0)
        self.assertTrue(all((s/r['directory']/'supervisor_result.json').exists() for r in l['attempts']))
    def test_retry_operator_stop_pauses_and_absolute_cap_survives_resume(self):
        c,s,plan,exe=self.retry_session(cases=('A',))
        calls=[]
        def fake(root,session,attempt,a,config,executable,first_role):
            calls.append(a['attempt_id'])
            return result('interrupted','not_assessed','operator_stop') if len(calls)==1 else result('completed','failure','radar_clearance_guard')
        with patch('sim_core.supervisor.active_simulators',return_value=[]):
            self.assertEqual(supervise(self.root,s,exe,attempt_runner=fake),2)
            self.assertEqual(len(calls),1)
            l=read_json(s/'ledger.json');l.update(state='pending',reason='operator_resume')
            write_json(s/'ledger.json',l)
            self.assertEqual(supervise(self.root,s,exe,attempt_runner=fake),0)
        l=read_json(s/'ledger.json')
        self.assertEqual(len(calls),5);self.assertEqual(l['state'],'complete')
        self.assertEqual(len(set(calls)),5)
    def test_retry_controller_and_missing_smokes_failures_pause_immediately(self):
        for reason,status,outcome in [('controller_error','failed','not_assessed'),
                                      ('smoke_tests_required','completed','failure')]:
            with self.subTest(reason=reason):
                c,s,plan,exe=self.retry_session(cases=('A',))
                calls=[]
                def fake(*args):
                    calls.append(1);return result(status,outcome,reason)
                with patch('sim_core.supervisor.active_simulators',return_value=[]):
                    self.assertEqual(supervise(self.root,s,exe,attempt_runner=fake),2)
                l=read_json(s/'ledger.json')
                self.assertEqual(len(calls),1);self.assertEqual(l['reason'],reason)
    def test_retry_reconcile_interrupted_catch_up_result_exactly_once(self):
        c,s,plan,exe=self.retry_session(cases=('A','B'))
        l=read_json(s/'ledger.json');prepare_retry_schedule(c,l,plan)
        for _ in range(3):
            begin_attempt(s,c,l)
            commit_scheduled_attempt(c,l,result('completed','failure','radar_clearance_guard'),plan)
        self.assertEqual(l['case'],1)
        begin_attempt(s,c,l);commit_scheduled_attempt(c,l,result('completed','success','done'),plan)
        self.assertEqual(l['retry_phase'],'catch_up')
        a,record=begin_attempt(s,c,l)
        value=result('completed','success','done',token=record['token'])
        write_json(a/'supervisor_result.json',value)
        before=len(l['attempts'])
        reconcile(s,c,l,plan);reconcile(s,c,l,plan)
        self.assertEqual(l['state'],'complete');self.assertEqual(len(l['attempts']),before)
        self.assertEqual(len(l['trial_results']),2)
    def test_retry_cap_before_launch_after_manual_or_prior_attempts(self):
        c,s,plan,exe=self.retry_session(cases=('A',))
        l=read_json(s/'ledger.json')
        # Historical pause-mode failures existed before policy activation.
        old=copy.deepcopy(c);old['retry'].update(max_retries_per_trial=0,retry_on=[],on_exhausted='pause')
        for _ in range(3):
            l['state']='pending';begin_attempt(s,old,l)
            commit_attempt(old,l,result('completed','failure','radar_clearance_guard'))
        l['state']='pending';prepare_retry_schedule(c,l,plan)
        self.assertEqual(len(l['attempts']),3);self.assertEqual(l['retry_phase'],'catch_up')
        self.assertEqual(l['next_attempt'],4)
        for _ in range(2):
            begin_attempt(s,c,l)
            commit_scheduled_attempt(c,l,result('completed','failure','radar_clearance_guard'),plan)
        self.assertEqual(len(l['attempts']),5);self.assertEqual(l['state'],'complete')
    def test_retry_schedule_validation_and_existing_finalized_failures(self):
        c,s,plan,exe=self.retry_session(cases=('A',))
        for limits in ((0,5),(4,3),(True,5),(3,False)):
            with self.assertRaises(ValueError):retry_schedule(c,*limits)
        c2=copy.deepcopy(c);c2['schedule']['repeat_set']={'mode':'until_stopped'}
        with self.assertRaises(ValueError): retry_schedule(c2,3,5)
        plan2=copy.deepcopy(plan);plan2['retry_schedule']['schema_version']=True
        with self.assertRaises(ValueError):get_retry_schedule(plan2)
        l=read_json(s/'ledger.json')
        # A legacy finalized failure can be retried in catch-up without relaunching successes.
        old=copy.deepcopy(c);old['retry'].update(max_retries_per_trial=0,on_exhausted='continue')
        begin_attempt(s,old,l);commit_attempt(old,l,result('completed','failure','radar_clearance_guard'))
        self.assertEqual(l['state'],'complete')
        prepare_retry_schedule(c,l,plan)
        self.assertEqual(l['state'],'pending');self.assertEqual(l['retry_phase'],'catch_up')
        self.assertEqual(len(l['trial_results']),0)
    def test_cli_retry_schedule_policy_saved_and_conflicting_flags_rejected(self):
        with patch('sim_core.supervisor.active_simulators',return_value=[]),patch('sim_core.supervisor.supervise',return_value=0):
            args=['auto','--entry','OriginalController.py','--executable','SpaceCRAFT.exe','--retry-schedule','3','5']
            self.assertEqual(main(args,self.root),0)
            active=read_json(self.root/'sim_core/state/active_session.json')
            plan=read_json(self.root/active['session']/'plan.json')
            self.assertEqual(plan['retry_schedule']['total_attempts'],5)
            self.assertIn('task_failure',plan['config']['retry']['retry_on'])
            self.assertEqual(main(args+['--retries','0'],self.root),2)
    def test_retry_policy_revision_preserves_history_and_rejects_flight_changes(self):
        from run_simulation import set_retry_schedule
        c=self.config();c['mode']='automatic';c['limits']['attempt_wall_s']=10
        c['on_task_failure']='pause';c['retry']['retry_on']=[]
        s=new_session(self.root,c);l=read_json(s/'ledger.json')
        for _ in range(2):
            l['state']='pending';begin_attempt(s,c,l)
            commit_attempt(c,l,result('completed','failure','radar_clearance_guard'))
        write_json(s/'ledger.json',l)
        old_plan=read_json(s/'plan.json');old_results=copy.deepcopy(l['attempts'])
        for name in ('run_simulation.py','sim_core/supervisor.py','sim_core/tests/test_runner.py'):
            with (self.root/name).open('a') as f:f.write('\n# launcher upgrade fixture\n')
        new=set_retry_schedule(self.root,s,3,5)
        self.assertEqual(new['revisions'][0]['source_hashes'],old_plan['source_hashes'])
        self.assertEqual(read_json(s/'ledger.json')['attempts'],old_results)
        check_sources(self.root,new)
        l=read_json(s/'ledger.json');l['state']='pending'
        prepare_retry_schedule(new['config'],l,new)
        self.assertEqual(l['next_attempt'],3)
        (self.root/'sim_core/runtime.py').write_text('# changed public I/O\n')
        before=(s/'plan.json').read_bytes()
        with self.assertRaises(ValueError):set_retry_schedule(self.root,s,3,5)
        self.assertEqual((s/'plan.json').read_bytes(),before)
    def test_retry_multi_pass_repetitions_preserve_trial_ids_parameters(self):
        c,s,plan,exe=self.retry_session(cases=('A','B'),main_attempts=1,total_attempts=2)
        c['schedule'].update(repetitions_per_case=2,repeat_set=dict(mode='fixed',count=2))
        c['schedule']['cases'][0]['parameters']={'tag':'A'}
        c['schedule']['cases'][1]['parameters']={'tag':'B'}
        plan.update(config=c,config_hash=json_hash(c));write_json(s/'plan.json',plan)
        calls=[];seen=set()
        def fake(root,session,attempt,a,config,executable,first_role):
            calls.append((a['trial_id'],a['retry_phase']))
            self.assertEqual(a['parameters']['tag'],a['case_id'])
            ok=a['case_id']=='B' or a['trial_id'] in seen
            seen.add(a['trial_id'])
            return result('completed','success' if ok else 'failure','done' if ok else 'radar_clearance_guard')
        with patch('sim_core.supervisor.active_simulators',return_value=[]):
            self.assertEqual(supervise(self.root,s,exe,attempt_runner=fake),0)
        l=read_json(s/'ledger.json')
        self.assertEqual(len(l['trial_results']),8)
        self.assertEqual(len(calls),12)
        self.assertTrue(all(phase=='main' for _,phase in calls[:8]))
        self.assertTrue(all(phase=='catch_up' for _,phase in calls[8:]))
        self.assertEqual([t for t,_ in calls[8:]],[t for t,_ in calls[:8] if '_A_' in t])
        self.assertEqual((l['pass'],l['case'],l['repetition']),(2,0,0))
    def test_retry_cli_upgrade_exhausted_budget_and_resume_without_reset(self):
        c=self.config();c['mode']='automatic';c['limits']['attempt_wall_s']=10
        c['retry'].update(max_retries_per_trial=0,on_exhausted='pause')
        s=new_session(self.root,c);l=read_json(s/'ledger.json')
        begin_attempt(s,c,l);commit_attempt(c,l,result('failed','not_assessed','startup_timeout'))
        write_json(s/'ledger.json',l)
        self.assertTrue(l['reason'].startswith('retry_exhausted'))
        exe=self.root/'SpaceCRAFT.exe';exe.write_text('offline process fixture')
        rel=s.relative_to(self.root).as_posix();calls=[]
        def fake(*args):
            calls.append(1);return result('failed','not_assessed','startup_timeout')
        real=supervise
        with patch('sim_core.supervisor.active_simulators',return_value=[]):
            self.assertEqual(main(['set-retries','--session',rel],self.root),0)
            with patch('sim_core.supervisor.supervise',side_effect=lambda *a,**k:real(*a,**k,attempt_runner=fake)):
                self.assertEqual(main(['resume','--session',rel,'--executable',str(exe)],self.root),0)
            self.assertEqual(main(['resume','--session',rel,'--executable',str(exe),'--retry'],self.root),2)
        l=read_json(s/'ledger.json')
        self.assertEqual(len(l['attempts']),5);self.assertEqual(len(calls),4)
        self.assertEqual(l['state'],'complete')
        self.assertEqual(l['trial_results'][0]['result']['reason'],'startup_timeout')
    def test_offline_analysis_hook_runs_after_completed_campaign(self):
        analyzer=self.root/'offline_analysis.py'
        analyzer.write_text("import argparse,json\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--session');p.add_argument('--show',action='store_true');a=p.parse_args()\n"
            "s=Path.cwd()/a.session\nassert json.loads((s/'ledger.json').read_text())['state']=='complete'\n"
            "(s/'analysis_ran.txt').write_text('offline after completion')\n")
        exe=self.root/'SpaceCRAFT.exe';exe.write_text('offline process fixture')
        def fake(*args):return result('completed','success','done')
        real=supervise
        with patch('sim_core.supervisor.active_simulators',return_value=[]),patch('sim_core.supervisor.supervise',
                side_effect=lambda *a,**k:real(*a,**k,attempt_runner=fake)):
            self.assertEqual(main(['auto','--entry','OriginalController.py','--executable',str(exe),
                                  '--analysis','offline_analysis.py'],self.root),0)
        active=read_json(self.root/'sim_core/state/active_session.json')
        s=self.root/active['session']
        self.assertTrue((s/'analysis_ran.txt').is_file())
        self.assertEqual(read_json(s/'plan.json')['post_analysis']['sha256'],analysis_spec(self.root,'offline_analysis.py')['sha256'])
    def test_offline_analysis_source_pin_and_pause_are_respected(self):
        c,s,l=self.session()
        analyzer=self.root/'offline_analysis.py';analyzer.write_text("raise RuntimeError('must not execute')\n")
        plan=read_json(s/'plan.json');plan['post_analysis']=analysis_spec(self.root,'offline_analysis.py')
        write_json(s/'plan.json',plan)
        self.assertEqual(run_post_analysis(self.root,s,2),2)
        analyzer.write_text("raise RuntimeError('modified analysis')\n")
        with self.assertRaises(ValueError):run_post_analysis(self.root,s,0)
        for name in ('Controller.py','OriginalController.py','../outside.py'):
            with self.assertRaises(ValueError):analysis_spec(self.root,name)
    @unittest.skipIf(os.name=="nt","POSIX subprocess fixture; native Windows smoke remains required")
    def test_actual_process_handshake_with_fake_spacecraft(self):
        c=self.config()
        c["mode"]="automatic"
        c["limits"].update(attempt_wall_s=5,startup_wall_s=2,shutdown_wall_s=1)
        c=validate(self.root,c)
        s=new_session(self.root,c)
        l=read_json(s/"ledger.json")
        a,assignment=begin_attempt(s,c,l)
        exe=self.root/"SpaceCRAFT.exe"
        exe.write_text("#!/usr/bin/env python3\nimport os,json,pathlib\n"
            "s=pathlib.Path(os.environ['STU4_SESSION'])\n"
            "a=s/os.environ['STU4_ATTEMPT']\n"
            "r={'execution_status':'completed','task_outcome':'success','reason':'fixture_done','token':os.environ['STU4_ATTEMPT_TOKEN']}\n"
            "(a/'result.json').write_text(json.dumps(r))\n")
        exe.chmod(0o755)
        value=execute_attempt(self.root,s,a,assignment,c,exe)
        self.assertEqual(value["reason"],"fixture_done")
        launch=read_json(a/"launch.json")
        self.assertIn("--sim=Local/Repos/STU4-Sim/Sims/COMPETITION_SIM.simconf",launch["command"])
    @unittest.skipIf(os.name=="nt","POSIX process fixture")
    def test_process_exit_zero_without_result_is_failure(self):
        c=self.config()
        c["mode"]="automatic"
        c["limits"].update(attempt_wall_s=3,startup_wall_s=1,shutdown_wall_s=.5)
        c=validate(self.root,c)
        s=new_session(self.root,c)
        l=read_json(s/"ledger.json")
        a,assignment=begin_attempt(s,c,l)
        exe=self.root/"SpaceCRAFT.exe"
        exe.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
        exe.chmod(0o755)
        value=execute_attempt(self.root,s,a,assignment,c,exe)
        self.assertEqual(value["reason"],"process_exit")
        self.assertEqual(value["execution_status"],"failed")
    @unittest.skipIf(os.name=="nt","POSIX process fixture")
    def test_live_heartbeat_does_not_hide_stalled_simulation(self):
        c=self.config()
        c["mode"]="automatic"
        c["limits"].update(attempt_wall_s=4,startup_wall_s=2,shutdown_wall_s=.5,clock_stall_wall_s=.25)
        c=validate(self.root,c)
        s=new_session(self.root,c)
        l=read_json(s/"ledger.json")
        a,assignment=begin_attempt(s,c,l)
        exe=self.root/"SpaceCRAFT.exe"
        exe.write_text("#!/usr/bin/env python3\nimport os,json,pathlib,time\n"
            "s=pathlib.Path(os.environ['STU4_SESSION']);a=s/os.environ['STU4_ATTEMPT'];token=os.environ['STU4_ATTEMPT_TOKEN']\n"
            "while not (s/'STOP_NOW').exists():\n"
            "    t=a/'temp.json';t.write_text(json.dumps({'token':token,'ready':True,'sim_s':0,'wall_unix_s':time.time()}));os.replace(t,a/'heartbeat.json');time.sleep(.03)\n"
            "(a/'result.json').write_text(json.dumps({'token':token,'execution_status':'interrupted','task_outcome':'not_assessed','reason':'operator_stop'}))\n")
        exe.chmod(0o755)
        value=execute_attempt(self.root,s,a,assignment,c,exe)
        self.assertEqual(value["reason"],"simulation_clock_stall")

if __name__ == "__main__":
    unittest.main()
