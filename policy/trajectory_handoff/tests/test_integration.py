"""Real numeric subprocess + mathematical plant; no native simulator/SDK.

About 40 seconds of wall time. Uses sim_core.runtime.execute_entry to exercise
the standalone entry loader, public log tailing, IPC, takeover and cleanup.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from system_identification.model import Model,matrix
from policy.trajectory_handoff.reference import Reference
from policy.trajectory_handoff import worker as worker_module
from sim_core.runtime import execute_entry

ROOT=Path(__file__).resolve().parents[3]


class Finished(BaseException):pass


class MathematicalContext:
    def __init__(self,output):
        self.root=ROOT;self.output_dir=Path(output);self.model_path=ROOT/'nominal_model.json'
        self.parameters={'task':'smoke','smoke_duration_s':15.,'smoke_takeovers':1}
        self.config={'entry':'policy/trajectory_handoff/entry.py'}
        self.assignment={'token':'mathematical_smoke_actual_loader'}
        self.reference=Reference(ROOT/'policy/trajectory_handoff/assets',self.model_path)
        self.m=Model.load(self.model_path);self.x=self.reference.states[0].copy();self.sim=0.
        self.command_times=[-1.];self.commands=[np.zeros(6)];self.fuel=0.;self.result=None;self.closed=False
        self.t0=time.monotonic();self.center=self.reference.states[0,:3]+[300.,300.,0.];self.obstacle_velocity=np.zeros(3)
    def clock(self):return time.monotonic()-self.t0
    def emit(self,name,value):
        with (self.output_dir/name).open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
    def advance(self):
        now=self.clock()
        if now>self.sim:
            self.x=self.m.rollout(self.x,self.sim,[now],self.command_times,np.asarray(self.commands))[-1];self.sim=now
    def checkpoint(self,**kw):pass
    def request_stop_after_task(self):pass
    def event(self,name,**value):self.emit('events.txt',dict(event=name,**value))
    def state(self):
        self.advance();x=self.x;q=x[6:10];p=x[:3]+matrix(q).T@self.m.ell
        packet=dict(read_begin_sim_s=self.sim,read_end_sim_s=self.sim,position_P_m=p.tolist(),
            velocity_P_m_s=(x[3:6]+self.m.vbias).tolist(),q_PB_xyzw_reported=q.tolist(),
            omega_relP_P_rad_s=(matrix(q).T@(x[10:13]+self.m.wbias)).tolist(),
            target_P_m=self.reference.manifest['target_P_m'],fuel_used_kg=self.fuel,damage_J=0.)
        self.emit('states.txt',packet);return packet
    def radar(self,n):
        self.advance();p=self.x[:3]+matrix(self.x[6:10]).T@self.m.ell;r=self.center+self.obstacle_velocity*self.sim-p
        radial=float((self.obstacle_velocity-self.x[3:6])@(r/np.linalg.norm(r)))
        row=np.array([r,[np.linalg.norm(r)-1.,radial,2.],[self.sim,0.,0.]])
        self.emit('radar.txt',dict(read_sim_s=self.sim,observations=[row.tolist()]));return [row]
    def command(self,a,b):
        self.advance();u=np.r_[a,b];self.command_times.append(self.sim);self.commands.append(u)
        rec=dict(send_begin_sim_s=self.sim,send_end_sim_s=self.sim,linear_body_request_m_s2=list(a),angular_body_request_rad_s2=list(b))
        self.emit('commands.txt',rec);return rec
    def sleep(self,seconds):time.sleep(seconds)
    def finish(self,**value):self.result=value;raise Finished()


class IntegrationTests(unittest.TestCase):
    def mathematical_smoke(self,duration,takeovers,*,moving_hazard=False):
        workers=[];original=worker_module.Worker
        def make(*args):
            w=original(*args);workers.append(w);return w
        with tempfile.TemporaryDirectory() as tmp:
            c=MathematicalContext(tmp)
            c.parameters.update(smoke_duration_s=duration,smoke_takeovers=max(1,takeovers),smoke_takeover=bool(takeovers))
            if moving_hazard:
                direction=c.reference.at(20.)[0][:3]-c.x[:3];direction/=np.linalg.norm(direction)
                c.center=c.x[:3]+matrix(c.x[6:10]).T@c.m.ell+60.*direction;c.obstacle_velocity=-6.*direction
            with patch.object(worker_module,'Worker',make):
                with self.assertRaises(Finished):execute_entry(c,{})
            path=Path(tmp)/'solver_results.txt'
            responses=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
            if not c.result.get('smoke_passed'):
                path=Path(tmp)/'solver_results.txt'
                responses=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
                diagnostic=[{k:r.get(k) for k in ('status','reason','wall_s')} for r in responses]
                self.fail(str(c.result)+' responses='+str(diagnostic)+' worker='+str((Path(tmp)/'numeric_worker'/'worker.txt').read_text()))
            self.assertGreaterEqual(c.result['takeovers'],takeovers)
            if moving_hazard:self.assertGreaterEqual(c.result['guard_takeovers'],1)
            self.assertTrue(all('timing' in r and 'received_sim_s' in r for r in responses))
            self.assertTrue(all(r['timing']['queue_wall_s']>=0 for r in responses))
            self.assertGreater(c.result['commands'],30)
            self.assertTrue(all(not w.alive() for w in workers))
            self.assertEqual(len(workers),2)
            self.assertEqual({w.role for w in workers},{'monitor','planner'})
            monitor=[json.loads(x) for x in (Path(tmp)/'monitor_results.txt').read_text().splitlines()]
            self.assertGreaterEqual(len(monitor),3)
            self.assertTrue(any(r.get('status')=='monitor_clear' for r in monitor))
            # Independent monitor must publish while a mission job is running.
            active_windows=[(r['timing']['worker_started_monotonic_s'],r['timing']['submitted_monotonic_s']+r['timing']['end_to_end_wall_s']) for r in responses]
            self.assertTrue(any(a<m['parent_received_monotonic_s']<b for a,b in active_windows for m in monitor))

    def test_shared_runner_entry_and_real_worker_complete_mathematical_smoke(self):
        self.mathematical_smoke(15.,1)

    def test_repeated_handoffs_keep_streaming_commands_and_renew_active_paths(self):
        self.mathematical_smoke(30.,3)

    def test_moving_public_radar_hazard_triggers_local_handover(self):
        # Synthetic radar + mathematical vehicle only; no contact physics.
        self.mathematical_smoke(15.,0,moving_hazard=True)

    def test_plan_expiry_and_failed_replans_continue_with_explicit_coast(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=MathematicalContext(tmp);c.parameters={'task':'smoke','smoke_duration_s':12.,'smoke_takeover':False};clock=[0.]
            c.clock=lambda:clock[0]
            c.sleep=lambda seconds:clock.__setitem__(0,round(clock[0]+.1,6))
            state=c.x.tolist()
            class StubWorker:
                def __init__(self,*args):
                    self.started=time.monotonic();self.submitted=None;self.job_id=None;self.role=args[-1]
                    self.directory=Path(args[2]);self.directory.mkdir();self.calls=0
                def ready(self):return True
                def alive(self):return True
                def submit(self,job):self.job_id=job['job_id'];self.job=job
                def poll(self):
                    if self.job_id is None:return None
                    self.job_id=None;self.calls+=1
                    if self.role=='monitor':
                        now=self.job['source_sim_s']
                        return dict(job_id=self.job['job_id'],generation=self.job['generation'],source_sim_s=now,status='monitor_clear',
                            observation_available=True,valid_until_sim_s=now+3.,prefix_times=[now,now+1.2],prefix_states=[self.job['state']]*2)
                    if self.calls>1:return dict(job_id=self.job['job_id'],generation=self.job['generation'],source_sim_s=self.job['source_sim_s'],status='blocked',reason='numeric_budget_exhausted')
                    return dict(job_id=self.job['job_id'],generation=0,source_sim_s=0.,status='mpc_feasible',valid_until_sim_s=4.,
                        activation_sim_s=.2,prefix_times=[0.,.2],prefix_states=[state,state],
                        plan=dict(times=[.2,.4],states=[state,state],commands=[np.zeros(6).tolist()],gains=[np.zeros((6,12)).tolist()],phase=.2))
                def close(self):pass
            with patch.object(worker_module,'Worker',StubWorker):
                with self.assertRaises(Finished):execute_entry(c,{})
            self.assertEqual(c.result['reason'],'handoff_smoke_complete')
            self.assertTrue(c.result['smoke_passed']);self.assertEqual(c.result['takeovers'],1)
            self.assertGreaterEqual(c.result['planner_failures'],1)
            self.assertGreaterEqual(c.result['explicit_coasts'],1)
            self.assertGreater(max(c.command_times),10.)
            self.assertTrue(all(np.linalg.norm(u[:3])==0 for t,u in zip(c.command_times,c.commands) if t>=.4))


if __name__=='__main__':unittest.main(verbosity=2)
