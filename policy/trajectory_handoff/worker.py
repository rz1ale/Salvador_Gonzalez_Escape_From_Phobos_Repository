"""One owned numeric subprocess. Reads only already-authorized public logs.

The subprocess has no SDK imports, sockets, native launcher or simulator writes.
Perception, model LPs and QPs cannot block the controller's FF+FB update loop.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from sim_core.common import read_json,write_json


class Worker:
    def __init__(self,root,model,directory,settings,role='planner'):
        if role not in ('planner','monitor'):raise ValueError('Unknown numeric worker role')
        self.role=role
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        for name in ('request.json','response.json','ready.json','STOP'):(self.directory/name).unlink(missing_ok=True)
        (self.directory/'owner_heartbeat').touch();write_json(self.directory/'settings.json',settings)
        env=os.environ.copy()
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        self.log=(self.directory/'worker.txt').open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-m','policy.trajectory_handoff.worker','--model',str(Path(model).resolve()),
            '--directory',str(self.directory.resolve()),'--role',role],cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT)
        self.started=time.monotonic();self.submitted=None;self.job_id=None;self.last_job=None;self.response_sequence=-1
    def ready(self):return (self.directory/'ready.json').is_file()
    def alive(self):return self.process.poll() is None
    def submit(self,job):
        if self.job_id is not None:raise RuntimeError('Only one numerical request may be pending')
        submitted=time.monotonic();value=dict(job,submitted_monotonic_s=submitted,
            deadline_monotonic_s=submitted+float(job['solve_wall_s']))
        write_json(self.directory/'request.json',value);self.job_id=job['job_id'];self.submitted=submitted
        self.response_sequence=-1
        self.last_job={k:value.get(k) for k in ('job_id','source_sim_s','solve_wall_s','submitted_monotonic_s',
            'deadline_monotonic_s','authorization_expires_sim_s')}
    def poll(self):
        (self.directory/'owner_heartbeat').touch();p=self.directory/'response.json'
        if not p.is_file():return None
        value=read_json(p)
        if value.get('job_id')!=self.job_id:return None
        sequence=value.get('response_sequence',0)
        if sequence<=self.response_sequence:return None
        self.response_sequence=sequence
        value['parent_received_monotonic_s']=time.monotonic()
        if value.get('is_final',True):self.job_id=None;self.submitted=None
        return value
    def diagnostic(self):
        try:progress=read_json(self.directory/'ready.json')
        except (OSError,ValueError) as exc:progress=dict(stage='progress_unavailable',details=str(exc))
        return dict(job_id=self.job_id,last_job=self.last_job,worker=progress,
            parent_elapsed_wall_s=time.monotonic()-self.submitted if self.submitted is not None else None)
    def close(self):
        try:
            (self.directory/'STOP').touch()
            if self.alive():
                try:self.process.wait(timeout=.3)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:self.process.wait(timeout=2.)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2.)
        finally:self.log.close()


class Tail:
    """Complete JSON lines only. A partial final write is retried on next poll."""
    def __init__(self,path):self.path=Path(path);self.offset=0;self.pending=[]
    def read(self):
        if not self.path.is_file():return
        with self.path.open('rb') as f:
            f.seek(self.offset)
            while True:
                pos=f.tell();line=f.readline()
                if not line or not line.endswith(b'\n'):self.offset=pos;break
                self.pending.append(json.loads(line.decode('utf-8-sig')));self.offset=f.tell()


class PerceptionDeadline(Exception):pass


class PublicPerception:
    def __init__(self,attempt,s,*,log=True):
        from obstacle_characterization.radar import RadarTracker,load_settings
        cfg=load_settings();cfg['forecast_horizon_s']=s['forecast_horizon_s']
        cfg['track_expiry_mode']=s.get('radar_track_expiry_mode','retain_occupancy')
        cfg['birth_initialization']=s.get('radar_birth_initialization','two_position')
        self.tracker=RadarTracker(cfg);self.states=Tail(Path(attempt)/'states.txt');self.radar=Tail(Path(attempt)/'radar.txt')
        self.frame=None;self.diagnostics=Path(attempt)/'obstacle_diagnostics.txt'
        self.last_log=-float('inf');self.log=bool(log)
    def update(self,until,*,deadline_wall_s=None,clock=time.monotonic):
        def checkpoint():
            if deadline_wall_s is not None and clock()>=deadline_wall_s:raise PerceptionDeadline()
        checkpoint()
        self.states.read();self.radar.read()
        while self.radar.pending and self.radar.pending[0]['read_sim_s']<=until+1e-8:
            checkpoint();row=self.radar.pending[0];when=row['read_sim_s']
            while self.states.pending and self.states.pending[0]['read_end_sim_s']<=when+1e-8:
                self.tracker.add_own_state(self.states.pending.pop(0))
            checkpoint()
            self.frame=self.tracker.update(row['observations'],when)
            self.radar.pending.pop(0)
        checkpoint()
        if self.log and self.frame is not None and self.frame['generated_sim_s']-self.last_log>=1.:
            f=self.frame
            with self.diagnostics.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time_sim_s=f['generated_sim_s'],sequence=f['sequence'],clock=f['clock'],
                    coverage=f['coverage'],diagnostics=f['diagnostics']))+'\n')
            self.last_log=f['generated_sim_s']
        return self.frame


def run_job(plant,reference,job,perception,s,*,solver=None,clock=time.monotonic,progress=None,on_assessment=None):
    """One soft deadline includes queue, public perception and numeric work.

    CPython's monotonic clock is shared across processes. Parent receipt/state/
    activation/expiry checks remain the hard execution gates. Atomic numeric
    operations can overrun a soft deadline; their actual timing stays visible.
    """
    if solver is None:
        from .online import solve as solver
    started=clock();budget=float(job['solve_wall_s']);submitted=float(job.get('submitted_monotonic_s',started))
    deadline=min(float(job.get('deadline_monotonic_s',submitted+budget)),submitted+budget)
    if not all(math.isfinite(v) for v in (budget,submitted,deadline)) or not 0<budget<=s['solve_wall_s']:
        raise ValueError('Invalid end-to-end job budget')
    identity={k:job.get(k) for k in ('job_id','generation','source_sim_s')}
    stage='queue';perception_started=perception_finished=solve_started=None
    before=dict(getattr(getattr(plant.model,'allocator',None),'stats',{}))
    def announce(name):
        if progress is not None:progress(name,job,submitted,deadline)
    def current_ready(value):
        now=clock();partial=dict(value)
        partial['timing']=dict(submitted_monotonic_s=submitted,worker_started_monotonic_s=started,
            deadline_monotonic_s=deadline,requested_budget_wall_s=budget,queue_wall_s=max(0.,started-submitted),
            perception_wall_s=max(0.,perception_finished-perception_started),
            numerical_wall_s=max(0.,now-solve_started),end_to_end_wall_s=max(0.,now-submitted),
            soft_deadline_overrun_wall_s=max(0.,now-deadline))
        if on_assessment is not None:on_assessment(partial)
    try:
        if clock()>=deadline:
            value=dict(identity,status='blocked',reason='job_budget_exhausted',budget_stage=stage,wall_s=0.)
        else:
            stage='perception';announce(stage);perception_started=clock()
            frame=perception.update(job['source_sim_s'],deadline_wall_s=deadline,clock=clock)
            perception_finished=clock();remaining=deadline-perception_finished
            if remaining<=0:raise PerceptionDeadline()
            stage='solve';announce(stage);solve_started=clock()
            adjusted=dict(job,solve_wall_s=min(s['solve_wall_s'],max(0.,deadline-solve_started)))
            if adjusted['solve_wall_s']<=0:
                value=dict(identity,status='blocked',reason='job_budget_exhausted',budget_stage=stage,wall_s=0.)
            elif on_assessment is not None:value=solver(plant,reference,adjusted,frame,s,on_current=current_ready)
            else:value=solver(plant,reference,adjusted,frame,s)
    except PerceptionDeadline:
        value=dict(identity,status='blocked',reason='job_budget_exhausted',budget_stage='perception',wall_s=0.)
    except Exception:
        value=dict(identity,status='worker_error',budget_stage=stage,details=traceback.format_exc(),wall_s=0.)
    finished=clock()
    if value.get('reason')=='numeric_budget_exhausted':value['budget_stage']='solve'
    value['timing']=dict(submitted_monotonic_s=submitted,worker_started_monotonic_s=started,
        deadline_monotonic_s=deadline,requested_budget_wall_s=budget,queue_wall_s=max(0.,started-submitted),
        perception_wall_s=max(0.,(perception_finished if perception_finished is not None else finished)-perception_started) if perception_started is not None else 0.,
        numerical_wall_s=max(0.,finished-solve_started) if solve_started is not None else 0.,
        end_to_end_wall_s=max(0.,finished-submitted),soft_deadline_overrun_wall_s=max(0.,finished-deadline))
    after=getattr(getattr(plant.model,'allocator',None),'stats',{})
    value['allocation_cache']={k:v-before.get(k,0) for k,v in after.items()}
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--role',choices=['planner','monitor'],default='planner')
    args=p.parse_args();folder=Path(args.directory);s=read_json(folder/'settings.json')
    from system_identification.model import Model
    from .reference import Reference
    from .core import Plant
    from .online import solve
    if args.role=='monitor':
        from .guard import monitor as solve
    plant=Plant(Model.load(args.model),s['integration_step_s'],certified_allocation=s.get('certified_allocation_cache',True))
    reference=Reference(Path(__file__).with_name('assets'),args.model);perception=PublicPerception(folder.parent,s,log=args.role=='monitor')
    def progress(stage,job=None,submitted=None,deadline=None):
        now=time.monotonic()
        write_json(folder/'ready.json',dict(pid=os.getpid(),numeric_only=True,stage=stage,role=args.role,
            job_id=job.get('job_id') if job else None,source_sim_s=job.get('source_sim_s') if job else None,
            stage_started_monotonic_s=now,submitted_monotonic_s=submitted,deadline_monotonic_s=deadline,
            remaining_budget_wall_s=deadline-now if deadline is not None else None))
    progress('idle');seen=None
    while not (folder/'STOP').exists():
        if time.time()-(folder/'owner_heartbeat').stat().st_mtime>10.:return 0
        path=folder/'request.json'
        if not path.is_file():time.sleep(.02);continue
        job=read_json(path)
        if job.get('job_id')==seen:time.sleep(.02);continue
        seen=job['job_id']
        def publish_assessment(value):
            write_json(folder/'response.json',dict(value,is_final=False,response_sequence=1,response_phase='current_path'))
        try:
            value=run_job(plant,reference,job,perception,s,solver=solve,progress=progress,on_assessment=publish_assessment)
        except Exception:
            value=dict(job_id=seen,generation=job.get('generation'),source_sim_s=job['source_sim_s'],
                status='worker_error',details=traceback.format_exc())
        progress('publishing',job,job.get('submitted_monotonic_s'),job.get('deadline_monotonic_s'))
        write_json(folder/'response.json',dict(value,is_final=True,response_sequence=2,response_phase='planning_result',worker_role=args.role));progress('idle')
    return 0


if __name__=='__main__':raise SystemExit(main())
