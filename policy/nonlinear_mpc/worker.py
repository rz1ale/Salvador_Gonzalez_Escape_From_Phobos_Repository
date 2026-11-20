"""One owned numeric worker; atomic snapshot files, no SDK or native launcher."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from sim_core.common import read_json, write_json


class Worker:
    def __init__(self,root,model_path,directory):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        # Attempt directories are unique. Never inherit an old worker's receipt.
        for name in ('request.json','response.json','ready.json','STOP'):
            (self.directory/name).unlink(missing_ok=True)
        (self.directory/'owner_heartbeat').touch()
        env=os.environ.copy()
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[key]='1'
        self.log=(self.directory/'worker.txt').open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-m','policy.nonlinear_mpc.worker',
            '--model',str(Path(model_path).resolve()),'--directory',str(self.directory.resolve())],
            cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT)
        self.started=time.monotonic();self.submitted=None;self.job_id=None

    def ready(self):return (self.directory/'ready.json').is_file()
    def submit(self,job):
        if self.job_id is not None:raise RuntimeError('Only one numerical solve may be in flight')
        write_json(self.directory/'request.json',job)
        self.job_id=job['job_id'];self.submitted=time.monotonic()
    def poll(self):
        (self.directory/'owner_heartbeat').touch()
        path=self.directory/'response.json'
        if not path.is_file():return None
        value=read_json(path)
        if value.get('job_id')!=self.job_id:return None
        self.job_id=None;self.submitted=None
        return value
    def alive(self):return self.process.poll() is None
    def close(self):
        try:
            (self.directory/'STOP').touch()
            if self.process.poll() is None:
                try:self.process.wait(timeout=.3)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:self.process.wait(timeout=2.)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2.)
        finally:self.log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--directory',required=True)
    args=p.parse_args();folder=Path(args.directory)
    from .optimizer import Model, Predictor, solve
    model=Model.load(args.model);predictor=Predictor(model,.15)
    write_json(folder/'ready.json',dict(pid=os.getpid(),numeric_only=True))
    seen=None
    while not (folder/'STOP').exists():
        # A closed manual GUI can terminate the owner without Python finally.
        if time.time()-(folder/'owner_heartbeat').stat().st_mtime>10.:
            print('Owner heartbeat expired; numeric worker exiting',flush=True)
            return 0
        path=folder/'request.json'
        if not path.is_file():time.sleep(.03);continue
        job=read_json(path)
        if job.get('job_id')==seen:time.sleep(.03);continue
        seen=job['job_id']
        try:
            predictor.step=job['settings']['integration_step_s']
            value=solve(predictor,job)
        except Exception:
            value=dict(job_id=seen,generation=job.get('generation'),status='worker_error',details=traceback.format_exc())
        write_json(folder/'response.json',value)
    return 0


if __name__=='__main__':raise SystemExit(main())
