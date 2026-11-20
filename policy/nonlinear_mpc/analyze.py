"""Local post-run report; compatible with run_simulation.py --analysis."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sim_core.common import local_path, read_json, write_json
from policy.nonlinear_mpc.optimizer import Model, Predictor, score_win


def rows(path):
    if not path.is_file():return []
    with path.open(encoding='utf-8-sig') as f:return [json.loads(line) for line in f if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True);p.add_argument('--show',action='store_true')
    args=p.parse_args();session=local_path(ROOT,args.session)
    if not session.is_dir():raise ValueError('Session directory does not exist')
    output=ROOT/'policy'/'nonlinear_mpc'/'analysis_results'/session.name
    output.mkdir(parents=True,exist_ok=True)
    plan=read_json(session/'plan.json')
    model_rel=plan['config'].get('model')
    predictor=None;fuel_note='Selected model unavailable or changed; fuel replay not performed'
    if model_rel:
        model_path=local_path(ROOT,model_rel,must_exist=True)
        if hashlib.sha256(model_path.read_bytes()).hexdigest()==plan['source_hashes'].get(model_rel):
            predictor=Predictor(Model.load(model_path),.15)
            fuel_note='Public command replay with frozen model; no parameter regression'
    summaries=[];series=[]
    for result_path in sorted(session.rglob('result.json')):
        folder=result_path.parent;r=read_json(result_path)
        samples=rows(folder/'states.txt');solves=rows(folder/'solver_results.txt');telemetry=rows(folder/'policy_telemetry.txt')
        if not samples:continue
        end=samples[-1];metrics=r.get('metrics',{})
        last=telemetry[-1] if telemetry else {}
        settings_path=folder/'policy_settings.json'
        params=read_json(settings_path) if settings_path.is_file() else {}
        estimate=metrics.get('score_formula_estimate',last.get('score_formula_estimate'))
        entry=dict(attempt=folder.relative_to(session).as_posix(),execution_status=r.get('execution_status'),
            task_outcome=r.get('task_outcome'),reason=r.get('reason'),native_score_captured=bool(r.get('official_score_captured',False)),
            score_formula_estimate=estimate,simulation_time_s=end['read_end_sim_s'],
            distance_m=float(np.linalg.norm(np.array(end['position_P_m'])-end['target_P_m'])),
            fuel_kg=end['fuel_used_kg'],damage_J=end['damage_J'],solver_results=len(solves),
            accepted_plans=metrics.get('accepted_plans',last.get('accepted_plans',0)),
            activated_plans=metrics.get('activated_plans',last.get('activated_plans',0)),
            rejected_results=metrics.get('rejected_results',last.get('rejected_results',0)),
            nmpc_monitor_steps=metrics.get('nmpc_monitor_steps',last.get('nmpc_monitor_steps',0)),
            recovery_monitor_steps=metrics.get('recovery_monitor_steps',last.get('recovery_monitor_steps',0)))
        durations=[s['solve_wall_s'] for s in solves if 'solve_wall_s' in s]
        if durations:entry['solve_wall_median_p95_s']=np.percentile(durations,[50,95]).tolist()
        commands=rows(folder/'commands.txt')
        if predictor is not None and commands:
            t0=samples[0]['read_end_sim_s'];t1=end['read_end_sim_s'];predicted=0.
            ct=np.array([u['send_end_sim_s'] for u in commands]);last=0.
            for i,cmd in enumerate(commands):
                left=max(t0,float(ct[i]));right=min(t1,float(ct[i+1]) if i+1<len(ct) else t1)
                if right>left:
                    u=np.r_[cmd['linear_body_request_m_s2'],cmd['angular_body_request_rad_s2']]
                    predicted+=(right-left)*predictor.allocation(u)[1]
            observed=end['fuel_used_kg']-samples[0]['fuel_used_kg']
            entry.update(predicted_fuel_increment_kg=float(predicted),observed_fuel_increment_kg=float(observed),
                fuel_observed_to_predicted_ratio=float(observed/predicted) if predicted>1e-9 else None)
        entry['fuel_validation']=fuel_note
        summaries.append(entry);series.append((entry['attempt'],samples,solves))
    if not summaries:raise ValueError('No completed attempt result/state pairs found')
    report=dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),session=str(session),attempts=summaries,
        evidence='Recorded public observations; formulas are SCORE_ESTIMATE. Native alert is authoritative.')
    write_json(output/'metrics.json',report)
    lines=['NMPC local report','Native scorer is authoritative. All score_formula values are estimates.','All attempts, including failures, are retained.','']
    for row in summaries:lines.append(json.dumps(row,sort_keys=True))
    (output/'metrics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    if not args.show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for name,samples,solves in series:
        t=np.array([s['read_end_sim_s'] for s in samples]);t-=t[0]
        d=[np.linalg.norm(np.array(s['position_P_m'])-s['target_P_m']) for s in samples]
        axes[0,0].plot(t,d,label=name[-24:]);axes[0,1].plot(t,[s['fuel_used_kg'] for s in samples])
        axes[1,0].plot(t,[s['damage_J'] for s in samples])
        valid=[s for s in solves if 'solve_wall_s' in s]
        if valid:axes[1,1].plot([s['source_sim_s']-samples[0]['read_end_sim_s'] for s in valid],[s['solve_wall_s'] for s in valid],'.-')
    for ax,title in zip(axes.ravel(),('Distance to target [m]','Public fuel [kg]','Public damage [J]','Numeric solve wall time [s]')):
        ax.set_title(title);ax.set_xlabel('Time since first logged state [s]');ax.grid(True,alpha=.3)
    axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'run_metrics.png',dpi=150)
    if args.show:plt.show()
    plt.close(fig)
    print(json.dumps(dict(output=str(output),attempts=len(summaries),official_score_captured=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
