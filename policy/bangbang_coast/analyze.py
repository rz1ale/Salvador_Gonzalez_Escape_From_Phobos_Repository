"""Read-only offline log summary for bangbang_coast.

Usage: python policy/bangbang_coast/analyze.py --session policy/bangbang_coast/run_logs/SESSION_ID
Creates analysis_results/<SESSION_ID>/metrics.txt, not inside any raw flight logs.
Does not query simulator, private states or native scorer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def lines(path):
    if not path.exists():
        return []
    records=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        try: records.append(json.loads(line))
        except json.JSONDecodeError:continue
    return records


def analyze_attempt(folder,score):
    states=lines(folder/'states.txt')
    events=lines(folder/'events.txt')
    result_path=folder/'result.json'
    result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.is_file() else {}
    arrival=[e for e in events if e.get('event')=='arrival_observed']
    plan=[e for e in events if e.get('event')=='nominal_plan']
    if not states:
        return {'attempt':str(folder),'status':'NO_STATES','result':result.get('reason','unknown')}
    t=np.array([s['read_end_sim_s'] for s in states],float)
    d=np.array([np.linalg.norm(np.asarray(s['position_P_m'])-np.asarray(s['target_P_m'])) for s in states])
    f=np.array([s['fuel_used_kg'] for s in states],float)
    damage=np.array([s['damage_J'] for s in states],float)
    speed=np.array([np.linalg.norm(s['velocity_P_m_s']) for s in states],float)
    reached=bool(arrival)
    best_plan=plan[-1] if plan else {}
    report={'attempt':str(folder),'status': 'OBSERVED_ARRIVAL' if reached else 'NO_OBSERVED_ARRIVAL',
            'result':result.get('reason','unknown'), 'first_time_s':float(t[0]),
            'last_time_s':float(t[-1]),'min_distance_m':float(d.min()),
            'last_distance_m':float(d[-1]),'fuel_used_kg_last':float(f[-1]),
            'damage_J_last':float(damage[-1]),'max_speed_m_s':float(speed.max()),
            'estimated_plan_score':best_plan.get('estimated_score'),
            'estimated_plan_arrival_sim_s':best_plan.get('estimated_arrival_sim_s')}
    if reached:
        final=arrival[-1]
        T=float(final['sim_s']) if 'sim_s' in final else float(t[-1])
        fuel=float(final['fuel_used_kg'])
        dmg=float(final['damage_J'])
        report['approximate_score_at_observed_arrival']=(50+100*(score['nominal_time_s']/max(T,1e-3) -
             fuel/score['nominal_fuel_kg'] -2*dmg/score['damage_threshold_J']))
        report['arrival_time_sim_s']=T
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True)
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    session=Path(args.session)
    if not session.is_absolute():session=root/session
    session=session.resolve()
    if not (session/'trials').is_dir():
        raise ValueError('Expected a shared-runner session with trials/ directory')
    settings=json.loads((root/'policy/bangbang_coast/settings.json').read_text(encoding='utf-8-sig'))
    score=settings['objective']['score']
    reports=[analyze_attempt(p,score) for p in sorted((session/'trials').rglob('attempt_*')) if p.is_dir()]
    out=root/'policy/bangbang_coast/analysis_results'/session.name
    out.mkdir(parents=True,exist_ok=True)
    with (out/'metrics.txt').open('w',encoding='utf-8') as f:
        f.write('BANGBANG COAST OFFLINE REPORT\n')
        f.write('NOTE\tEstimated scores only. Native scorer is authoritative.\n')
        f.write('session\t'+str(session)+'\n')
        for i,r in enumerate(reports):
            f.write('\nATTEMPT\t'+str(i+1)+'\n')
            for key,val in r.items():f.write(key+'\t'+str(val)+'\n')
    if args.show:
        for r in reports:print(json.dumps(r,indent=2))
    print('Wrote '+str(out/'metrics.txt'))
    return 0

if __name__=='__main__':
    sys.exit(main())
