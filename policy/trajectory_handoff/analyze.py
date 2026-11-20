"""Local post-run summary; accepts the shared runner's --session/--show hook."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np


ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def records(path):
    if not path.is_file():return []
    lines=path.read_text(encoding='utf-8-sig').splitlines(keepends=True);out=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:out.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not line.endswith('\n'):break
            raise
    return out


def response_records(attempt):
    received=records(attempt/'solver_results.txt')
    identity=lambda r:(r.get('job_id'),r.get('response_sequence',0))
    ids={identity(r) for r in received}
    rows=[dict(r,recorded_response_received=True) for r in received]
    latest=attempt/'numeric_worker/response.json'
    if latest.is_file():
        value=json.loads(latest.read_text(encoding='utf-8-sig'))
        if identity(value) not in ids and 'source_sim_s' in value:
            rows.append(dict(value,recorded_response_received=False))
    return sorted(rows,key=lambda r:(r['source_sim_s'],r.get('response_sequence',0)))


def planning_responses(responses):
    """Replay one final/latest version per job; keep the interim receipt visible."""
    jobs={}
    for response in responses:
        key=response['job_id'];previous=jobs.get(key)
        interim_received=bool(previous and previous.get('interim_received')) or bool(
            not response.get('is_final',True) and response.get('recorded_response_received'))
        if previous is None or response.get('response_sequence',0)>=previous.get('response_sequence',0):
            jobs[key]=dict(response,interim_received=interim_received)
        else:previous['interim_received']=interim_received
    return sorted(jobs.values(),key=lambda r:r['source_sim_s'])


def active_plan_at(now,generation,responses,events,latest):
    """Recover only a plan present in public responses and a recorded activation."""
    if latest.get('generation')==generation and abs(latest.get('source_sim_s',-1)-now)<1e-6:
        return latest.get('active_plan'),True
    activation=next((e for e in reversed(events) if e.get('event') in ('handoff_activated','explicit_coast_selected')
        and e.get('generation')==generation and e.get('sim_s',float('inf'))<=now+.25),None)
    if activation is None:return None,generation==0
    if activation.get('plan') is not None:return activation['plan'],True
    if activation.get('mode')=='nominal':return None,True
    candidate=next((r for r in reversed(responses) if r.get('recorded_response_received')
        and r.get('generation')==generation-1 and r.get('status')=='mpc_feasible'
        and 'plan' in r and 0<=activation['sim_s']-r['activation_sim_s']<.4),None)
    return (candidate['plan'],True) if candidate else (None,False)


def recorded_latency(solver,telemetry):
    rows=[]
    for k,r in enumerate(solver,1):
        receipt=r.get('received_sim_s');method='explicit_receipt'
        if receipt is None:
            method='telemetry_counter_transition'
            i=next((i for i,t in enumerate(telemetry) if t.get('worker_results',0)>=k),None)
            receipt=telemetry[max(0,i-1)]['sim_s'] if i is not None else None
        age=receipt-r['source_sim_s'] if receipt is not None else None
        outside=max(0.,age-r['wall_s']) if age is not None and r.get('wall_s') is not None else None
        rows.append(dict(source_sim_s=r['source_sim_s'],receipt_sim_s=receipt,receipt_method=method,
            source_to_receipt_s=age,numeric_wall_s=r.get('wall_s'),outside_numeric_s=outside,timing=r.get('timing')))
    ages=[r['source_to_receipt_s'] for r in rows if r['source_to_receipt_s'] is not None]
    outside=[r['outside_numeric_s'] for r in rows if r['outside_numeric_s'] is not None]
    return dict(received_jobs=len(solver),maximum_source_to_receipt_s=max(ages,default=None),
        median_outside_numeric_s=float(np.median(outside)) if outside else None,
        outside_numeric_includes_queue_perception_IPC_and_polling=True,rows=rows)


def logged_receipt(response,generation,states,model,s,wall_s):
    from policy.trajectory_handoff.reference import latent_state,packet_row
    from policy.trajectory_handoff.entry import response_usable
    estimated=response['source_sim_s']+wall_s
    packet=next((p for p in states if p['read_end_sim_s']>=estimated),None)
    if packet is None:return None,'Public log ends before estimated receipt'
    stamp=packet['read_end_sim_s'];history=[p for p in states if stamp-1<=p['read_end_sim_s']<=stamp]
    actual=latent_state([p['read_end_sim_s'] for p in history],[packet_row(p) for p in history],model.p)
    return response_usable(response,generation,stamp,actual,s)


def pipeline_replay(attempt,model_path,reference,s,jobs,states):
    """Paired LOCAL pipeline timing; no original native CPU contention or IPC."""
    from policy.trajectory_handoff.core import Model,Plant
    from policy.trajectory_handoff.worker import PublicPerception,run_job
    outputs={}
    for enabled in (False,True):
        plant=Plant(Model.load(model_path),s['integration_step_s'],certified_allocation=enabled)
        perception=PublicPerception(attempt,s,log=False);rows=[]
        for job in jobs:
            early=[]
            result=run_job(plant,reference,job,perception,s,on_assessment=early.append)
            timing=result['timing'];accepted,reason=logged_receipt(result,job['generation'],states,plant.model,s,timing['end_to_end_wall_s'])
            current=early[0] if early else None
            current_accepted,current_reason=logged_receipt(current,job['generation'],states,plant.model,s,
                current['timing']['end_to_end_wall_s']) if current else (None,None)
            rows.append(dict(source_sim_s=job['source_sim_s'],status=result['status'],reason=result.get('reason'),
                timing=timing,allocation_cache=result.get('allocation_cache'),
                authorization_scope=result.get('authorization_scope'),planning_reason=result.get('planning_reason'),
                current_path_assessment=result.get('current_path_assessment'),
                interim_timing=current.get('timing') if current else None,
                interim_accepted_at_estimated_logged_receipt=current_accepted,interim_receipt_reason=current_reason,
                accepted_at_estimated_logged_receipt=accepted,receipt_reason=reason))
        totals=[r['timing']['end_to_end_wall_s'] for r in rows]
        outputs['certified_cache' if enabled else 'declared_solver']=dict(rows=rows,
            median_end_to_end_wall_s=float(np.median(totals)) if totals else None,
            maximum_end_to_end_wall_s=max(totals,default=None))
    a=outputs['declared_solver']['median_end_to_end_wall_s'];b=outputs['certified_cache']['median_end_to_end_wall_s']
    return dict(local_host_only=True,native_CPU_contention_or_IPC_reproduced=False,
        paired_median_speedup=a/b if a and b else None,**outputs)


def replay_planning(attempt,session):
    """Read-only replay/counterfactual of recorded nominal and active-plan states.

    Uses only public own-state/radar/command logs and the matching model/cache.
    Active plans require their recorded activation and public command arrays.
    No later flight outcome is inferred from the replay.
    """
    from policy.trajectory_handoff.reference import Reference,sha256,packet_row,latent_state
    from policy.trajectory_handoff.core import Model,Plant
    from policy.trajectory_handoff.worker import PublicPerception
    from policy.trajectory_handoff.online import assess,forecast,solve,law
    from policy.trajectory_handoff.entry import response_usable
    plan=json.loads((session/'plan.json').read_text(encoding='utf-8-sig'))
    model_path=ROOT/plan['config']['model'];assets=Path(__file__).with_name('assets')
    pins=plan['source_hashes']
    if sha256(model_path)!=pins.get(plan['config']['model']):raise ValueError('Replay needs the originally selected model bytes')
    if sha256(assets/'nominal_trajectory.npz')!=pins.get('policy/trajectory_handoff/assets/nominal_trajectory.npz'):
        raise ValueError('Replay needs the originally selected trajectory cache')
    s=json.loads((attempt/'policy_settings.json').read_text(encoding='utf-8-sig'))
    from policy.trajectory_handoff.check import settings
    ref=Reference(assets,model_path);plant=Plant(Model.load(model_path),s['integration_step_s'])
    corrected_settings=dict(settings(),**s)
    corrected_settings.update(obstacle_uncertainty_geometry='ellipsoid',radar_track_expiry_mode='retain_occupancy',
        radar_birth_initialization='two_position',forecast_horizon_s=max(s.get('forecast_horizon_s',20.),45.))
    tracker=PublicPerception(attempt,corrected_settings,log=False)
    recorded_settings=dict(s,radar_track_expiry_mode=s.get('radar_track_expiry_mode','drop_unassessed'),
        radar_birth_initialization=s.get('radar_birth_initialization','radial_prior'))
    recorded_tracker=PublicPerception(attempt,recorded_settings,log=False)
    telemetry=records(attempt/'policy_telemetry.txt');states=records(attempt/'states.txt');commands=records(attempt/'commands.txt')
    all_responses=response_records(attempt);responses=planning_responses(all_responses)
    events=records(attempt/'events.txt');rows=[];pipeline_jobs=[]
    latest_path=attempt/'numeric_worker/request.json'
    latest=json.loads(latest_path.read_text(encoding='utf-8-sig')) if latest_path.is_file() else {}
    assignment_path=attempt/'assignment.json'
    params=json.loads(assignment_path.read_text(encoding='utf-8-sig')).get('parameters',{}) if assignment_path.is_file() else {}
    for r in responses:
        now=r['source_sim_s'];frame=tracker.update(now);recorded_frame=recorded_tracker.update(now)
        if frame is None or 'prefix_states' not in r:
            rows.append(dict(source_sim_s=now,skipped='No recorded prefix/perception'));continue
        row=min(telemetry,key=lambda x:abs(x['sim_s']-now))
        active,recovered=active_plan_at(now,r['generation'],all_responses,events,latest)
        if not recovered:
            rows.append(dict(source_sim_s=now,skipped='No public activation/plan pair for this generation'));continue
        phase=now-(row['sim_s']-row['phase_s'])
        cmd=next((c for c in reversed(commands) if c['send_end_sim_s']<=now),None)
        if cmd is None:
            rows.append(dict(source_sim_s=now,skipped='Missing causal command history'));continue
        old=cmd['linear_body_request_m_s2']+cmd['angular_body_request_rad_s2']
        first=min(states,key=lambda x:abs(x['read_end_sim_s']-telemetry[0]['sim_s']))
        launch=np.asarray(latest.get('launch_position_P',first['position_P_m']))
        up=np.asarray(latest.get('launch_up_P',launch/np.linalg.norm(launch)))
        pt=np.asarray(r['prefix_times']);px=np.asarray(r['prefix_states'])
        prev=law(ref,phase+pt[-2]-now,active,px[-2],pt[-2],s)
        horizon=s['mpc_stages']*s['mpc_stage_s']
        nt,nx,_,_=forecast(plant,ref,px[-1],pt[-1],pt[-1]+horizon,phase+pt[-1]-now,active,prev,s)
        geometry=r.get('prefix_assessment',{}).get('uncertainty_geometry',s.get('obstacle_uncertainty_geometry','sphere'))
        recorded_prefix=assess(plant,recorded_frame,pt,px,dict(s,obstacle_uncertainty_geometry=geometry),launch,up)
        recorded_continued=assess(plant,recorded_frame,nt,nx,dict(s,obstacle_uncertainty_geometry=geometry),launch,up)
        before=assess(plant,frame,nt,nx,dict(s,obstacle_uncertainty_geometry='sphere'),launch,up)
        after=assess(plant,frame,nt,nx,dict(s,obstacle_uncertainty_geometry='ellipsoid'),launch,up)
        job=dict(job_id=r['job_id'],generation=r['generation'],source_sim_s=now,state=px[0].tolist(),
            phase=phase,active_plan=active,last_command=old,launch_position_P=launch.tolist(),launch_up_P=up.tolist(),
            fuel_kg=first.get('fuel_used_kg',0.),damage_J=first.get('damage_J',0.),score_constants=ref.manifest['score_constants'],
            force_mpc=params.get('task')=='smoke' and params.get('smoke_takeover',True) and row.get('takeovers',0)<params.get('smoke_takeovers',1))
        job['solve_wall_s']=(r.get('timing') or {}).get('requested_budget_wall_s',r.get('solve_budget_wall_s',s['solve_wall_s']))
        source_state=min(states,key=lambda x:abs(x['read_end_sim_s']-now))
        job.update(fuel_kg=source_state['fuel_used_kg'],damage_J=source_state['damage_J'])
        if latest.get('job_id')==r['job_id']:
            # Exact last public request; its old cross-host monotonic timestamps
            # must NOT become deadlines in a local replay.
            for key in ('state','phase','active_plan','last_command','solve_wall_s','force_mpc'):
                if key in latest:job[key]=latest[key]
        pipeline_jobs.append(dict(job))
        corrected=solve(plant,ref,job,frame,corrected_settings)
        estimated_receipt=now+corrected.get('wall_s',0.)
        packet=next((p for p in states if p['read_end_sim_s']>=estimated_receipt),None)
        accepted=None;receipt_reason='Public log ends before estimated receipt'
        if packet is not None:
            stamp=packet['read_end_sim_s'];history=[p for p in states if stamp-1<=p['read_end_sim_s']<=stamp]
            actual=latent_state([p['read_end_sim_s'] for p in history],[packet_row(p) for p in history],plant.model.p)
            accepted,receipt_reason=response_usable(corrected,r['generation'],stamp,actual,s)
        report=dict(source_sim_s=now,objects=len(frame['objects']),coverage_conditions=frame['coverage']['conditions'],
            mode='linear_mpc' if active is not None else 'nominal',generation=r['generation'],
            retained_expired_tracks=frame['coverage'].get('retained_expired_tracks',0),
            retained_expired_unresolved_observations=frame['coverage'].get('retained_expired_unresolved_observations',0),
            birth_reinitializations_total=frame['diagnostics'].get('birth_reinitializations_total',0),
            recorded_status=r.get('status'),recorded_reason=r.get('reason'),
            recorded_response_received=r['recorded_response_received'],
            recorded_geometry=geometry,recorded_prefix_replay=recorded_prefix,recorded_continuation_replay=recorded_continued,
            continued_sphere=before,continued_ellipsoid=after,
            corrected_status=corrected['status'],corrected_reason=corrected.get('reason'),
            corrected_authorization_scope=corrected.get('authorization_scope'),
            corrected_assessed_until_sim_s=corrected.get('assessed_until_sim_s'),
            corrected_current_path_assessment=corrected.get('current_path_assessment'),
            corrected_planning_reason=corrected.get('planning_reason'),
            corrected_candidate=corrected.get('selected_candidate'),corrected_plan_assessment=corrected.get('plan_assessment'),
            corrected_activation_sim_s=corrected.get('activation_sim_s'),
            corrected_wall_s=corrected.get('wall_s'),accepted_at_estimated_logged_receipt=accepted,receipt_reason=receipt_reason,
            qp=corrected.get('qp'),feasible_start_checks=corrected.get('feasible_start_checks'))
        for key,replayed in [('prefix_assessment',recorded_prefix),('continued_assessment',recorded_continued)]:
            original=r.get(key) or {}
            if original.get('minimum_inflated_clearance_m') is not None:
                report['recorded_'+key+'_margin_error_m']=replayed['minimum_inflated_clearance_m']-original['minimum_inflated_clearance_m']
        rows.append(report)
    def intervention_record(row):
        return ({k:row.get(k) for k in ('source_sim_s','corrected_activation_sim_s','objects','retained_expired_tracks',
            'corrected_plan_assessment','accepted_at_estimated_logged_receipt','receipt_reason')} if row is not None else None)
    first=next((r for r in rows if r.get('corrected_status')=='mpc_feasible' and r.get('accepted_at_estimated_logged_receipt')),None)
    verified=next((r for r in rows if r.get('corrected_status')=='mpc_feasible'),None)
    return dict(counterfactual=True,input='Public logs only; recorded state reinitialization per planning job',
        model_and_cache_match=True,native_rerun=False,full_mission_success_established=False,
        earliest_accepted_intervention=intervention_record(first),earliest_verified_candidate=intervention_record(verified),
        last_logged_state_sim_s=states[-1]['read_end_sim_s'] if states else None,
        pipeline=pipeline_replay(attempt,model_path,ref,corrected_settings,pipeline_jobs,states),
        later_recorded_states_follow_original_controls=True,rows=rows)


def analyze(session,show=False,replay=False):
    folder=Path(__file__).with_name('analysis_results')/session.name;folder.mkdir(parents=True,exist_ok=True)
    attempts=[];plot_data=[]
    paths=sorted({p.parent for name in ('result.json','supervisor_result.json','assignment.json') for p in session.rglob(name)})
    for attempt in paths:
        path=attempt/'result.json';supervisor=attempt/'supervisor_result.json'
        result=(json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else
                json.loads(supervisor.read_text(encoding='utf-8-sig')) if supervisor.is_file() else
                dict(execution_status='unknown',reason='no_result_record'))
        states=records(attempt/'states.txt');telemetry=records(attempt/'policy_telemetry.txt');solver=records(attempt/'solver_results.txt')
        monitor=records(attempt/'monitor_results.txt');events=records(attempt/'events.txt')
        counts={k:sum(r.get('status')==k for r in solver) for k in sorted({r.get('status','missing') for r in solver})}
        item=dict(attempt=str(attempt.relative_to(session)),result=result,worker_status_counts=counts,
            monitor_status_counts={k:sum(r.get('status')==k for r in monitor) for k in sorted({r.get('status','missing') for r in monitor})},
            monitor_received=len(monitor),
            risk_fallback_activations=sum(e.get('event')=='handoff_activated' and bool(e.get('risk_accepted')) for e in events),
            explicit_coast_selections=sum(e.get('event')=='explicit_coast_selected' for e in events),
            recorded_latency=recorded_latency(solver,telemetry),
            worker_reason_counts={k:sum(r.get('reason')==k for r in solver) for k in sorted({r['reason'] for r in solver if r.get('reason')})},
            stop_requested_by_controller=bool(result.get('stop_sim_requested')),
            worker_wall_s_max=max([r.get('wall_s',0.) for r in solver],default=None),
            minimum_recorded_distance_m=min([r['distance_m'] for r in telemetry],default=None),
            last_public_fuel_kg=states[-1]['fuel_used_kg'] if states else None,
            last_public_damage_J=states[-1]['damage_J'] if states else None,
            official_score_captured=False,native_crash_inferred=False)
        if replay and (attempt/'policy_settings.json').is_file():
            try:item['replay']=replay_planning(attempt,session)
            except ValueError as exc:item['replay']=dict(error=str(exc),native_rerun=False)
        attempts.append(item);plot_data.append((attempt.name,states,telemetry,solver))
    report=dict(session=str(session),attempts=attempts,all_physical_debris_avoidance_qualified=False)
    (folder/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['Trajectory handoff run analysis','Native scorer is authoritative; estimates use public simulation time.']
    for a in attempts:
        result=a['result'];lines.extend(['',a['attempt'],json.dumps(result,sort_keys=True),
            'worker_status_counts = '+json.dumps(a['worker_status_counts']),
            'worker_reason_counts = '+json.dumps(a['worker_reason_counts']),
            'controller_requested_stop = '+str(a['stop_requested_by_controller']),
            'worker_wall_s_max = '+str(a['worker_wall_s_max'])])
        lines.append('recorded_latency = '+json.dumps({k:v for k,v in a['recorded_latency'].items() if k!='rows'}))
        lines.append('monitor_status_counts = '+json.dumps(a['monitor_status_counts']))
        lines.append('risk_fallback_activations = '+str(a['risk_fallback_activations']))
        lines.append('explicit_coast_selections = '+str(a['explicit_coast_selections']))
        if 'replay' in a:
            lines.append('COUNTERFACTUAL REPLAY (does not establish a later flight outcome)')
            lines.append('earliest_accepted_intervention = '+json.dumps(a['replay'].get('earliest_accepted_intervention')))
            lines.append('earliest_verified_candidate = '+json.dumps(a['replay'].get('earliest_verified_candidate')))
            pipeline=a['replay'].get('pipeline',{})
            lines.append('LOCAL pipeline median speedup = '+str(pipeline.get('paired_median_speedup')))
            for name in ('declared_solver','certified_cache'):
                if pipeline.get(name,{}).get('rows'):lines.append('LOCAL final pipeline '+name+' = '+json.dumps(pipeline[name]['rows'][-1]))
            for r in a['replay'].get('rows',[]):
                lines.append(json.dumps({k:r.get(k) for k in ('source_sim_s','mode','generation','recorded_reason','corrected_status',
                    'corrected_authorization_scope','corrected_assessed_until_sim_s','corrected_current_path_assessment','corrected_planning_reason',
                    'corrected_candidate','accepted_at_estimated_logged_receipt','retained_expired_tracks',
                    'birth_reinitializations_total',
                    'recorded_prefix_assessment_margin_error_m','recorded_continued_assessment_margin_error_m')}))
    (folder/'metrics.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:
        import matplotlib
        if not show:matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        for label,states,telemetry,solves in plot_data:
            if states:
                ts=np.array([r['read_end_sim_s'] for r in states]);p=np.array([r['position_P_m'] for r in states]);goal=np.array([r['target_P_m'] for r in states])
                ax[0,0].plot(ts,np.linalg.norm(p-goal,axis=1),label=label)
                ax[0,1].plot(ts,[r['fuel_used_kg'] for r in states]);ax[0,2].plot(ts,[r['damage_J'] for r in states])
            if telemetry:
                t=[r['sim_s'] for r in telemetry]
                ax[1,0].step(t,[int(r['mode'] in ('linear_mpc','mission_mpc','local_avoidance')) for r in telemetry],where='post')
                ax[1,1].plot(t,[r['policy_work_wall_s'] for r in telemetry])
            if solves:ax[1,2].plot([r['source_sim_s'] for r in solves],[r.get('wall_s',np.nan) for r in solves],'.-')
        for a,title in zip(ax.flat,['Distance to target [m]','Public fuel [kg]','Public damage [J]',
                'Online takeover active [0/1]','Controller work wall time [s]','Numeric solve wall time [s]']):
            a.set_title(title);a.set_xlabel('Simulation time [s]');a.grid(True,alpha=.3)
        if any(states for _,states,_,_ in plot_data):ax[0,0].legend(fontsize=6)
        fig.savefig(folder/'run_metrics.png',dpi=150)
        if show:plt.show()
        plt.close(fig)
    except ImportError:
        lines.append('Matplotlib unavailable; metrics still generated.')
    print(json.dumps(dict(analysis_directory=str(folder),attempts=len(attempts)),indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--show',action='store_true')
    p.add_argument('--replay',action='store_true',help='Replay public nominal/active planning states using matching model/cache and recorded activations')
    a=p.parse_args();session=(ROOT/a.session).resolve()
    if not session.is_relative_to(ROOT) or not session.is_dir():raise ValueError('Session must be inside this solution clone')
    analyze(session,a.show,a.replay);return 0


if __name__=='__main__':raise SystemExit(main())
