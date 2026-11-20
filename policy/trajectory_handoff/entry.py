"""Shared-runner run(context) entry. All SDK I/O remains on this thread."""
from __future__ import annotations
import json
from pathlib import Path
import time
import uuid
import numpy as np
# The shared runner loads this file under a private standalone module name.
# Public entry files therefore use solution-root absolute package imports.
from policy.trajectory_handoff import VERSION
from policy.trajectory_handoff.check import preflight,settings
from policy.trajectory_handoff.reference import Reference,packet_row,latent_state,error,interpolate_state,clip_request,schedule_control,coast_plan,plan_end
from policy.trajectory_handoff.worker import Worker


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def plan_control(plan,state,when,reference,s):
    if plan is None:return None
    phase=plan.get('phase',0.)+when-plan['times'][0]
    return schedule_control(plan,state,when,reference,phase,s,check_tracking=True)


def response_usable(response,generation,now,state,s):
    if response.get('generation')!=generation:return False,'generation_changed'
    if response.get('status') not in ('nominal_clear','active_clear','rejoin_clear','mpc_feasible','monitor_clear','guard_feasible','guard_risk_fallback','monitor_hazard'):return False,response.get('reason',response.get('status'))
    if now>=response.get('valid_until_sim_s',-1):return False,'assessment_expired'
    ts=np.asarray(response['prefix_times']);xs=np.asarray(response['prefix_states'])
    if now<ts[0]-1e-6 or now>=ts[-1]-.03:return False,'activation_deadline_missed'
    predicted=interpolate_state(ts,xs,now);d=error(state,predicted)
    if (np.linalg.norm(d[:3])>s['plan_position_tolerance_m'] or np.linalg.norm(d[3:6])>s['plan_velocity_tolerance_m_s']
        or np.linalg.norm(d[6:9])>s['plan_attitude_tolerance_rad'] or np.linalg.norm(d[9:])>s['plan_rate_tolerance_rad_s']):
        return False,'prefix_prediction_deviated'
    return True,None


def worker_diagnostic(response,reason):
    """Compact cause in result.json/native stdout, without trajectory arrays."""
    fields=('admissible','minimum_inflated_clearance_m','critical_object_id',
            'collision_assessment','uncertainty_geometry','conditions','objects_assessed',
            'retained_expired_tracks','retained_expired_unresolved_observations')
    compact=lambda a:{k:a[k] for k in fields if k in a} if a else None
    return dict(reason=reason,status=response.get('status'),source_sim_s=response.get('source_sim_s'),
        worker_wall_s=response.get('wall_s'),solve_budget_wall_s=response.get('solve_budget_wall_s'),
        timing=response.get('timing'),budget_stage=response.get('budget_stage'),allocation_cache=response.get('allocation_cache'),
        prefix=compact(response.get('prefix_assessment')),
        continued=compact(response.get('continued_assessment')),
        qp_statuses=[r.get('status') for r in response.get('qp',[])],
        feasible_starts=[dict(translation_scale=r.get('translation_scale'),assessment=compact(r.get('assessment')))
                         for r in response.get('feasible_start_checks',[])])


def send_readiness(plan,when,send_now,s,reference=None,phase_origin=0.):
    if send_now-when>s['max_send_latency_s']:return 'send_state_too_old'
    if plan is not None and plan.get('kind')!='coast':
        times=np.asarray(plan['times'])
        i=int(np.searchsorted(times,when,side='right'))
        j=int(np.searchsorted(times,send_now,side='right'))
        if i!=j:return 'stage_changed_before_send'
    elif reference is not None:
        for times in (reference.command_times,reference.gain_times):
            if np.searchsorted(times,when-phase_origin,side='right')!=np.searchsorted(times,send_now-phase_origin,side='right'):
                return 'stage_changed_before_send'
    return None


def renewal_budget(settings,assessment,now):
    """Planning budget is independent of diagnostic monitor freshness."""
    return float(settings['solve_wall_s'])


def scheduled_snapshot(active,pending,now):
    if pending is None:return active
    next_plan=pending.get('plan') if pending['status']!='rejoin_clear' else None
    phase=pending.get('rejoin_phase',next_plan.get('phase',0.) if next_plan else 0.)
    end=plan_end(next_plan) if next_plan else now+600.
    return dict(kind='schedule',times=[now,end],current=active,next=next_plan,
        switch_sim_s=pending['activation_sim_s'],next_phase=phase)


def run(context):
    init_wall=time.monotonic();params=context.parameters
    if set(params)-{'task','smoke_duration_s','smoke_takeover','smoke_takeovers','overrides'}:raise ValueError('Unknown handoff parameter')
    task=params.get('task','mission');duration=float(params.get('smoke_duration_s',25.));required=params.get('smoke_takeovers',3)
    if task not in ('mission','smoke') or type(required) is not int or not 1<=required<=5:raise ValueError('Invalid handoff task/takeovers')
    if task=='smoke' and not 12<=duration<=40:raise ValueError('Smoke duration must be 12..40 s')
    force=task=='smoke' and bool(params.get('smoke_takeover',True))
    if context.model_path is None:raise ValueError('Select nominal_model.json in the runner plan')
    s=settings(params.get('overrides'));folder=Path(__file__).with_name('assets')
    definition,_,target,constants,evidence=preflight(context.root,context.model_path,folder)
    reference=Reference(folder,context.model_path);output=Path(context.output_dir)
    (output/'policy_settings.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
    if task=='smoke':context.request_stop_after_task()
    context.event('trajectory_handoff_initialized',version=VERSION,task=task,static_contract=evidence,
        offline_validation=reference.manifest['validation'],offline_solver=reference.manifest['solver'],settings=s,
        execution_policy='continue_with_monitored_schedule_or_explicit_coast',native_score_authoritative=True,
        radar_all_debris_coverage=False,worker_roles=['monitor','planner'])
    workers={};disabled=set();times=[];rows=[];started=launch=up=phase_origin=None
    active=pending=None;generation=0;last_command=np.zeros(6);last_submit=-np.inf;last_monitor_submit=-np.inf
    last_radar=last_advisory=-np.inf;assessment=last_monitor=None;clear_count=0;arrival_since=None;arrival_record=None
    urgent=True;preferred_side=1;expiry_warning=False;monitor_warning=None
    counters=dict(commands=0,worker_results=0,final_worker_results=0,accepted_assessments=0,rejected_results=0,
        takeovers=0,rejoins=0,radar_packets=0,monitor_results=0,guard_takeovers=0,risk_fallbacks=0,
        explicit_coasts=0,planner_failures=0)
    first_command_wall=None
    def retire(state,now,phase,reason):
        nonlocal active,pending,generation,urgent
        active=coast_plan(state,now,phase,s['mission_timeout_s']);pending=None;generation+=1;urgent=True
        counters['explicit_coasts']+=1
        context.event('explicit_coast_selected',reason=reason,sim_s=now,generation=generation,clearance_verified=False,plan=active)
    try:
        while True:
            loop_wall=time.monotonic();prior_command=last_command.copy();context.checkpoint(sample=False)
            packet=context.state();stamp=float(packet['read_end_sim_s']);times.append(stamp);rows.append(packet_row(packet))
            while len(times)>3 and times[1]<stamp-1.:times.pop(0);rows.pop(0)
            state=latent_state(times,rows,definition['parameters']);now=float(context.clock())
            if np.linalg.norm(np.asarray(packet['target_P_m'])-target)>.1:
                context.finish(outcome='failure',reason='public_target_differs_from_reference',stop_sim=True)
            if started is None:
                started=now;launch=np.asarray(packet['position_P_m']);up=launch/np.linalg.norm(launch)
                initial_phase=reference.closest_phase(state,0.,s['initial_phase_window_s']);d=error(state,reference.at(initial_phase)[0])
                if (np.linalg.norm(d[:3])>s['initial_position_tolerance_m'] or np.linalg.norm(d[3:6])>s['initial_velocity_tolerance_m_s'] or
                    np.linalg.norm(d[6:9])>s['initial_attitude_tolerance_rad'] or np.linalg.norm(d[9:])>s['initial_rate_tolerance_rad_s']):
                    context.finish(outcome='failure',reason='initial_state_outside_reference_basin',stop_sim=True,initial_error=d.tolist())
                phase_origin=now-initial_phase
            phase=now-phase_origin;elapsed=now-started;distance=float(np.linalg.norm(np.asarray(packet['position_P_m'])-target))
            common=dict(sim_s=now,elapsed_controller_s=elapsed,distance_m=distance,fuel_kg=packet['fuel_used_kg'],damage_J=packet['damage_J'],official_score_captured=False,**counters)
            if packet['damage_J']>constants['DamageThreshold_J']:
                context.finish(outcome='failure',reason='observed_official_damage_threshold',stop_sim=True,**common)
            if distance<=constants['KeepOutRadius_m']-s['arrival_margin_m']:
                if arrival_since is None:
                    arrival_since=now
                    estimate=50+100*(constants['NominalTime_s']/max(now,1e-6)-2*packet['damage_J']/constants['DamageThreshold_J']-packet['fuel_used_kg']/constants['NominalFuel_kg'])
                    arrival_record=dict(arrival_first_observed_sim_s=now,score_formula_estimate=estimate,score_time_basis='public simulation epoch; scorer offset unavailable')
                if now-arrival_since>=s['arrival_dwell_s']:context.finish(outcome='success',reason='observed_arrival',**common,**arrival_record)
            else:arrival_since=None;arrival_record=None
            if elapsed>=s['mission_timeout_s']:context.finish(outcome='failure',reason='policy_mission_timeout',stop_sim=True,**common)
            if task=='smoke' and elapsed>=duration:
                passed=(counters['commands']>=10 and counters['accepted_assessments']>=2 and counters['monitor_results']>=3 and
                    counters['radar_packets']>=3 and packet['damage_J']==0 and (not force or counters['takeovers']>=required))
                context.finish(outcome='success' if passed else 'failure',reason='handoff_smoke_complete' if passed else 'handoff_smoke_failed',
                    smoke_passed=bool(passed),mission_success_assessed=False,stop_sim=True,**common)
            # Monitor is polled first; its local action can invalidate a pending
            # mission plan. Numerical failures are diagnostic, not simulator stops.
            for role in ('monitor','planner'):
                worker=workers.get(role)
                if worker is None or role in disabled:continue
                problem=None
                if not worker.alive():problem='numeric_worker_exited'
                elif not worker.ready() and time.monotonic()-worker.started>s['worker_startup_timeout_wall_s']:problem='numeric_worker_startup_timeout'
                elif worker.submitted is not None and time.monotonic()-worker.submitted>s['worker_hard_timeout_wall_s']:problem='numeric_worker_hard_timeout'
                if problem:
                    disabled.add(role);(worker.directory/'STOP').touch();urgent=True
                    context.event('numeric_role_unavailable',role=role,reason=problem,sim_s=now)
                    continue
                response=worker.poll()
                if response is None:continue
                response.update(received_sim_s=now,worker_role=role);append(output/('monitor_results.txt' if role=='monitor' else 'solver_results.txt'),response)
                if role=='monitor':
                    counters['monitor_results']+=1
                    if response.get('generation')==generation:last_monitor=response
                    if response.get('urgent_replan'):urgent=True
                else:
                    counters['worker_results']+=1
                    if response.get('is_final',True):counters['final_worker_results']+=1
                usable,reason=response_usable(response,generation,now,state,s)
                if not usable:
                    counters['rejected_results']+=1
                    if role=='planner':counters['planner_failures']+=1
                    context.event('handoff_result_rejected',role=role,**worker_diagnostic(response,reason));continue
                counters['accepted_assessments']+=1
                if role=='monitor':
                    if response['status'] in ('guard_feasible','guard_risk_fallback'):
                        # Preserve the predicted bridge. A pending earlier switch
                        # must activate first; a fresh-generation monitor follows.
                        if pending is not None and pending['activation_sim_s']<response['activation_sim_s']-.02:continue
                        pending=response;clear_count=0
                else:
                    assessment=response
                    if response['status']=='mpc_feasible' and pending is None:pending=response;clear_count=0
                    elif response['status']=='rejoin_clear' and pending is None:
                        clear_count+=1
                        if clear_count>=s['rejoin_clear_assessments']:pending=response
                    elif response['status']!='rejoin_clear':clear_count=0
            if pending is not None and now>=pending['activation_sim_s']:
                predicted=(interpolate_state(np.asarray(pending['plan']['times']),np.asarray(pending['plan']['states']),now)
                    if 'plan' in pending else np.asarray(pending['prefix_states'][-1]))
                delta=error(state,predicted)
                ok=(now-pending['activation_sim_s']<.4 and np.linalg.norm(delta[:3])<=s['plan_position_tolerance_m'] and
                    np.linalg.norm(delta[3:6])<=s['plan_velocity_tolerance_m_s'] and np.linalg.norm(delta[6:9])<=s['plan_attitude_tolerance_rad'] and
                    np.linalg.norm(delta[9:])<=s['plan_rate_tolerance_rad_s'])
                if ok:
                    if 'plan' in pending:
                        active=pending['plan'];phase_origin=pending['activation_sim_s']-active['phase']
                        preferred_side=active.get('preferred_side',preferred_side)
                        if pending.get('worker_role')=='monitor':
                            counters['guard_takeovers']+=1
                            if pending.get('risk_accepted'):counters['risk_fallbacks']+=1
                        else:counters['takeovers']+=1
                    else:active=None;counters['rejoins']+=1;phase_origin=pending['activation_sim_s']-pending['rejoin_phase']
                    generation+=1;urgent=active is not None and active.get('kind')=='local_avoidance'
                    context.event('handoff_activated',mode='nominal' if active is None else active.get('kind','linear_mpc'),sim_s=now,generation=generation,
                        plan=active if active is not None else None,risk_accepted=bool(pending.get('risk_accepted')),source_job_id=pending['job_id'])
                else:
                    urgent=True;context.event('activation_rejected',reason='activation_state_deviated',sim_s=now,generation=generation)
                pending=None
            phase=now-phase_origin
            if active is not None and active.get('kind')!='coast' and now>=plan_end(active):retire(state,now,phase,'executable_horizon_complete')
            # Four-second assessments are now diagnostics only. The independent
            # monitor remains active during a slow/failed mission solve.
            expired=assessment is not None and now>=assessment['valid_until_sim_s']
            if expired and not expiry_warning:
                context.event('planner_assessment_expired_continuing',sim_s=now,generation=generation);expiry_warning=True
            if not expired:expiry_warning=False
            monitor_stale=last_monitor is None or now-last_monitor['source_sim_s']>s['monitor_stale_s']
            unavailable=last_monitor is not None and not last_monitor.get('observation_available',False)
            warning='monitor_stale' if monitor_stale else 'radar_unassessed' if unavailable else None
            if elapsed>s['startup_grace_s'] and warning:
                if warning!=monitor_warning:context.event('monitor_warning',reason=warning,sim_s=now,generation=generation)
                if active is None or active.get('kind')!='coast':retire(state,now,phase,warning)
            monitor_warning=warning
            if now-stamp>s['max_state_age_s']:context.finish(outcome='failure',reason='own_state_too_old',stop_sim=True,**common)
            u=schedule_control(active,state,now,reference,phase,s,check_tracking=True)
            if u is None:
                retire(state,now,phase,'tracking_departure');u=schedule_control(active,state,now,reference,phase,s)
            context.checkpoint(sample=False);send_now=float(context.clock())
            if pending is not None and now<pending['activation_sim_s']<=send_now:continue
            stale=send_readiness(active,now,send_now,s,reference,phase_origin)
            if stale=='stage_changed_before_send':continue
            if stale:context.finish(outcome='failure',reason=stale,stop_sim=True,**common)
            sent=context.command(u[:3],u[3:]);last_command=u.copy();counters['commands']+=1
            if first_command_wall is None:
                first_command_wall=time.monotonic()-init_wall
                context.event('handoff_first_command',policy_work_wall_s=first_command_wall,sim_s=sent['send_end_sim_s'],before_numeric_worker=True)
                # Both children are numeric-only, BLAS=1, and start AFTER FF+FB.
                workers['monitor']=Worker(context.root,context.model_path,output/'monitor_worker',s,'monitor')
                workers['planner']=Worker(context.root,context.model_path,output/'numeric_worker',s,'planner')
            if sent['send_end_sim_s']-now>s['max_send_latency_s']:context.finish(outcome='failure',reason='command_send_deadline_missed',stop_sim=True,**common)
            now=float(context.clock())
            if now-last_radar>=s['radar_poll_period_s']:
                context.radar(500);last_radar=float(context.clock());counters['radar_packets']+=1;now=float(context.clock())
            snapshot=dict(generation=generation,state=state.tolist(),source_sim_s=stamp,phase=stamp-phase_origin,
                active_plan=scheduled_snapshot(active,pending,stamp),last_command=prior_command.tolist(),launch_position_P=launch.tolist(),launch_up_P=up.tolist(),
                fuel_kg=packet['fuel_used_kg'],damage_J=packet['damage_J'],score_constants=constants,preferred_side=preferred_side,
                committed_until_sim_s=active.get('committed_until_sim_s',stamp) if active is not None else stamp)
            monitor=workers.get('monitor')
            if monitor is not None and 'monitor' not in disabled and monitor.ready() and monitor.job_id is None and now-last_monitor_submit>=s['monitor_period_s']:
                advisory=now-last_advisory>=s['update_period_s']
                monitor.submit(dict(snapshot,job_id=uuid.uuid4().hex,solve_wall_s=s['monitor_budget_wall_s'],do_advisory=advisory));last_monitor_submit=now
                if advisory:last_advisory=now
            planner=workers.get('planner');interval=s['emergency_replan_period_s'] if urgent else s['update_period_s']
            if planner is not None and 'planner' not in disabled and planner.ready() and planner.job_id is None and pending is None and now-last_submit>=interval:
                planner.submit(dict(snapshot,job_id=uuid.uuid4().hex,solve_wall_s=s['solve_wall_s'],force_mpc=force and counters['takeovers']<required));last_submit=now;urgent=False
            append(output/'policy_telemetry.txt',dict(**common,mode='nominal' if active is None else active.get('kind','linear_mpc'),phase_s=phase,generation=generation,
                worker_busy=planner.job_id is not None if planner else False,monitor_busy=monitor.job_id is not None if monitor else False,
                assessment_valid_until_sim_s=assessment.get('valid_until_sim_s') if assessment else None,
                monitor_source_sim_s=last_monitor.get('source_sim_s') if last_monitor else None,pending=pending is not None,
                policy_work_wall_s=time.monotonic()-loop_wall))
            context.sleep(max(.005,s['control_period_s']-(time.monotonic()-loop_wall)))
    finally:
        for worker in workers.values():worker.close()
