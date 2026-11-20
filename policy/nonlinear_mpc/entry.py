"""The only live policy entry. Shared runner owns connection, logs and lifecycle."""
from __future__ import annotations
import json
from pathlib import Path
import time
import uuid
import numpy as np
from obstacle_characterization import RadarStream
from policy.nonlinear_mpc import VERSION
from policy.nonlinear_mpc.optimizer import (settings, Predictor, packet_row, reported, matrix,
    command_at, future_schedule, recovery_request, obstacle_arrays, swept_clearances, score_win)
from policy.nonlinear_mpc.check import preflight
from policy.nonlinear_mpc.worker import Worker


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def combined(active,pending,now,last_command):
    if pending is None:return active
    times=[now];commands=[np.asarray(last_command).tolist()]
    if active:
        for t,u in zip(active['command_times_sim_s'],active['commands']):
            if now<t<pending['activation_sim_s']:times.append(t);commands.append(u)
    for t,u in zip(pending['command_times_sim_s'],pending['commands']):
        if t>now:times.append(t);commands.append(u)
    return dict(activation_sim_s=now,expires_sim_s=pending['expires_sim_s'],
        command_times_sim_s=times,commands=commands)


def assess(predictor,x,now,frame,history,plan,s,launch,up,horizon):
    end=now+horizon
    if plan is not None:end=min(end,plan['expires_sim_s']-1e-5)
    if end<=now+.01:return dict(admissible=False,reason='plan_expired')
    if frame['generated_sim_s']>now+1e-7 or end-frame['generated_sim_s']>frame['default_query']['forecast_horizon_s']:
        return dict(admissible=False,reason='radar_forecast_time_unavailable')
    times=np.linspace(now,end,max(2,int(np.ceil((end-now)/s['prediction_step_s']))+1))
    ct,cu=future_schedule(history,plan,now,end)
    xs,_=predictor.rollout(x,now,times,ct,cu)
    pp=np.array([reported(predictor.model,v)[0] for v in xs])
    centers,radii=obstacle_arrays(frame,times,s['vehicle_radius_m'],s['clearance_m'],s['trajectory_chord_padding_m'])
    clearance=swept_clearances(pp,centers,radii)
    floor=float(np.min((pp-launch)@up+s['launch_floor_tolerance_m']))
    rates=float(np.max(np.linalg.norm(xs[:,10:13],axis=1)))
    minimum=float(clearance.min()) if len(clearance) else None
    conditions=frame['coverage']['conditions']
    # Other conditions remain visible; partial/no detections never certify free space.
    severe=any(k in conditions for k in ('unresolved_capacity_loss','current_own_pose_unavailable','future_measurement'))
    ok=(minimum is None or minimum>0) and floor>=-1e-6 and rates<=s['max_rate_rad_s'] and not severe
    return dict(admissible=bool(ok),minimum_inflated_clearance_m=minimum,
        launch_plane_margin_m=floor,maximum_rate_rad_s=rates,coverage_conditions=conditions,
        radar_assessed_objects=len(frame['objects']),certified_safe=False)


def deviation(predictor,x,now,plan,s):
    if not plan or plan.get('kind')!='nmpc' or now<plan['activation_sim_s']:return False
    times=np.asarray(plan['prediction_times_sim_s']);states=np.asarray(plan['predicted_states'])
    ref=np.array([np.interp(now,times,states[:,j]) for j in range(13)])
    q=ref[6:10]/np.linalg.norm(ref[6:10]);actual=x[6:10]/np.linalg.norm(x[6:10])
    angle=2*np.arccos(np.clip(abs(q@actual),0,1))
    p,_=reported(predictor.model,x);rp,_=reported(predictor.model,ref)
    return bool(np.linalg.norm(p-rp)>s['plan_position_tolerance_m'] or
        np.linalg.norm(x[3:6]-ref[3:6])>s['plan_velocity_tolerance_m_s'] or
        np.linalg.norm(x[10:13]-ref[10:13])>s['plan_rate_tolerance_rad_s'] or
        angle>s['plan_attitude_tolerance_rad'])


def choose_recovery(predictor,x,now,target,frame,history,s,launch,up):
    candidates=[recovery_request(predictor,x,target,mode,s['command_bound_scale'])
                for mode in ('brake','coast','progress')]
    # Cheap escape seeds. Every candidate includes angular damping, gravity and
    # shared allocation; full nonlinear trajectory is screened before command.
    from system_identification.model import gravity
    m=predictor.model;B=matrix(x[6:10])
    g=gravity(x[:3],m.p['phobos_mu_prior_m3_s2'],m.p['uniform_external_acceleration_P_m_s2'])
    for axis in range(3):
        for sign in (-1,1):
            desired=sign*.075*np.eye(3)[axis]-g
            angular=-x[10:13]/1.5-np.asarray(m.p['angular_bias_B_rad_s2'])
            raw=m.h*np.r_[np.linalg.solve(m.Kl,B@desired),np.linalg.solve(m.Ka,angular)]
            z=predictor.encode(raw);z*=min(1.,s['command_bound_scale']/max(np.max(abs(z)),1e-12))
            candidates.append(predictor.decode(z))
    for u in candidates:
        plan=dict(kind='recovery',activation_sim_s=now,expires_sim_s=now+s['horizon_s']+s['activation_lead_s'],
            command_times_sim_s=[now],commands=[u.tolist()])
        trial_history=history+[(now,u.tolist())]
        check=assess(predictor,x,now,frame,trial_history,plan,s,launch,up,s['guard_horizon_s'])
        if check['admissible']:return plan,check
    return None,dict(admissible=False,reason='no_admissible_recovery_candidate',certified_safe=False)


def run(context):
    allowed={'task','smoke_duration_s','overrides'}
    if set(context.parameters)-allowed:raise ValueError('Unknown NMPC case parameter')
    task=context.parameters.get('task','mission')
    if task not in ('mission','smoke'):raise ValueError('task must be mission or smoke')
    duration=float(context.parameters.get('smoke_duration_s',12.))
    if task=='smoke' and not 10<=duration<=30:raise ValueError('Policy smoke must last 10..30 simulation seconds')
    s=settings(context.parameters.get('overrides',{}))
    if context.model_path is None:raise ValueError('Select nominal_model.json through the shared run plan')
    model,constants,evidence=preflight(context.root,context.model_path,s)
    predictor=Predictor(model,s['integration_step_s'])
    if task=='smoke':context.request_stop_after_task()
    stream=RadarStream(context)
    output=Path(context.output_dir);log=output/'policy_telemetry.txt'
    (output/'policy_settings.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
    context.event('nmpc_initialized',version=VERSION,task=task,settings=s,score_constants=constants,
        official_static_source_evidence=evidence,model=str(context.model_path),
        fuel_model='source-informed mean-period estimate; not calibrated',
        vehicle_bound='4 m assumed sphere about SDK reported point; enclosure unverified')
    worker=Worker(context.root,context.model_path,output/'numeric_worker')
    active=pending=None;generation=0;history=[];state_t=[];state_y=[]
    last_command=np.zeros(6);last_sent=-np.inf;last_submit=-np.inf
    started=None;launch=up=None;arrival_since=None;arrival_record=None;recovery_since=None
    counters=dict(worker_results=0,accepted_plans=0,activated_plans=0,rejected_results=0,
        recovery_selections=0,nonzero_commands=0,frames=0,nmpc_monitor_steps=0,recovery_monitor_steps=0)
    source_sequences=set();initial_fuel=initial_damage=None
    try:
        while True:
            loop_wall=time.monotonic()
            context.checkpoint(sample=False)
            packet=context.state();stamp=float(packet['read_end_sim_s'])
            state_t.append(stamp);state_y.append(packet_row(packet))
            while len(state_t)>2 and state_t[1]<stamp-1.:state_t.pop(0);state_y.pop(0)
            t_initial,x,_=model.causal_initial(np.asarray(state_t),np.asarray(state_y),stamp)
            frame=stream.poll(own_state=packet);source_sequences.add(frame['sequence'])
            now=float(context.clock())
            if now>t_initial+1e-8:
                ct,cu=future_schedule(history,None,t_initial,now)
                xs,_=predictor.rollout(x,t_initial,[now],ct,cu);x=xs[0]
            if started is None:
                started=now;launch=np.asarray(packet['position_P_m'],float)
                up=launch/np.linalg.norm(launch)
                initial_fuel=packet['fuel_used_kg'];initial_damage=packet['damage_J']
            distance=float(np.linalg.norm(np.asarray(packet['position_P_m'])-np.asarray(packet['target_P_m'])))
            common=dict(sim_s=now,elapsed_controller_s=now-started,distance_m=distance,
                fuel_kg=packet['fuel_used_kg'],damage_J=packet['damage_J'],official_score_captured=False,
                score_formula_estimate=None,
                score_time_basis='simulation epoch; native scorer start offset unavailable',**counters)
            if packet['damage_J']>constants['DamageThreshold_J']:
                context.finish(outcome='failure',reason='observed_official_damage_threshold',stop_sim=True,**common)
            if distance<=constants['KeepOutRadius_m']-s['arrival_margin_m']:
                if arrival_since is None:
                    arrival_since=now
                    arrival_record=dict(arrival_first_observed_sim_s=now,
                        arrival_first_observed_fuel_kg=packet['fuel_used_kg'],
                        score_formula_estimate=score_win(max(now,s['score_time_floor_s']),packet['fuel_used_kg'],packet['damage_J'],constants))
                if now-arrival_since>=s['arrival_dwell_s']:
                    common.update(arrival_record)
                    context.finish(outcome='success',reason='observed_arrival',**common)
            else:arrival_since=None;arrival_record=None
            if task=='smoke' and now-started>=duration:
                passed=counters['activated_plans']>=1 and counters['nonzero_commands']>=1 and len(source_sequences)>=2 and packet['damage_J']<=initial_damage
                context.finish(outcome='success' if passed else 'failure',reason='nmpc_smoke_complete' if passed else 'nmpc_smoke_failed',
                    smoke_passed=bool(passed),mission_success_assessed=False,**common)
            if now-started>=s['mission_timeout_s']:
                context.finish(outcome='failure',reason='policy_mission_timeout',stop_sim=True,**common)

            if not worker.alive():
                context.finish(outcome='failure',reason='numeric_worker_exited',stop_sim=True,**common)
            if not worker.ready() and time.monotonic()-worker.started>30:
                context.finish(outcome='failure',reason='numeric_worker_startup_timeout',stop_sim=True,**common)
            if worker.submitted is not None and time.monotonic()-worker.submitted>s['worker_timeout_wall_s']:
                context.finish(outcome='failure',reason='numeric_solve_hard_timeout',stop_sim=True,**common)
            response=worker.poll()
            if response is not None:
                counters['worker_results']+=1
                append(output/'solver_results.txt',response)
                valid=(response.get('status')=='feasible' and response.get('generation')==generation and
                    now<response.get('activation_sim_s',0)-.05)
                if valid:
                    proposed=combined(active,response,now,last_command)
                    check=assess(predictor,x,now,frame,history,proposed,s,launch,up,
                        min(response['expires_sim_s']-now,frame['generated_sim_s']+9.9-now))
                    valid=check['admissible']
                if valid:
                    response['kind']='nmpc';pending=response;counters['accepted_plans']+=1
                else:
                    counters['rejected_results']+=1
                    context.event('nmpc_result_rejected',job_id=response.get('job_id'),status=response.get('status'),
                        late=now>=response.get('activation_sim_s',0)-.05,generation_current=generation)
            if pending is not None and now>=pending['activation_sim_s']:
                active=pending;pending=None;generation+=1;counters['activated_plans']+=1;recovery_since=None

            scheduled=combined(active,pending,now,last_command)
            check=None
            if active is not None and command_at(active,now) is not None and not deviation(predictor,x,now,active,s):
                check=assess(predictor,x,now,frame,history,scheduled,s,launch,up,s['guard_horizon_s'])
            if check is None or not check['admissible']:
                active,pending=None,None;generation+=1
                active,check=choose_recovery(predictor,x,now,packet['target_P_m'],frame,history,s,launch,up)
                counters['recovery_selections']+=1
                if active is None:
                    context.finish(outcome='failure',reason='no_admissible_recovery',stop_sim=True,guard=check,**common)
            in_mpc=active.get('kind')=='nmpc'
            counters['nmpc_monitor_steps' if in_mpc else 'recovery_monitor_steps']+=1
            if not in_mpc:
                recovery_since=now if recovery_since is None else recovery_since
                if now-recovery_since>s['recovery_timeout_s']:
                    context.finish(outcome='failure',reason='no_usable_nmpc_plan',stop_sim=True,**common)

            u=command_at(active,now)
            if u is None:raise RuntimeError('No live command after policy selection')
            # Periodic refresh uses actual send brackets, not desired activation time.
            if now-last_sent>=s['monitor_period_s'] or np.max(abs(u-last_command))>1e-12:
                sent=context.command(u[:3],u[3:]);last_sent=sent['send_end_sim_s'];last_command=u.copy()
                history.append((float(last_sent),u.tolist()))
                history=history[-100:]
                if np.linalg.norm(u)>1e-12:counters['nonzero_commands']+=1
            if worker.ready() and worker.job_id is None and pending is None and now-last_submit>=s['update_period_s']:
                activation=now+s['activation_lead_s']
                try:ct,cu=future_schedule(history,active,now,activation)
                except ValueError:
                    active=None
                else:
                    job=dict(job_id=uuid.uuid4().hex,generation=generation,state=x.tolist(),source_sim_s=now,
                        activation_sim_s=activation,target=packet['target_P_m'],fuel_kg=packet['fuel_used_kg'],
                        damage_J=packet['damage_J'],score_constants=constants,settings=s,obstacles=frame,
                        committed_times=ct,committed_commands=cu,warm_plan=active if in_mpc else None,
                        launch_position_P=launch.tolist(),launch_up_P=up.tolist())
                    worker.submit(job);last_submit=now
            counters['frames']=len(source_sequences)
            append(log,dict(**{k:v for k,v in common.items() if k not in counters},**counters,
                active_kind=active.get('kind') if active else None,generation=generation,pending=pending is not None,
                worker_busy=worker.job_id is not None,guard=check,controller_work_wall_s=time.monotonic()-loop_wall))
            context.sleep(s['monitor_period_s'])
    finally:
        worker.close()
