"""Independent local monitor and bounded pulse/coast search; public radar only."""
from __future__ import annotations
import time
import numpy as np
from .online import forecast,assess,readiness,Deadline,check_time,law
from .reference import plan_end,matrix,coast_control


def monitor(plant,reference,job,frame,s,*,on_current=None):
    start=time.monotonic();deadline=start+job['solve_wall_s'];now=float(job['source_sim_s']);phase=job['phase']
    activation=now+s['monitor_activation_lead_s'];active=job.get('active_plan');state=np.asarray(job['state']);old=np.asarray(job['last_command'])
    result=dict(job_id=job['job_id'],generation=job['generation'],source_sim_s=now,worker_role='monitor',
        activation_sim_s=activation,valid_until_sim_s=now+s['monitor_stale_s'],status='monitor_unassessed',
        observation_available=False,clearance_verified=False,certified_safe=False,replan_requested=True)
    ok,reason=readiness(frame,now,s)
    if not ok:return dict(result,reason=reason,wall_s=time.monotonic()-start)
    launch=np.asarray(job['launch_position_P']);up=np.asarray(job['launch_up_P']);reports=[]
    try:
        pt,px,_,prev=forecast(plant,reference,state,now,activation,phase,active,old,s,deadline=deadline)
        nt,nx,_,_=forecast(plant,reference,px[-1],activation,activation+s['monitor_horizon_s'],phase+activation-now,active,prev,s,deadline=deadline)
        times=np.r_[pt,nt[1:]];states=np.r_[px,nx[1:]];current=assess(plant,frame,times,states,s,launch,up)
        result.update(prefix_times=pt.tolist(),prefix_states=px.tolist(),prefix_assessment=assess(plant,frame,pt,px,s,launch,up),
            current_path_assessment=current,observation_available=True,clearance_verified=current['admissible'])
        future=assess(plant,frame,nt,nx,s,launch,up);result['current_continuation_assessment']=future
        if current.get('collision_assessment')=='unassessed':
            return dict(result,observation_available=False,reason='radar_coverage_incomplete',wall_s=time.monotonic()-start)
        if current['admissible']:
            result.update(status='monitor_clear',replan_requested=False)
            if job.get('do_advisory'):
                at,ax,_,_=forecast(plant,reference,state,now,now+s['advisory_horizon_s'],phase,active,old,s,step=.75,deadline=deadline)
                advisory=assess(plant,frame,at,ax,s,launch,up)
                result.update(advisory_assessment=advisory,replan_requested=not advisory['admissible'],urgent_replan=False)
            return dict(result,wall_s=time.monotonic()-start)
        # The prefix has already been committed. Select future actions without
        # pretending an initially violated envelope can be made clear retroactively.
        result['urgent_replan']=True
        if future['admissible']:
            return dict(result,status='monitor_hazard',continue_current=True,reason='committed_prefix_only_conflict',wall_s=time.monotonic()-start)
        base=law(reference,phase+activation-now,active,px[-1],activation,s)
        desired=np.asarray(base[:3]);scales=plant.scales[:3];v=matrix(px[-1,6:10])@px[-1,3:6]
        forward=reference.manifest['target_P_m']-px[-1,:3];forward/=max(np.linalg.norm(forward),1e-9)
        side=np.cross(forward,up);side/=max(np.linalg.norm(side),1e-9)
        if np.linalg.norm(side)<.5:side=np.array([1.,0.,0.])
        preferred=int(job.get('preferred_side',1) or 1)
        lateral=matrix(px[-1,6:10])@side*preferred
        brake=-v/max(np.linalg.norm(v),1e-9)
        radial=matrix(px[-1,6:10])@up
        directions=[(scales*lateral,preferred),(-scales*lateral,-preferred),(scales*brake,0),
            (scales*radial,0),(-scales*radial,0),(desired,0)]
        for axis in np.eye(3):directions.extend([(scales*axis,0),(-scales*axis,0)])
        best=None;checked=0;hold=px[-1,6:10].copy()
        library=[(np.zeros(3),0,s['pulse_durations_s'][0])]+[(u,sgn,d) for u,sgn in directions for d in s['pulse_durations_s']]
        for pulse,sgn,duration in library[:s['monitor_max_candidates']]:
            check_time(deadline);ts=activation+np.unique(np.r_[0.,np.arange(.5,s['monitor_horizon_s'],.5),duration,s['monitor_horizon_s']])
            xs=[px[-1].copy()];us=[];previous=prev
            for a,b in zip(ts[:-1],ts[1:]):
                check_time(deadline);u=coast_control(reference,phase+a-now,xs[-1],s,hold)
                if a<activation+duration-1e-8:u[:3]=pulse
                u=plant.canonical_request(u);us.append(u);xs.append(plant.transition(xs[-1],previous,u,b-a));previous=u
            xs=np.asarray(xs);check=assess(plant,frame,ts,xs,s,launch,up);checked+=1
            margin=float(check.get('minimum_inflated_clearance_m',-1e12));floor=check.get('launch_plane_margin_m',-1e12)
            fuel=sum(plant.fuel_rate(u)*dt for u,dt in zip(us,np.diff(ts)))
            rank=(bool(check['admissible']),min(margin,floor),-fuel-(s['side_switch_cost'] if sgn and sgn!=preferred else 0.))
            reports.append(dict(pulse_s=float(duration),side=sgn,assessment=check,fuel_kg=fuel))
            gains=np.array([reference.at(phase+t-now)[2].copy() for t in ts[:-1]]);gains[:,:3,:]=0.
            plan=dict(kind='local_avoidance',times=ts.tolist(),states=xs.tolist(),commands=np.asarray(us).tolist(),gains=gains.tolist(),
                phase=float(phase+activation-now),source='guard_pulse_coast',committed_until_sim_s=float(activation+duration),
                preferred_side=sgn or preferred,clearance_verified=bool(check['admissible']))
            if best is None or rank>best[0]:best=(rank,plan,check)
            # A fully assessed first clear pulse is a useful low-latency incumbent.
            if check['admissible']:break
        if best is not None:
            _,plan,check=best
            if not check['admissible'] and check['minimum_inflated_clearance_m']<future['minimum_inflated_clearance_m']+s['guard_preemption_gain_m']:
                return dict(result,status='monitor_hazard',continue_current=True,reason='no_better_sampled_continuation',candidate_reports=reports,wall_s=time.monotonic()-start)
            verified=bool(check['admissible'] and result['prefix_assessment']['admissible'])
            plan['clearance_verified']=verified
            return dict(result,status='guard_feasible' if verified else 'guard_risk_fallback',plan=plan,
                plan_assessment=check,clearance_verified=verified,risk_accepted=not verified,
                selected_candidate=plan['source'],candidates_checked=checked,candidate_reports=reports,
                wall_s=time.monotonic()-start)
        return dict(result,status='monitor_hazard',reason='no_candidate_within_budget',candidate_reports=reports,wall_s=time.monotonic()-start)
    except Deadline:
        if 'best' in locals() and best is not None:
            _,plan,check=best
            if not check['admissible'] and check['minimum_inflated_clearance_m']<future['minimum_inflated_clearance_m']+s['guard_preemption_gain_m']:
                return dict(result,status='monitor_hazard',continue_current=True,reason='no_better_sampled_continuation',candidate_reports=reports,wall_s=time.monotonic()-start)
            verified=bool(check['admissible'] and result['prefix_assessment']['admissible'])
            plan['clearance_verified']=verified
            return dict(result,status='guard_feasible' if verified else 'guard_risk_fallback',plan=plan,
                plan_assessment=check,clearance_verified=verified,risk_accepted=not verified,
                candidates_checked=checked,candidate_reports=reports,budget_exhausted=True,wall_s=time.monotonic()-start)
        return dict(result,reason='monitor_budget_exhausted',wall_s=time.monotonic()-start)
