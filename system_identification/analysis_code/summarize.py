"""Audit one declared ID session. No model fitting/promotion or simulator connection."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sim_core.common import digest, local_path, read_json, write_json
from system_identification.campaigns.flight_math import fit_coast, radar_threat, rotation

AXES=('TX','TY','TZ','RX','RY','RZ')
INPUTS=('u_ax','u_ay','u_az','u_alphax','u_alphay','u_alphaz')

def last_record(path):
    last=None
    if Path(path).is_file():
        with Path(path).open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip(): last=line
    return json.loads(last) if last else None

def inspect_attempt(rows,directory,profile,metadata,quality,summary):
    """Deterministic acquisition checks, not a fitted model or safety certification."""
    findings=[];pulses=[];details=dict(last_phase=quality.get('phases',[{}])[-1].get('name') if quality.get('phases') else None,
        guard_details=quality.get('guard_details'));settings=metadata.get('settings',{})
    def finding(level,code,**values): findings.append(dict(level=level,code=code,**values))
    if not rows:
        if summary['outcome']=='success': finding('error','successful_attempt_missing_telemetry')
        return dict(findings=findings,pulses=pulses,details=details)
    t=numeric(rows,'t_sim_s');span=numeric(rows,'read_end_s')-numeric(rows,'read_begin_s')
    measurement_keys=('h_launch_m','vx_P_m_s','vy_P_m_s','vz_P_m_s','qPB_x','qPB_y','qPB_z','qPB_w',
                      'wx_B_rad_s','wy_B_rad_s','wz_B_rad_s','fuel_kg','damage_J')
    values=np.c_[t,span,*[numeric(rows,k) for k in measurement_keys+INPUTS]]
    if not np.isfinite(values).all():
        finding('error','nonfinite_telemetry')
        return dict(findings=findings,pulses=pulses,details=details)
    if np.any(np.diff(t)<0) or np.any(span<0): finding('error','timestamps_reversed')
    if np.any(np.diff(t)==0): finding('warning','duplicate_state_times',count=int(np.sum(np.diff(t)==0)))
    gap=float(np.max(np.diff(t))) if len(t)>1 else 0.
    details.update(sample_gap_max_s=gap,read_span_max_s=float(span.max()))
    if gap>settings.get('sample_gap_max_s',float('inf')) or span.max()>settings.get('read_span_max_s',float('inf')):
        finding('error' if summary['outcome']=='success' else 'warning','timing_limit_exceeded',**details)
    if summary['outcome']=='success':
        speed=np.linalg.norm(np.c_[*[numeric(rows,k) for k in ('vx_P_m_s','vy_P_m_s','vz_P_m_s')]],axis=1)
        rate=np.linalg.norm(np.c_[*[numeric(rows,k) for k in ('wx_B_rad_s','wy_B_rad_s','wz_B_rad_s')]],axis=1)
        if (numeric(rows,'h_launch_m').max()>settings.get('height_hard_max_m',float('inf'))+1e-6 or
                speed.max()>settings.get('speed_max_m_s',float('inf'))+1e-6 or
                rate.max()>settings.get('rate_max_rad_s',float('inf'))+1e-6):
            finding('error','successful_attempt_exceeds_configured_envelope')
        if np.ptp(numeric(rows,'damage_J'))>1e-6: finding('error','successful_attempt_has_damage_increment')
    baseline=[dict(t=float(r['t_sim_s']),v=np.array([float(r[k]) for k in ('vx_P_m_s','vy_P_m_s','vz_P_m_s')]))
              for r in rows if r['phase']=='baseline_coast']
    gravity=None
    if baseline:
        try:
            gravity,rms=fit_coast(baseline,settings.get('coast_washout_s',.25))
            details.update(coast_gravity_P_m_s2=gravity.tolist(),coast_velocity_rms_m_s=rms)
        except ValueError as exc: finding('error' if summary['outcome']=='success' else 'warning','coast_fit_unavailable',details=str(exc))
    phases=quality.get('phases',[])
    clean={p['phase']:p for p in quality.get('clean_intervals',[])}
    positive_gaps=np.diff(t)[np.diff(t)>0]
    tolerance=max(.02,1.5*float(np.median(positive_gaps))) if len(positive_gaps) else .02
    def sample_at(stamp):
        index=int(np.argmin(abs(t-stamp)))
        if abs(t[index]-stamp)>tolerance: raise ValueError('No timestamped state close to phase boundary')
        row=rows[index]
        return (t[index],np.array([float(row[k]) for k in ('vx_P_m_s','vy_P_m_s','vz_P_m_s')]),
                rotation(np.array([float(row[k]) for k in ('qPB_x','qPB_y','qPB_z','qPB_w')])),
                np.array([float(row[k]) for k in ('wx_B_rad_s','wy_B_rad_s','wz_B_rad_s')]))
    for i,phase in enumerate(phases):
        if phase.get('kind')!='isolated_pulse' or not phase.get('complete'): continue
        coast=next((p for p in phases[i+1:] if p.get('kind')=='coast'),None)
        if not coast or not coast.get('complete'): continue
        # Preserve explicit censoring: both intervals must have been retained clean.
        if any(p['name'] not in clean or clean[p['name']]['start_sim_s']>p['start_sim_s']+1e-6 or
               clean[p['name']]['end_sim_s']<p['end_sim_s']-1e-6 for p in (phase,coast)):
            finding('warning','pulse_or_readout_censored',phase=phase['name']);continue
        pulse_rows=[r for r in rows if r['phase']==phase['name']]
        end_rows=[r for r in rows if r['phase']==phase['name']+'_end']
        if not pulse_rows or not end_rows:
            finding('error','missing_published_pulse_boundaries',phase=phase['name']);continue
        u=np.array([[float(r[k]) for k in INPUTS] for r in pulse_rows])
        axis=int(np.argmax(np.max(abs(u),axis=0)))
        sign=float(np.sign(np.mean(u[:,axis])));amplitude=float(np.mean(abs(u[:,axis])))
        if sign==0 or np.max(abs(np.delete(u,axis,axis=1)))>1e-12 or not np.allclose(u[:,axis],sign*amplitude,atol=1e-12,rtol=1e-9):
            finding('error','pulse_not_isolated_and_constant',phase=phase['name']);continue
        if profile.get('family')=='saturation' and (axis!=profile.get('axis') or sign!=profile.get('sign') or
                not np.isclose(amplitude,profile.get('amplitude',-1),rtol=1e-9,atol=1e-12)):
            finding('error','pulse_differs_from_declared_profile',phase=phase['name']);continue
        first,last=pulse_rows[0],end_rows[0]
        exposure=.5*(float(last['send_begin_s'])+float(last['send_end_s'])-
                     float(first['send_begin_s'])-float(first['send_end_s']))
        if exposure<=0 or abs(exposure-phase.get('observed_request_exposure_s',exposure))>1e-5:
            finding('error','inconsistent_pulse_exposure',phase=phase['name']);continue
        if gravity is None:
            finding('error' if summary['outcome']=='success' else 'warning','pulse_missing_coast_correction',phase=phase['name']);continue
        try:
            ta,va,Ba,wa=sample_at(phase['start_sim_s'])
            tb,vb,Bb,wb=sample_at(coast['end_sim_s'])
            response=np.r_[Ba@(vb-va-gravity*(tb-ta)),wb-wa]/exposure
            if not np.isfinite(response).all(): raise ValueError('Nonfinite interval response')
        except ValueError as exc:
            finding('error','pulse_readout_unavailable',phase=phase['name'],details=str(exc));continue
        signed=float(sign*response[axis])
        same_units=response[:3] if axis<3 else response[3:]
        coupling=float(np.linalg.norm(np.delete(same_units,axis%3))/max(abs(response[axis]),1e-12))
        observation=dict(axis=axis,sign=int(sign),amplitude=amplitude,measured_signed_response=signed,
            duration_observed_s=exposure,response_body=response.tolist(),phase=phase['name'],
            readout_interval_s=float(tb-ta),same_unit_coupling_ratio=coupling,
            exposure_error_s=exposure-phase.get('requested_duration_s',exposure),
            source='recomputed_public_telemetry',censored_flight=summary['outcome']!='success',
            trial_id=summary['trial_id'],attempt_id=summary['attempt_id'])
        pulses.append(observation)
        if signed<=0: finding('warning','nonpositive_commanded_axis_response',phase=phase['name'],response=signed)
    if profile.get('family')=='saturation' and summary['outcome']=='success' and not pulses:
        finding('error','successful_axis_trial_has_no_verified_pulse')
    mixed=np.array([[float(r[k]) for k in INPUTS] for r in rows if r['kind']=='coupled'])
    if len(mixed):
        scale=np.max(abs(mixed),axis=0);normalized=mixed/np.where(scale>0,scale,1)
        rank=int(np.linalg.matrix_rank(normalized))
        details['coupled_input_rank']=rank
        if profile.get('family')=='coupled' and summary['outcome']=='success' and rank<6:
            finding('error','coupled_excitation_not_rank_six',rank=rank)
    if summary['reason']=='radar_clearance_guard':
        packet=last_record(Path(directory)/'radar.txt');state=last_record(Path(directory)/'states.txt')
        if packet and state and settings:
            try:
                now=.5*(state['read_begin_sim_s']+state['read_end_sim_s'])
                issue,margin,count=radar_threat(packet['observations'],now,dict(v=np.array(state['velocity_P_m_s'])),settings)
                reported=quality.get('guard_details',{}).get('clearance_margin_m')
                stamps=[z[2][0] for z in packet['observations'] if z[2][0]>0 and now-z[2][0]<=settings['radar_max_age_s']]
                details['radar_guard']=dict(recomputed_reason=issue,recomputed_margin_m=margin,reported_margin_m=reported,
                    valid_measurements=count,oldest_valid_age_s=max(now-x for x in stamps) if stamps else None,
                    last_reported_closest_range_m=min(z[1][0] for z in packet['observations']) if packet['observations'] else None,
                    note='Conservative forecast margin; neither collision nor false-alarm classification.')
                if reported is not None and (margin is None or abs(margin-reported)>1e-5):
                    finding('warning','radar_replay_differs_from_recorded_guard')
            except (KeyError,ValueError,IndexError) as exc: finding('warning','radar_replay_unavailable',details=str(exc))
    return dict(findings=findings,pulses=pulses,details=details)

def assessment(plan,ledger,summaries,observations,censored,candidates):
    issues=[dict(trial_id=r['trial_id'],attempt_id=r['attempt_id'],**f)
            for r in summaries for f in r.get('findings',[])]
    identities=[(r['trial_id'],r['attempt_id']) for r in summaries]
    if len(set(identities))!=len(identities): issues.append(dict(level='error',code='duplicate_attempt_identity'))
    errors=[f for f in issues if f['level']=='error']
    sign_warnings=[f for f in issues if f['code']=='nonpositive_commanded_axis_response']
    outcomes=ledger.get('trial_results')
    if outcomes is None: outcomes=[dict(trial_id=r['trial_id'],result=dict(task_outcome=r['outcome'])) for r in summaries]
    successful={r['trial_id'] for r in outcomes if r['result'].get('task_outcome')=='success'}
    unresolved=[r['trial_id'] for r in outcomes if r['result'].get('task_outcome')!='success']
    coverage=[]
    for axis in range(6):
        for sign in (-1,1):
            obs=[x for x in observations if x['axis']==axis and x['sign']==sign]
            levels=sorted({x['amplitude'] for x in obs})
            coverage.append(dict(axis=AXES[axis],sign=sign,levels=levels,pulses=len(obs),
                mean_signed_response_by_level={str(level):float(np.mean([x['measured_signed_response'] for x in obs if x['amplitude']==level])) for level in levels},
                same_unit_coupling_max=max((x['same_unit_coupling_ratio'] for x in obs),default=None)))
    schedule=plan.get('config',{}).get('schedule',{})
    expected=None
    if schedule.get('repeat_set',{}).get('mode')=='fixed':
        expected=schedule['repeat_set']['count']*sum(c.get('repetitions',schedule.get('repetitions_per_case',1)) for c in schedule['cases'])
    if ledger['state']=='complete' and expected is not None and len(outcomes)!=expected:
        item=dict(level='error',code='finalized_trial_count_mismatch',expected=expected,observed=len(outcomes))
        issues.append(item);errors.append(item)
    complete=ledger['state']=='complete' and not unresolved and (expected is None or len(successful)==expected)
    if errors or sign_warnings:
        action='REVIEW_DATA_QUALITY';recommendation='Review the flagged attempts, frames and published requests before further collection or fitting.'
    elif not complete:
        action='RESOLVE_UNFINISHED_TRIALS';recommendation='Resume the existing schedule or explicitly review unresolved trials; keep prior attempts.'
    elif not observations:
        action='ACQUIRE_AXIS_DATA';recommendation='Run the existing scout.json acquisition after the native smoke gate passes.'
    elif any(len(r['levels'])<2 for r in coverage):
        action='COLLECT_MISSING_DIRECTIONS';recommendation='Collect missing signed-axis levels using the existing saturation.json cases.'
    elif any(len(r['levels'])<3 for r in coverage):
        action='EXTEND_EXCITATION';recommendation='Run the existing full_campaign.json for intermediate levels, timing, coupled data and sealed validation/test flights.'
    else:
        action='MODEL_FITTING_NEXT';recommendation='Acquisition checks passed. Run analysis_code/fit_model.py --latest --show for model export, causal replay metrics and plots; retain flight-level splits and sealed testing.'
    return dict(schema_version=1,quality_status='FAIL' if errors else 'WARN' if any(f['level']=='warning' for f in issues) else 'PASS',
        collection_complete=complete,next_step=action,recommendation=recommendation,
        attempts=len(summaries),expected_trials=expected,successful_trials=len(successful),failed_attempts=sum(r['outcome']!='success' for r in summaries),
        unresolved_trials=unresolved,verified_successful_axis_pulses=len(observations),
        retained_censored_axis_pulses=len(censored),coverage=coverage,issues=issues,
        note='Acquisition QA only. PASS is not model validation, a certified envelope, or permission to change native simulation settings.')

def report_text(session,review,summaries,candidates):
    lines=['SYSTEM ID ACQUISITION REPORT',f'Session: {session}',
        f"Acquisition QA: {review['quality_status']} | collection complete: {review['collection_complete']}",
        f"Attempts: {review['attempts']} | successful trials: {review['successful_trials']} | unsuccessful attempts retained: {review['failed_attempts']}",
        f"NEXT STEP: {review['next_step']}",review['recommendation'],
        'Model status: not fitted or validated by this tool.','',
        'SIGNED-AXIS COVERAGE (training successes only; coupling compares same-unit outputs)',
        'Response units: TX/TY/TZ m/s^2; RX/RY/RZ rad/s^2 (interval means).',
        'Responses are sign-aligned: positive follows the requested direction. No fitted COM correction is assumed.',
        'axis\tsign\traw_request:mean_signed_response\tverified_pulses\tmax_same_unit_coupling_ratio']
    for row in review['coverage']:
        means='; '.join(f'{float(a):.6g}:{v:.6g}' for a,v in row['mean_signed_response_by_level'].items()) or 'missing'
        lines.append('\t'.join(map(str,[row['axis'],row['sign'],means,row['pulses'],row['same_unit_coupling_max']])))
    span=[r['diagnostics']['read_span_max_s'] for r in summaries if 'read_span_max_s' in r.get('diagnostics',{})]
    gaps=[r['diagnostics']['sample_gap_max_s'] for r in summaries if 'sample_gap_max_s' in r.get('diagnostics',{})]
    lines+=['',f'Retained telemetry maximum read span [s]: {max(span) if span else None}',
            f'Retained telemetry maximum sample gap [s]: {max(gaps) if gaps else None}']
    lines+=['','PLATEAU DIAGNOSTICS (tested range only; no full 6D envelope claim)']
    lines += [f"{AXES[p['axis']]} {p['sign']:+}: {p['status']}" for p in candidates]
    lines+=['',f"Completed pulse/readout intervals retained from censored flights: {review['retained_censored_axis_pulses']}",
            'Those intervals are listed separately and do not relabel any failed flight.','', 'ATTEMPT DIAGNOSTICS']
    for row in summaries:
        if row['outcome']=='success' and not row.get('findings'): continue
        lines.append(f"{row['trial_id']}/{row['attempt_id']}: {row['outcome']} / {row['reason']}")
        lines.append('  '+json.dumps({k:row.get('diagnostics',{}).get(k) for k in ('last_phase','guard_details')},sort_keys=True))
        for f in row.get('findings',[]): lines.append('  '+json.dumps(f,sort_keys=True))
        if row.get('diagnostics',{}).get('radar_guard'):
            lines.append('  '+json.dumps(row['diagnostics']['radar_guard'],sort_keys=True))
    lines+=['','See metrics.txt, assessment.json, audit.json and authority_observations.json for numeric detail.',
            'No upload or chat review is required for routine analysis. No simulator connection, fitting or model promotion occurred.']
    next_plan={'ACQUIRE_AXIS_DATA':'scout.json','COLLECT_MISSING_DIRECTIONS':'saturation.json',
               'EXTEND_EXCITATION':'full_campaign.json'}.get(review['next_step'])
    if next_plan:
        lines+=['','NEXT COMMAND (run manually from solution root after closing SpaceCRAFT; $py/$gui per workflow section 1):',
            "  & $py .\\run_simulation.py auto --config 'system_identification/campaigns/"+next_plan+
            "' --retry-schedule 3 5 --executable $gui --first-role --analysis 'system_identification/analysis_code/summarize.py'",
            'This analysis does not launch that command. First-role selection requires the intended spectator to be first.']
    return '\n'.join(lines)+'\n'

def telemetry(path):
    with Path(path).open(encoding='utf-8') as f:
        return list(csv.DictReader(f,delimiter='\t'))

def numeric(rows,key):
    return np.array([float(x[key]) for x in rows],float)

def plot_attempt(rows,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=numeric(rows,'t_sim_s');t=t-t[0]
    fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
    axes[0].plot(t,numeric(rows,'h_launch_m'));axes[0].set_ylabel('Launch-plane h [m]')
    for key,label in zip(('vx_P_m_s','vy_P_m_s','vz_P_m_s'),('vx','vy','vz')):
        axes[1].plot(t,numeric(rows,key),label=label)
    axes[1].set_ylabel('Reported v [m/s]');axes[1].legend(ncol=3)
    for key,label in zip(('wx_B_rad_s','wy_B_rad_s','wz_B_rad_s'),('wx','wy','wz')):
        axes[2].plot(t,numeric(rows,key),label=label)
    axes[2].set_ylabel('Body rate [rad/s]');axes[2].legend(ncol=3)
    for key in ('u_ax','u_ay','u_az','u_alphax','u_alphay','u_alphaz'):
        axes[3].step(t,numeric(rows,key),where='post',label=key)
    axes[3].set_ylabel('Public raw requests');axes[3].legend(ncol=3,fontsize=8)
    axes[3].set_xlabel('Simulation seconds from first recorded sample')
    for ax in axes: ax.grid(alpha=.3)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)

def audit(root,session,output,plots='none',include_heldout=False):
    root,session,output=Path(root),Path(session),Path(output)
    plan=read_json(session/'plan.json');ledger=read_json(session/'ledger.json')
    output.mkdir(parents=True,exist_ok=False)
    rows_out=[];observations=[];censored=[];files={}
    for record in ledger['attempts']:
        directory=local_path(session,record['directory'])
        metadata=read_json(directory/'metadata.json') if (directory/'metadata.json').is_file() else {}
        profile=metadata.get('profile',record.get('parameters',{}))
        quality=read_json(directory/'quality.json') if (directory/'quality.json').is_file() else {}
        value=record.get('result') or next((read_json(directory/n) for n in ('supervisor_result.json','result.json') if (directory/n).is_file()),{})
        summary={'trial_id':record['trial_id'],'attempt_id':record['attempt_id'],'split':profile.get('split','unknown'),
            'family':profile.get('family','unknown'),'outcome':value.get('task_outcome','uncommitted'),
            'reason':value.get('reason','unknown'),'clean_intervals':len(quality.get('clean_intervals',[]))}
        path=directory/'telemetry.txt'
        for f in directory.iterdir():
            if f.is_file() and f.suffix in ('.txt','.json'): files[f.relative_to(session).as_posix()]=digest(f)
        held=summary['split'] in ('validation','testing')
        samples=[]
        if path.is_file() and (not held or include_heldout):
            samples=telemetry(path)
        summary['heldout_not_inspected']=held and not include_heldout
        inspection=dict(findings=[],pulses=[],details={})
        if not summary['heldout_not_inspected']:
            try: inspection=inspect_attempt(samples,directory,profile,metadata,quality,summary)
            except (KeyError,ValueError,TypeError,IndexError) as exc:
                inspection=dict(findings=[dict(level='error',code='analysis_input_error',details=str(exc))],pulses=[],details={})
        summary.update(findings=inspection['findings'],diagnostics=inspection['details'])
        valid=not any(f['code'] in ('nonfinite_telemetry','analysis_input_error') for f in inspection['findings'])
        if samples and valid:
            t=numeric(samples,'t_sim_s');h=numeric(samples,'h_launch_m')
            speed=np.linalg.norm(np.c_[*[numeric(samples,k) for k in ('vx_P_m_s','vy_P_m_s','vz_P_m_s')]],axis=1)
            rates=np.linalg.norm(np.c_[*[numeric(samples,k) for k in ('wx_B_rad_s','wy_B_rad_s','wz_B_rad_s')]],axis=1)
            summary.update(samples=len(samples),duration_sim_s=float(t[-1]-t[0]),height_min_m=float(h.min()),height_max_m=float(h.max()),
                speed_max_m_s=float(speed.max()),rate_max_rad_s=float(rates.max()),
                read_span_p95_s=float(np.percentile(numeric(samples,'read_end_s')-numeric(samples,'read_begin_s'),95)),
                sample_interval_median_s=float(np.median(np.diff(t))) if len(t)>1 else None,
                fuel_kg=float(numeric(samples,'fuel_kg')[-1]-numeric(samples,'fuel_kg')[0]),
                damage_increment_J=float(numeric(samples,'damage_J').max()-numeric(samples,'damage_J')[0]))
            if plots=='all' or (plots=='smoke' and (summary['split']=='smoke' or summary['outcome']=='failure')):
                filename=re.sub(r'[^A-Za-z0-9_-]','_',record['trial_id']+'_'+record['attempt_id'])+'.png'
                plot_attempt(samples,output/filename,record['trial_id']+' / '+summary['reason'])
        rows_out.append(summary)
        if summary['split']=='training' and summary['family']=='saturation':
            target=observations if summary['outcome']=='success' else censored
            target.extend(inspection['pulses'])
    candidates=[]
    for axis in range(6):
        for sign in (-1,1):
            found=[x for x in observations if x['axis']==axis and x['sign']==sign]
            levels=sorted({x['amplitude'] for x in found})
            label='insufficient_coverage';detail={}
            if len(levels)>=2:
                top=levels[-2:]
                values=[np.array([x['measured_signed_response'] for x in found if x['amplitude']==a]) for a in top]
                means=[float(x.mean()) for x in values]
                difference=abs(means[1]-means[0])/max(abs(means[1]),abs(means[0]),1e-9)
                flat=difference<=.1 and top[1]/top[0]>=2 and min(means)>0
                label=('candidate_plateau_needs_repeat' if min(map(len,values))<2 else 'plateau_observed_in_tested_range') if flat else 'no_plateau_established'
                detail=dict(top_amplitudes=top,mean_signed_response=means,relative_difference=difference,
                            trials_per_level=[len(x) for x in values])
            candidates.append(dict(axis=axis,sign=sign,status=label,**detail))
    write_json(output/'audit.json',dict(schema_version=1,session=str(session),session_state=ledger['state'],
        attempts=rows_out,source_files_sha256=files,analysis_source_sha256=digest(__file__),
        heldout_numerics_inspected=include_heldout,
        note='Acquisition audit and empirical pulse responses; no fitted nominal model or native score.'))
    write_json(output/'authority_observations.json',dict(observations=observations,plateau_candidates=candidates,
        censored_observations=censored,
        note='Training-only motion-inferred interval estimates after local coast correction. Repetition and timing checks remain necessary.'))
    review=assessment(plan,ledger,rows_out,observations,censored,candidates)
    review['analysis_source_sha256']=digest(__file__)
    review['analysis_dependencies_sha256']={'flight_math.py':digest(Path(__file__).resolve().parents[1]/'campaigns/flight_math.py')}
    write_json(output/'assessment.json',review)
    (output/'report.txt').write_text(report_text(session,review,rows_out,candidates),encoding='utf-8')
    columns=['trial_id','attempt_id','split','family','outcome','reason','samples','duration_sim_s','height_min_m','height_max_m',
             'speed_max_m_s','rate_max_rad_s','fuel_kg','damage_increment_J','read_span_p95_s','sample_interval_median_s','clean_intervals','heldout_not_inspected']
    with (output/'metrics.txt').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,delimiter='\t',extrasaction='ignore');writer.writeheader();writer.writerows(rows_out)
    return output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    target=p.add_mutually_exclusive_group(required=True)
    target.add_argument('--session',help='Explicit existing session path relative to STU4-Solution')
    target.add_argument('--latest',action='store_true',help='Use the existing active-session pointer; no directory search/newest-file guessing')
    p.add_argument('--out',help='New output path relative to STU4-Solution')
    p.add_argument('--plots',choices=('none','smoke','all'),default='none')
    p.add_argument('--include-heldout',action='store_true',help='Explicitly inspect sealed validation/testing numerics')
    p.add_argument('--show',action='store_true',help='Print the readable local assessment as well as its output directory')
    args=p.parse_args(argv)
    selected=read_json(ROOT/'sim_core/state/active_session.json')['session'] if args.latest else args.session.replace('\\','/')
    session=local_path(ROOT,selected)
    config=read_json(session/'plan.json')['config']
    if config.get('entry')!='system_identification/campaigns/identification.py':
        raise ValueError('Selected session is not a system-identification campaign; supply --session explicitly for the intended ID session')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output=local_path(ROOT,args.out.replace('\\','/')) if args.out else ROOT/'system_identification/analysis_results'/session.name/('audit_'+stamp)
    directory=audit(ROOT,session,output,args.plots,args.include_heldout)
    if args.show: print((directory/'report.txt').read_text(encoding='utf-8'))
    print(directory)
    return 0

if __name__=='__main__': raise SystemExit(main())
