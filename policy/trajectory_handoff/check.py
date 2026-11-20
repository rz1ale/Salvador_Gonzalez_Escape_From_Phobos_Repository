"""Read-only static contract/provenance checks. Never imports native scripts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import VERSION
from .reference import Reference, matrix, sha256

PINS = {
    'Scoring':'aa59d4bfa07d00bbf187de225fdfd07bd7631da6f0438ffc17fec49a00d2214c',
    'JetSelectionActuator':'3b94557faa254142f4830cc6c2197df92f7cfe9eab06f8f18bd666653f12fcb7',
    'Radar':'dd48398db5ffa29221d3a03902db9250e86df0f8f1eb9b53f94fc061053a8061'}


def settings(overrides=None):
    s = json.loads(Path(__file__).with_name('settings.json').read_text(encoding='utf-8-sig'))
    overrides = overrides or {}
    if set(overrides)-set(s): raise ValueError('Unknown policy setting: '+str(set(overrides)-set(s)))
    s.update(overrides)
    for k,v in s.items():
        if isinstance(v,(float,int)) and (not np.isfinite(v) or v<0): raise ValueError('Invalid setting '+k)
    if s['schema_version']!=1 or s['correction_axes'] not in ('translation','all'): raise ValueError('Unsupported settings')
    if s['obstacle_uncertainty_geometry'] not in ('sphere','ellipsoid'):raise ValueError('Unknown obstacle uncertainty geometry')
    if s['radar_track_expiry_mode'] not in ('drop_unassessed','retain_occupancy'):raise ValueError('Unknown radar expiry mode')
    if s['radar_birth_initialization'] not in ('radial_prior','two_position'):raise ValueError('Unknown radar birth initialization')
    if type(s['certified_allocation_cache']) is not bool:raise ValueError('certified_allocation_cache must be Boolean')
    seeds=np.asarray(s['feasible_start_translation_scales'],float)
    if seeds.ndim!=1 or not 1<=len(seeds)<=5 or not np.isfinite(seeds).all() or np.max(abs(seeds))>1.:
        raise ValueError('Expected 1..5 finite feasibility-seed scales within [-1,1]')
    if not .02<=s['control_period_s']<=.5: raise ValueError('control_period_s outside supported range')
    if not .5<=s['mpc_stage_s']<=2 or not 4<=s['mpc_stages']<=20: raise ValueError('Invalid MPC grid')
    if s['activation_lead_s'] <= s['solve_wall_s']+.1: raise ValueError('Activation lead must exceed solve budget')
    if s['forecast_horizon_s'] < s['activation_lead_s']+s['mpc_stage_s']*s['mpc_stages']:
        raise ValueError('Radar forecast must cover prefix plus MPC horizon')
    if not 2<=s['tail_stages']<=16:raise ValueError('tail_stages must be 2..16')
    if not 2<=s['planner_obstacles']<=100:raise ValueError('Invalid planner obstacle working set')
    if not s['mpc_stages']*s['mpc_stage_s']<s['advisory_horizon_s']<s['forecast_horizon_s']-s['activation_lead_s']:
        raise ValueError('Detailed, advisory and statistical forecast horizons must be ordered')
    if not .2<=s['monitor_period_s']<=1. or not 0<s['monitor_budget_wall_s']<s['monitor_activation_lead_s']-.1:
        raise ValueError('Invalid monitor schedule')
    if not 3<=s['monitor_horizon_s']<=15. or not 3<=s['monitor_max_candidates']<=40:raise ValueError('Invalid local monitor horizon/library')
    for key in ('arrival_time_scales','pulse_durations_s'):
        value=np.asarray(s[key],float)
        if value.ndim!=1 or not 1<=len(value)<=5 or not np.isfinite(value).all() or np.min(value)<=0:raise ValueError('Invalid '+key)
    if max(s['pulse_durations_s'])>=s['monitor_horizon_s']:raise ValueError('Pulse must leave a coast continuation')
    if type(s['affine_planning']) is not bool:raise ValueError('affine_planning must be Boolean')
    for key in ('mpc_stages','tail_stages','monitor_max_candidates','planner_obstacles','qp_iterations','mpc_candidate_count'):
        if type(s[key]) is not int:raise ValueError('Expected integer '+key)
    if min(s['arrival_time_scales'])<.6 or max(s['arrival_time_scales'])>1.5:raise ValueError('Unsupported arrival scales')
    if not 0<s['goal_tolerance_m']<=49.5:raise ValueError('Invalid target tolerance')
    if s['startup_grace_s']>=s['assessment_valid_s']+15: raise ValueError('Unbounded optimistic startup is unsupported')
    return s


def official(root, definition):
    root = Path(root).resolve(); repo = root.parent/'STU4-Sim'
    path = repo/'Sims'/'COMPETITION_SIM.simconf'
    cfg = json.loads(path.read_text(encoding='utf-8-sig')); systems=[]
    def visit(item):
        if isinstance(item,dict):
            if 'Nametag' in item and 'Inst_Parameters' in item: systems.append(item)
            for v in item.values(): visit(v)
        elif isinstance(item,list):
            for v in item: visit(v)
    visit(cfg)
    selected={}; sources={}
    for name,pin in PINS.items():
        matches=[a for a in systems if a.get('Nametag')==name and a.get('Active') is True]
        if len(matches)!=1: raise ValueError('Expected one active official '+name)
        selected[name]=matches[0]
        ref=matches[0]['Source']; prefix='Local/Repos/STU4-Sim/'
        if not ref.startswith(prefix): raise ValueError('Unexpected official source path')
        source=(repo/ref[len(prefix):]).resolve()
        if not source.is_relative_to(repo.resolve()): raise ValueError('Source escapes official repository')
        digest=hashlib.sha256(source.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()
        if digest!=pin: raise ValueError(name+' changed; review the competition contract before rebuilding')
        sources[name]=digest
    c={k:float(selected['Scoring']['Inst_Parameters'][k]) for k in ('NominalTime_s','NominalFuel_kg','DamageThreshold_J','KeepOutRadius_m')}
    entities=cfg['DataManager']['entities']
    # Only own vehicle and the public goal enter the reference. No debris map or seed.
    own=next(e for e in entities if e.get('#Required',{}).get('Name')=='SpaceBuggy')
    goal=next(e for e in entities if e.get('#Required',{}).get('Name')=='AsteroidCycler')
    if not np.allclose(own['Dynamics']['Mass'],definition['mass_kg']) or not np.allclose(
            np.array(own['Dynamics']['Inertia_BodyFrame'][1:]).reshape(3,3),definition['inertia_B_kg_m2']):
        raise ValueError('Official mass/inertia differs from selected model')
    caps=[own['Thrusters']['MaxAccel_m_s2'],own['Thrusters']['MaxAngAccel_rad_s2']]
    if not np.allclose(caps,definition['actuator']['request_norm_caps']): raise ValueError('Request cap mismatch')
    jetref=selected['JetSelectionActuator']['Inst_Parameters']['JetConfigName']; prefix='Local/Repos/STU4-Sim/'
    if not jetref.startswith(prefix): raise ValueError('Unexpected official jet reference')
    jetpath=(repo/jetref[len(prefix):]).resolve()
    if not jetpath.is_relative_to(repo.resolve()): raise ValueError('Jet table escapes official repository')
    # Check the static jet geometry without constructing the LP/backend on the flight thread.
    jet=np.loadtxt(jetpath,delimiter=',',skiprows=1)
    force=-jet[:,6,None]*jet[:,3:6]/np.linalg.norm(jet[:,3:6],axis=1)[:,None]
    A=np.c_[force/definition['mass_kg'],np.linalg.solve(np.array(definition['inertia_B_kg_m2']),np.cross(jet[:,:3],force).T).T]
    if not np.allclose(A,definition['actuator']['jet_matrix'],rtol=1e-9,atol=1e-12): raise ValueError('Embedded allocation matrix mismatch')
    if not np.allclose(jet[:,6],definition['actuator']['jet_force_N'],rtol=1e-9): raise ValueError('Embedded jet force mismatch')
    q=np.array(own['Rotation'][1:]); q/=np.linalg.norm(q)
    p=np.array(own['Location'][1:]); ell=np.asarray(definition['parameters']['reporting_offset_B_m'])
    x0=np.r_[p-matrix(q).T@ell,own['Velocity'][1:],q,own['AngVelocity'][1:]]
    if np.linalg.norm(goal['Velocity'][1:])>1e-9: raise ValueError('This reference builder assumes the supplied stationary public target')
    def count(name):
        value=next(a for a in systems if a.get('Nametag')==name and a.get('Active') is True)['Inst_Parameters']['NumDebris']
        return int(value[1] if isinstance(value,list) else value)
    evidence=dict(configuration_sha256=sha256(path),source_sha256_lf=sources,jet_table_sha256=sha256(jetpath),
                  radar_num_debris=count('Radar'),spawn_num_debris=count('SpawnDebris'),
                  organizer_authenticity_established=False)
    return x0,np.array(goal['Location'][1:],float),c,evidence


def preflight(root, model_path, assets=None):
    if not Path(model_path).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Selected model must be inside this solution clone')
    definition=json.loads(Path(model_path).read_text(encoding='utf-8-sig'))
    if definition.get('model_type')!='public_request_mean_allocation_6dof_v1': raise ValueError('Unsupported model input contract')
    backend=Path(root)/'system_identification'/'model.py'
    expected=definition.get('provenance',{}).get('backend_source_sha256')
    if not expected or sha256(backend)!=expected: raise ValueError('Backend differs from model provenance')
    x0,target,c,evidence=official(root,definition)
    if assets is not None:
        ref=Reference(assets,model_path)
        if ref.manifest['static_contract']!=evidence: raise ValueError('Official static files changed since reference generation')
    return definition,x0,target,c,evidence


def check_entry_loader(root):
    """Exercise the ACTUAL runner loader; stop at run()'s first context access.

    The probe provides no SDK objects and creates no session. This catches entry
    import failures that a normal package import/path/hash check cannot detect.
    """
    from sim_core.runtime import execute_entry
    class RunReached(BaseException):pass
    class Probe:
        def __init__(self):
            self.root=Path(root)
            self.config={'entry':'policy/trajectory_handoff/entry.py'}
            self.assignment={'token':'handoff_preflight_loader'}
        @property
        def parameters(self):raise RunReached()
    try:execute_entry(Probe(),{})
    except RunReached:
        return dict(loader='sim_core.runtime.execute_entry',run_reached=True,native_started=False)
    raise ValueError('Entry-loader probe did not reach run(context)')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',default='nominal_model.json')
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    d,x,target,c,e=preflight(root,root/args.model,Path(__file__).with_name('assets'))
    s=settings();ref=Reference(Path(__file__).with_name('assets'),root/args.model)
    loader=check_entry_loader(root)
    # CLI-only numeric capability check. Flight imports this module without
    # importing SciPy/the obstacle backend on the foreground control path.
    from obstacle_characterization.collision import ellipsoid_separating_normal
    from obstacle_characterization.radar import load_settings
    shared_query=callable(ellipsoid_separating_normal)
    shared_retention=load_settings({'track_expiry_mode':'retain_occupancy'})['track_expiry_mode']=='retain_occupancy'
    shared_birth=load_settings({'birth_initialization':'two_position'})['birth_initialization']=='two_position'
    print(json.dumps(dict(valid=True,policy_version=VERSION,native_started=False,model_type=d['model_type'],score_constants=c,
        trajectory_duration_s=ref.duration,offline_validation=ref.manifest['validation'],static_contract=e,
        offline_solver=ref.manifest['solver'],online_value_provenance=ref.manifest['online_value'],
        entry_loader=loader,
        shared_ellipsoid_query_available=shared_query,
        shared_retained_occupancy_available=shared_retention,
        shared_two_position_initialization_available=shared_birth,
        end_to_end_job_budget=True,
        independent_local_monitor=True,
        assessment_expiry_stops_sim=False,
        target_reaching_coarse_tail=True,
        advisory_forecast_s=s['advisory_horizon_s'],
        implicit_nominal_on_plan_expiry=False,
        certified_allocation_cache=s['certified_allocation_cache'],
        radar_only_all_debris_avoidance_qualified=False),indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
