"""Offline configuration/source contract check; never starts a simulation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from sim_core.common import local_path, validate, read_json
from .optimizer import Model, settings
from system_identification.model import load_jet_matrix


def source_hash(path):
    text=Path(path).read_text(encoding='utf-8-sig').replace('\r\n','\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def official_contract(root,s,model=None):
    root=Path(root).resolve()
    sim=root.parent/'STU4-Sim'/'Sims'/'COMPETITION_SIM.simconf'
    if not sim.is_file():raise ValueError('Missing unchanged sibling official COMPETITION_SIM.simconf')
    config=read_json(sim);systems=[]
    def visit(x):
        if isinstance(x,dict):
            if 'Nametag' in x and 'Inst_Parameters' in x:systems.append(x)
            for y in x.values():visit(y)
        elif isinstance(x,list):
            for y in x:visit(y)
    visit(config)
    scored=[x for x in systems if x.get('Nametag')=='Scoring' and x.get('Active') is True]
    if len(scored)!=1:raise ValueError('Expected one active official Scoring system')
    values={k:float(scored[0]['Inst_Parameters'][k]) for k in
        ('NominalTime_s','NominalFuel_kg','DamageThreshold_J','KeepOutRadius_m')}
    if any(v<=0 for v in values.values()):raise ValueError('Invalid official scoring constants')
    receipts={}
    for name in ('Scoring','JetSelectionActuator'):
        matches=[x for x in systems if x.get('Nametag')==name and x.get('Active') is True]
        if len(matches)!=1:raise ValueError('Missing/ambiguous official '+name)
        source=matches[0]['Source'];prefix='Local/Repos/STU4-Sim/'
        if not source.startswith(prefix):raise ValueError('Unexpected official source reference')
        path=(root.parent/'STU4-Sim'/source[len(prefix):]).resolve()
        if not path.is_relative_to((root.parent/'STU4-Sim').resolve()):raise ValueError('Official source escapes simulator repository')
        actual=source_hash(path)
        if actual!=s['official_source_sha256_lf'][name]:
            raise ValueError(name+' differs from reviewed supplied source; review objective/allocation before updating the policy pin')
        receipts[name]=dict(path=str(path),sha256_lf=actual)
    if s['arrival_margin_m']>=values['KeepOutRadius_m']:raise ValueError('Arrival margin exceeds mission radius')
    evidence=dict(configuration_path=str(sim),configuration_sha256=hashlib.sha256(sim.read_bytes()).hexdigest(),sources=receipts)
    if model is not None:
        import numpy as np
        bodies=[]
        def body_visit(x):
            if isinstance(x,dict):
                if x.get('#Required',{}).get('Name')=='SpaceBuggy':bodies.append(x)
                for v in x.values():body_visit(v)
            elif isinstance(x,list):
                for v in x:body_visit(v)
        body_visit(config)
        if len(bodies)!=1:raise ValueError('Expected one official own SpaceBuggy definition')
        body=bodies[0];mass=float(body['Dynamics']['Mass'])
        inertia=np.asarray(body['Dynamics']['Inertia_BodyFrame'][1:],float).reshape(3,3)
        caps=[body['Thrusters']['MaxAccel_m_s2'],body['Thrusters']['MaxAngAccel_rad_s2']]
        if not np.isclose(mass,model.definition['mass_kg'],rtol=1e-10) or not np.allclose(inertia,model.J,rtol=1e-10):
            raise ValueError('Official own mass/inertia differs from selected model; update/revalidate model')
        if not np.allclose(caps,model.allocator.caps,rtol=1e-10):raise ValueError('Official request caps differ from model')
        act=next(x for x in systems if x.get('Nametag')=='JetSelectionActuator' and x.get('Active') is True)
        reference=act['Inst_Parameters']['JetConfigName'];prefix='Local/Repos/STU4-Sim/'
        if not reference.startswith(prefix):raise ValueError('Unexpected official jet-table reference')
        table=(root.parent/'STU4-Sim'/reference[len(prefix):]).resolve()
        if not table.is_relative_to((root.parent/'STU4-Sim').resolve()):raise ValueError('Jet-table path escapes official repository')
        A,forces=load_jet_matrix(table,mass,inertia)
        if A.shape!=model.allocator.A.shape or not np.allclose(A,model.allocator.A,rtol=1e-9,atol=1e-12) or not np.allclose(forces,model.allocator.forces,rtol=1e-9):
            raise ValueError('Current official jets differ from embedded model; update/revalidate model')
        evidence['plant_contract']=dict(mass_kg=mass,request_caps=caps,jet_table_path=str(table),
            jet_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),embedded_jet_matrix_matches=True)
    return values,evidence


def preflight(root,model_path,s):
    model=Model.load(model_path)
    expected=model.definition.get('provenance',{}).get('backend_source_sha256')
    backend=Path(root)/'system_identification'/'model.py'
    if expected and hashlib.sha256(backend.read_bytes()).hexdigest()!=expected:
        raise ValueError('Canonical backend differs from selected model provenance; review compatibility')
    if np_missing_forces(model):raise ValueError('Selected model must embed positive jet forces and mass for fuel estimation')
    c,evidence=official_contract(root,s,model)
    return model,c,evidence


def np_missing_forces(model):
    import numpy as np
    return (not np.isfinite(model.definition.get('mass_kg',0)) or model.definition.get('mass_kg',0)<=0 or
        np.any(model.allocator.forces<=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='policy/nonlinear_mpc/run_plan.json')
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    c=validate(root,read_json(local_path(root,args.config,must_exist=True)))
    if c.get('model') is None:raise ValueError('NMPC requires the selected root model JSON')
    results=[]
    for case in c['schedule']['cases']:
        parameters={**c.get('parameters',{}),**case.get('parameters',{})}
        s=settings(parameters.get('overrides',{}));model,score,evidence=preflight(root,local_path(root,c['model']),s)
        results.append(dict(case=case['id'],model_type=model.definition['model_type'],score_constants=score,
            formulation=s['formulation'],horizon_s=s['horizon_s'],source_evidence=evidence))
    print(json.dumps(dict(valid=True,cases=results,native_started=False,organizer_authenticity_established=False,
        vehicle_geometry='4 m assumed sphere about SDK reported origin; enclosing hull NOT verified',
        fuel_model='source-informed effective-period estimate; calibration pending'),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
