"""Rebuild solution-local experiment JSON. Does not connect/start/change a simulation."""
from __future__ import annotations
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SETTINGS='system_identification/campaigns/settings.json'
ENTRY='system_identification/campaigns/identification.py'
DEPENDENCIES=[SETTINGS,'system_identification/__init__.py',
    'system_identification/campaigns/__init__.py',
    'system_identification/campaigns/flight_math.py']

def case(name, family, split='training', **params):
    return {'id':name,'parameters':dict(family=family,split=split,**params)}

def balanced_patterns(index):
    # Six nonconstant Walsh columns: balanced signs and full six-input rank.
    codes=[(index+col)%7+1 for col in range(6)]
    return [[1 if bin(((row+index)%8)&code).count('1')%2==0 else -1
             for code in codes] for row in range(8)]

def config(cases):
    return {'schema_version':1,'mode':'automatic','manual_session':'continue',
        'entry':ENTRY,'parameters':{'settings_file':SETTINGS},'model':None,
        'dependencies':DEPENDENCIES,'output_root':'system_identification/data_sets',
        'schedule':{'cases':cases,'repetitions_per_case':1,
                    'repeat_set':{'mode':'fixed','count':1},'order':'listed'},
        'retry':{'max_retries_per_trial':0,'retry_on':['startup_timeout','backend_disconnect','process_exit'],
                 'on_exhausted':'pause'},'on_task_failure':'pause',
        'limits':{'startup_wall_s':150,'heartbeat_wall_s':45,'clock_stall_wall_s':90,
                  'attempt_wall_s':480,'shutdown_wall_s':30,'task_sim_s':None},
        'logging':{'state_period_s':.2}}

def plans():
    smokes=[case('SMOKE01_LIFT','smoke_lift','smoke',smoke_id='SMOKE01_LIFT'),
            case('SMOKE02_PULSES','smoke_pulses','smoke',smoke_id='SMOKE02_PULSES'),
            case('SMOKE03_COAST_MIXED','smoke_coast_mixed','smoke',smoke_id='SMOKE03_COAST_MIXED')]
    saturation=[]
    for axis,name in enumerate(('TX','TY','TZ','RX','RY','RZ')):
        levels=[.0002,.001,.004,.5] if axis<3 else [.0001,.0005,.003,.05]
        for sign,label in ((1,'P'),(-1,'N')):
            for i,amp in enumerate(levels):
                saturation.append(case(f'SAT_{name}_{label}_A{i+1:02d}','saturation',
                                       axis=axis,sign=sign,amplitude=amp,pulse_s=1. if axis<3 else .2))
    coasts=[case('COAST_H22','coast',height_m=22.,coast_s=5.),
            case('COAST_H28','coast',height_m=28.,coast_s=5.),
            case('COAST_ROLL_P','coast',attitude_offset_rad=[.15,0,0],coast_s=5.),
            case('COAST_PITCH_N','coast',attitude_offset_rad=[0,-.15,0],coast_s=5.)]
    timing=[]
    for axis,name in ((0,'TX'),(2,'TZ'),(3,'RX'),(5,'RZ')):
        timing.append(case('TIMING_'+name,'timing',axis=axis,amplitude=.004 if axis<3 else .003,
                           widths_s=[.3,.8,1.2] if axis<3 else [.1,.15,.2]))
        timing.append(case('TIMING_'+name+'_REVERSE','timing',axis=axis,amplitude=.001 if axis<3 else .0005,
                           widths_s=[.6,.4,.8] if axis<3 else [.2,.1,.3]))
    coupled=[]
    for i in range(4):
        patterns=balanced_patterns(i)
        coupled.append(case(f'COUPLED_{i+1:02d}','coupled',patterns=patterns,segment_s=.5,
                            attitude_offset_rad=[0,0,(i-1.5)*.06]))
    held=[]
    for i in range(8):
        # Different complete command profiles, not random rows from training flights.
        patterns=balanced_patterns(i+3)
        held.append(case(f'HELD_{i+1:02d}','validation','validation' if i<4 else 'testing',
                         patterns=patterns,segment_s=.45 if i<4 else .55,
                         base_request=[.0008,.0011,-.0009,.0004,-.00035,.00045],
                         attitude_offset_rad=[.04*(i%3-1),.05*((i+1)%3-1),0]))
    science=saturation+coasts+timing+coupled+held
    # An initial subset for checking practical runtime/coverage, not a replacement holdout.
    scout=[x for x in saturation if x['id'].endswith(('A01','A04'))]+coasts[:2]+coupled[:2]
    return {'smoke_tests.json':config(smokes),'full_campaign.json':config(science),
            'saturation.json':config(saturation),'coasts.json':config(coasts),
            'timing.json':config(timing),'coupled.json':config(coupled),
            'validation.json':config(held),'scout.json':config(scout)}

def main():
    for name,value in plans().items():
        (HERE/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        print(name+': '+str(len(value['schedule']['cases']))+' trials')

if __name__=='__main__': main()
