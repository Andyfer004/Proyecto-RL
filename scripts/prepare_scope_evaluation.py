"""Freeze scope methods, reserved cases and independent complementary evaluation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from thruster_landing.experiments import assert_disjoint
from thruster_landing.scope_evaluation import sha256

LOW=dict(x=(-.1,.1),y=(1.6,2.4),vx=(-.03,.03),vy=(-.3,-.05),theta=(-.01,.01),omega=(-.005,.005))
HIGH=dict(x=(-.3,.3),y=(8.,13.),vx=(-.1,.1),vy=(-1.,0.),theta=(-.03,.03),omega=(-.005,.005))


def exposure(state,wind):
    inside=wind==0 and any(all(a<=state[k]<=b for k,(a,b) in ranges.items()) for ranges in [LOW,HIGH])
    return 'within_assisted_initial_support' if inside else 'outside_assisted_initial_support'


def complementary_cases(n=32):
    cases=[]
    for group_index,group in enumerate(['represented_l1','represented_high','new_displacement','new_dynamics','wind_control']):
        for index in range(n):
            seed=440000+group_index*1000+index;rng=np.random.default_rng(seed)
            ranges=LOW if group=='represented_l1' else HIGH
            state={k:float(rng.uniform(*v)) for k,v in ranges.items()}
            if group=='new_displacement':state['x']=float(rng.choice([-1,1])*rng.uniform(.6,1.5))
            if group=='new_dynamics':
                state.update(theta=float(rng.choice([-1,1])*rng.uniform(.06,.12)),
                    vx=float(rng.choice([-1,1])*rng.uniform(.2,.5)),vy=float(rng.uniform(-1.8,-.8)),omega=float(rng.uniform(-.05,.05)))
            winds=[0.,-20.,20.,-40.,40.] if group=='wind_control' else [0.]
            for w_index,wind in enumerate(winds):
                condition=group if wind==0 else 'wind_'+('m' if wind<0 else 'p')+str(int(abs(wind)))
                cases.append(dict(id=f'complement-{condition}-{index:03d}',condition=condition,
                    seed=seed+100000*w_index,suite='complement',exposure=exposure(state,wind),
                    paired_wind_id=f'wind-base-{index:03d}' if group=='wind_control' else '',
                    options=dict(initial_state=state.copy(),wind_force=wind)))
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/scope_alignment_v1'));a=p.parse_args()
    protocol_path=Path('configs/scope_final_v1_protocol.json');complement_path=Path('configs/scope_final_v1_complement.json')
    if protocol_path.exists() or complement_path.exists():raise ValueError('Protocol already frozen; do not overwrite')
    a.root.mkdir(parents=True,exist_ok=True);models=a.root/'frozen_models';models.mkdir(exist_ok=False)
    candidates=[('dqn_assisted42','dqn','artifacts/recovery_v1/dqn_coverage_v2_seed42/best.policy.pt'),
                ('double_assisted42','double_dqn','artifacts/recovery_v1/double_coverage_v2_seed42/best.policy.pt'),
                ('dqn_assisted43','dqn','artifacts/recovery_v1/dqn_coverage_v2_seed43/best.policy.pt')]
    methods=[dict(id='pid',kind='pid',label='PID congelado',selection='configs/pid.json frozen before RL development')]
    audit=[]
    for mid,algorithm,path in candidates:
        payload=torch.load(path,weights_only=True,map_location='cpu');cfg=payload['config'];assert cfg['algorithm']==algorithm
        directory=Path(path).parent;ev=pd.read_csv(directory/'evaluations.csv');online=ev[ev.phase=='online']
        ranked=[]
        for (phase,steps),g in ev.groupby(['phase','online_steps'],sort=False):
            rates=g.groupby('condition').success.mean();succ=g[g.success]
            rank=(rates.min(),-succ.contact_error_x_m.mean() if len(succ) else -float('inf'))
            ranked.append((rank,phase,int(steps)))
        selected=max(ranked,key=lambda x:x[0]);assert selected[2]==payload['env_steps'] and selected[1]==payload['metadata']['phase']
        copies=models/(mid+'.policy.pt');shutil.copyfile(path,copies)
        classification=cfg['pretrain_classification_updates'];offline=cfg['pretrain_td_updates'];online_updates=payload['updates']-classification-offline
        assert online_updates>0
        metrics=pd.read_csv(directory/'metrics.csv');td=metrics[(metrics.phase!='classification')&(metrics.updates<=payload['updates'])]
        assert td.td_loss.notna().all() and td.nstep_loss.notna().all()
        audit.append(dict(method=mid,classification_updates=classification,offline_td_margin_updates=offline,
            online_td_margin_updates=online_updates,total_updates=payload['updates'],checkpoint_online_steps=payload['env_steps'],
            logged_td_loss_mean=float(td.td_loss.mean()),logged_nstep_loss_mean=float(td.nstep_loss.mean()),
            logged_margin_loss_mean=float(td.margin_loss.mean()),actual_algorithm=cfg['algorithm'],
            checkpoint_sha256=sha256(path),evidence=str(directory/'metrics.csv')))
        methods.append(dict(id=mid,kind='assisted',algorithm=algorithm,checkpoint=str(copies),source=path,
            label=('DQN' if algorithm=='dqn' else 'Double DQN')+' asistido por demostraciones PID, semilla '+str(cfg['seed']),
            selection='max min(group dev success), then min mean x successful contact; no check/test selection',
            selected_steps=payload['env_steps'],training_run_steps=30000,training_metadata=str(directory/'metadata.json')))
    for algorithm in ['dqn','double_dqn']:
        directory=Path('artifacts')/(algorithm+'_pilot_100k');path=directory/'best.policy.pt'
        payload=torch.load(path,weights_only=True,map_location='cpu');meta=json.loads((directory/'metadata.json').read_text())
        assert payload['env_steps']==meta['best_validation']['env_steps']
        copies=models/(algorithm+'_scratch.policy.pt');shutil.copyfile(path,copies)
        methods.append(dict(id=algorithm+'_scratch',kind='scratch',algorithm=algorithm,checkpoint=str(copies),source=str(path),
            label=algorithm+' desde cero, piloto uniforme semilla42',selected_steps=payload['env_steps'],training_run_steps=100000,
            selection=meta['best_checkpoint_rule'],training_metadata=str(directory/'metadata.json')))
    complement=dict(schema_version=1,split='test',role='independent_complement',
        origin='Deterministic NumPy default_rng per-case seeds440000+; prescribed without evaluating any policy; no rejection by outcome',
        scenarios=complementary_cases())
    existing=[json.loads(p.read_text()) for p in Path('configs').glob('*scenarios.json')]
    assert_disjoint(complement,*[])  # Exclusion across every previous split, including checks.
    for manifest in existing:assert_disjoint(complement,manifest)
    # Independent clearance check, no rollouts or policy-dependent filtering.
    for case in complement['scenarios']:
        s=case['options']['initial_state'];theta=s['theta']
        clearance=min(s['y']-.6*abs(np.sin(theta))-np.cos(theta),s['y']-.7*abs(np.sin(theta))-.8*np.cos(theta))
        assert clearance>.1 and abs(s['x'])+1<12 and s['y']+1<25 and 'gas' not in s
    complement_path.write_text(json.dumps(complement,indent=2)+'\n')
    (a.root/'methodology_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    fixed=[Path('configs/env.json'),Path('configs/pid.json'),Path('configs/pid_test_scenarios.json'),complement_path]
    fixed += [Path(m[k]) for m in methods for k in ['checkpoint','source','training_metadata'] if k in m]
    fixed += [p for p in Path('src/thruster_landing').rglob('*.py')]
    fixed += [Path('scripts')/s for s in ['prepare_scope_evaluation.py','evaluate_scope.py','report_scope.py','animate_scope.py']]
    fixed += [Path('tests/test_scope_evaluation.py')]
    protocol=dict(version='scope_final_v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        methods=methods,reserved_manifest='configs/pid_test_scenarios.json',complement_manifest=str(complement_path),
        env_config='configs/env.json',pid_config='configs/pid.json',gamma_for_report=.99,
        no_training=True,no_adjustment_after_results=True,previous_PID_test_exposure=True,
        previous_neural_final_test_evaluations=0,secondary_seed_is_sensitivity_not_campaign=True,
        primary_comparison=['pid','dqn_assisted42','double_assisted42','dqn_scratch','double_dqn_scratch'],
        selection_before_test=True,confidence_intervals='Descriptive rates, one main seed; no general algorithm superiority claim',
        metrics='Success and cause; contact-only velocity/angle/omega/precision; gas and duration separated by outcome; same-state comparisons',
        representativeness='Support labels refer to assisted reset distribution, not visited-state coverage. Baseline resets were centered vertical only. PID parameters were designed on varied validation including wind.',
        wind='Constant horizontal center-of-mass force: paired0,+/-20,+/-40N; no gusts or changes to physics',
        viability=dict(max_wind_N=40,hover_tilt_rad=float(np.arctan(40/981)),hover_force_N=float(np.hypot(40,981)),
            maximum_force_N=3000,minimum_high_foot_clearance_m=6.9,maximum_initial_downward_speed_m_s=1.8,
            ideal_max_thrust_braking_distance_m=1.8**2/(2*(3000/100-9.81)),
            note='Authority/clearance bounds only, not proof that every controller or combination will land'),
        artifacts_root=str(a.root),sha256={str(p):sha256(p) for p in fixed})
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    (a.root/'protocol_sha256.json').write_text(json.dumps({str(protocol_path):sha256(protocol_path)},indent=2)+'\n')
    print('Frozen',len(complement['scenarios']),'complementary plus100 reserved cases;',len(methods),'methods')

if __name__=='__main__':main()
