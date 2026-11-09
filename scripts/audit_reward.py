"""Run the five paired behaviours on all and only existing level-1 cases."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from thruster_landing.agents.pid import PIDConfig, PIDController
from thruster_landing.checkpoints import load_inference
from thruster_landing.envs import EnvConfig
from thruster_landing.evaluation import write_csv
from thruster_landing.experiments import digest, load_manifest, assert_disjoint, verify_reference
from thruster_landing.play import versions
from thruster_landing.reward_audit import BEHAVIOURS, inference_fingerprint, level1_only, rollout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/reward_audit_level1_v1'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();config=EnvConfig.from_json('configs/env.json');verify_reference()
    paths=['configs/curriculum_train_scenarios.json','configs/curriculum_validation_scenarios.json']
    manifests=[level1_only(load_manifest(path,split)) for path,split in zip(paths,['train','validation'])]
    assert_disjoint(*manifests)
    sources={p:digest(p) for p in paths+['configs/env.json','configs/pid.json','src/thruster_landing/envs/rewards.py']}
    policies={}
    for alg in ['dqn','double_dqn']:
        path=f'artifacts/curriculum_beta10_lr1e4_seed42_v1/{alg}_level1/last.policy.pt'
        sources[path]=digest(path);policies[alg],experiment=load_inference(path)
        assert policies[alg].env_steps==50000
        assert experiment['env_config']==json.loads(Path('configs/env.json').read_text())
    fingerprints={alg:inference_fingerprint(agent) for alg,agent in policies.items()}
    pid=PIDController(config,PIDConfig.from_json('configs/pid.json'));episodes=[];all_steps=[]
    for manifest in manifests:
        for i,scenario in enumerate(manifest['scenarios']):
            initial=None
            for behaviour in BEHAVIOURS:
                row,steps=rollout(config,scenario,manifest['split'],behaviour,pid=pid,policy=policies.get(behaviour))
                if initial is None:initial=row['initial_state']
                assert row['initial_state']==initial
                episodes.append(row);all_steps.extend(steps)
            if (i+1)%32==0:print(manifest['split'],i+1,'states, all five behaviours complete',flush=True)
    for alg,agent in policies.items():assert inference_fingerprint(agent)==fingerprints[alg]
    for path,expected in sources.items():assert digest(path)==expected
    write_csv(out/'episodes.csv',episodes);write_csv(out/'steps.csv',all_steps)
    df=pd.DataFrame(episodes);pairs=[]
    for (split,identifier),group in df.groupby(['split','scenario_id'],sort=False):
        pidrow=group[group.behaviour=='pid'].iloc[0]
        for other in ['off','up','dqn','double_dqn']:
            row=group[group.behaviour==other].iloc[0]
            pairs.append(dict(split=split,scenario_id=identifier,seed=row.seed,other=other,pid_success=bool(pidrow.success),other_success=bool(row.success),pid_outcome=pidrow.outcome,other_outcome=row.outcome,
                pid_return=pidrow['return'],other_return=row['return'],delta_return=pidrow['return']-row['return'],
                pid_discounted_return=pidrow.discounted_return,other_discounted_return=row.discounted_return,delta_discounted_return=pidrow.discounted_return-row.discounted_return))
    write_csv(out/'paired_differences.csv',pairs)
    meta=dict(experiment_id='reward_audit_level1_v1',gamma=.99,sources_sha256=sources,versions=versions(),
              scope='All 128 train + 32 validation level-1 cases, five paired behaviours. No other cases loaded for rollouts.',
              up_definition='Action 7 (all thrusters) at every decision, including when gas is empty.',
              pid_as_reference_only=True,training_steps=0,optimizer_steps=0,inference_fingerprints=fingerprints,
              episodes=len(episodes),environment_steps=len(all_steps),wall_seconds=time.perf_counter()-start,
              signal_tags='Non-exclusive predicates; tilt |theta| decreases; braking vy<0 and delta_vy>0 with minimum geometric clearance<0.5m; approach |x| decreases; descent/ascent geometric clearance; landing=success.')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(df.groupby(['split','behaviour']).agg(n=('success','size'),successes=('success','sum'),discounted=('discounted_return','mean'),raw=('return','mean')).to_string(),flush=True)
    print(pd.DataFrame(pairs).groupby('other')[['delta_return','delta_discounted_return']].min().to_string(),flush=True)
if __name__=='__main__':main()
