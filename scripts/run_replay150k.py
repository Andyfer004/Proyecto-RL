"""Two fresh100k runs, exact-prefix audit at50k, only capacity changed to150k."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from thruster_landing.agents.dqn import DQNConfig
from thruster_landing.checkpoints import read_checkpoint
from thruster_landing.train import train

BASELINE=Path('artifacts/exploration_comparison_level1_seed42_100k_v1')


def verify_prefix(algorithm, output):
    reference=BASELINE/f'{algorithm}_weighted_variant'
    old=read_checkpoint(reference/'step_00050000.resume.pt','resume')
    new=read_checkpoint(output/'last.resume.pt','resume')
    for network in ['online','target']:
        for key,value in old['agent'][network].items():assert torch.equal(value,new['agent'][network][key]),(algorithm,network,key)
    for key in old['agent']['replay']['arrays']:
        assert torch.equal(old['agent']['replay']['arrays'][key],new['agent']['replay']['arrays'][key]),key
    assert old['agent']['rng']==new['agent']['rng']
    assert old['agent']['replay']['rng']==new['agent']['replay']['rng']
    # action_space RNG is allocated from OS entropy by legacy checkpoint snapshots,
    # and is never used by either policy. All dynamics RNGs must match exactly.
    for key in old['environment']:
        if key!='action_rng':assert old['environment'][key]==new['environment'][key],key
    assert old['global_rng']['python']==new['global_rng']['python']
    assert old['global_rng']['numpy']==new['global_rng']['numpy']
    assert torch.equal(old['global_rng']['torch'],new['global_rng']['torch'])
    for counter in ['env_steps','updates','target_syncs']:assert old['agent'][counter]==new['agent'][counter]
    # Compare complete numerical decisions and physical trajectories, not just success counts.
    previous=pd.read_csv(reference/'decisions.csv').query('env_steps<=50000').reset_index(drop=True)
    current=pd.read_csv(output/'decisions.csv')
    pd.testing.assert_frame_equal(previous,current,check_exact=True)
    from thruster_landing.replay_exposure import SAMPLE_DTYPE
    sample_rows=np.fromfile(output/'learning_samples.bin',dtype=SAMPLE_DTYPE)
    sampled=sample_rows['transition_step'].reshape(-1,old['agent']['config']['batch_size'])
    rng=np.random.default_rng(old['agent']['config']['seed']+1)
    for i,t in enumerate(range(1000,50001,4)):
        expected_ids=rng.choice(t,64,replace=False)+1
        np.testing.assert_array_equal(sampled[i],expected_ids)
    assert len(sampled)==12251
    e=pd.read_csv(output/'episodes.csv');steps=e.loc[e.success,'env_steps'].tolist()
    expected=[2324,4257,23468] if algorithm=='dqn' else []
    assert steps==expected,(algorithm,steps,expected)
    result=dict(algorithm=algorithm,step=50000,networks_exact=True,replay_population_exact=True,policy_rng_exact=True,sampling_rng_exact=True,all_sampled_ids_exact=True,dynamics_environment_exact=True,global_rng_exact=True,unused_action_space_rng_equal=old['environment']['action_rng']==new['environment']['action_rng'],all_decisions_exact=True,success_steps=steps,reference_sha256=hashlib.sha256((reference/'step_00050000.resume.pt').read_bytes()).hexdigest())
    (output/'prefix_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PREFIX VERIFIED',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('artifacts/replay150k_weighted_seed42_100k_v1'));parser.add_argument('--continue-existing-prefix',action='store_true',help='Reuse only completed prefixes, verify before resuming; never rerun those steps');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.continue_existing_prefix)
    protocol=json.loads(Path('configs/replay150k_execution.json').read_text());exploration=json.loads(Path('configs/exploration_balance_v1_protocol.json').read_text())
    sources=['src/thruster_landing/agents/replay.py','src/thruster_landing/agents/dqn.py','src/thruster_landing/agents/exploration.py','src/thruster_landing/train.py','src/thruster_landing/replay_exposure.py','src/thruster_landing/training_telemetry.py','configs/replay150k_execution.json','configs/dqn_replay150k.json','configs/double_dqn_replay150k.json','configs/level1_train_scenarios.json','configs/curriculum_validation_scenarios.json','configs/exploration_balance_v1_protocol.json','configs/env.json']
    execution=dict(sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    manifest=args.output/'execution_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==execution
    else:manifest.write_text(json.dumps(execution,indent=2)+'\n')
    for alg in ['dqn','double_dqn']:
        config=DQNConfig.from_json(f'configs/{alg}_replay150k.json');parent=DQNConfig.from_json(f'configs/{alg}_exploration_v1.json')
        assert {k for k,v in config.to_dict().items() if v!=parent.to_dict()[k]}=={'buffer_capacity'}
        assert config.buffer_capacity==150000 and config.total_steps==100000
        kwargs=dict(train_path=protocol['train_manifest'],validation_path=protocol['validation_manifest'],exploration=exploration,run_protocol=protocol)
        prefix=args.output/f'{alg}_prefix50k';continued=args.output/f'{alg}_continued100k'
        if not (args.continue_existing_prefix and (prefix/'last.resume.pt').exists()):
            train(config,prefix,stop_steps=50000,**kwargs)
        verify_prefix(alg,prefix)  # Abort before attributing a difference to capacity if prefix is not exact.
        if not (args.continue_existing_prefix and (continued/'metadata.json').exists()):
            train(config,continued,resume=prefix/'last.resume.pt',**kwargs)
if __name__=='__main__':main()
