"""Only four fixed 100k runs: uniform controls and weighted exploration variants."""
import argparse
import hashlib
import json
from pathlib import Path
from thruster_landing.agents.dqn import DQNConfig
from thruster_landing.train import train


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('artifacts/exploration_comparison_level1_seed42_100k_v1'));args=parser.parse_args()
    protocol=json.loads(Path('configs/exploration_comparison_execution.json').read_text())
    exploration=json.loads(Path('configs/exploration_balance_v1_protocol.json').read_text())
    assert exploration['action_probabilities']==protocol['weighted_probabilities']
    for alg in ['dqn','double_dqn']:
        config=DQNConfig.from_json(f'configs/{alg}_exploration_v1.json')
        previous=DQNConfig.from_json(f'configs/{alg}_curriculum.json')
        assert {k for k,v in config.to_dict().items() if v!=previous.to_dict()[k]}=={'total_steps'}
        assert config.total_steps==100000 and config.seed==42
    args.output.mkdir(parents=True,exist_ok=False)
    names=['src/thruster_landing/agents/dqn.py','src/thruster_landing/agents/exploration.py','src/thruster_landing/train.py','src/thruster_landing/rl_evaluation.py','src/thruster_landing/training_telemetry.py','configs/exploration_comparison_execution.json','configs/exploration_balance_v1_protocol.json','configs/level1_train_scenarios.json','configs/curriculum_validation_scenarios.json','configs/dqn_exploration_v1.json','configs/double_dqn_exploration_v1.json','configs/env.json']
    (args.output/'execution_manifest.json').write_text(json.dumps(dict(sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}),indent=2)+'\n')
    for alg in ['dqn','double_dqn']:
        for method in ['uniform','weighted_variant']:
            print('START',alg,method,flush=True)
            train(DQNConfig.from_json(f'configs/{alg}_exploration_v1.json'),args.output/f'{alg}_{method}',train_path=protocol['train_manifest'],validation_path=protocol['validation_manifest'],exploration=exploration if method=='weighted_variant' else None,run_protocol=protocol)
if __name__=='__main__':main()
