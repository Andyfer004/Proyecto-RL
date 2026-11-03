"""Run only the fixed single-seed curriculum pilot, with mandatory joint L1 gate."""
import json
from pathlib import Path
from thruster_landing.agents.dqn import DQNConfig
from thruster_landing.curriculum import stop_after_level1
from thruster_landing.train import train
from thruster_landing.envs import EnvConfig

def main():
    protocol=json.loads(Path('configs/curriculum_protocol.json').read_text())
    output=Path('artifacts/curriculum_beta10_lr1e4_seed42_v1')
    output.mkdir(exist_ok=False)
    kwargs=dict(env_config=EnvConfig(),train_path='configs/curriculum_train_scenarios.json',validation_path='configs/curriculum_validation_scenarios.json',curriculum=protocol)
    runners=[]
    for alg in ['dqn','double_dqn']:
        cfg=DQNConfig.from_json(f'configs/{alg}_curriculum.json')
        _,runner=train(cfg,output/f'{alg}_level1',stop_steps=50000,**kwargs)
        runners.append(runner)
    stop=stop_after_level1(runners)
    gate=dict(stop=stop,steps_per_algorithm=50000,reason=protocol['stop_rule'],algorithms={alg:dict(training_successes=sum(r['success'] for r in run['episodes']),validation_successes=sum(r['success'] for r in run['validation'])) for alg,run in zip(['dqn','double_dqn'],runners)})
    (output/'level1_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('L1 GATE:',json.dumps(gate),flush=True)
    if stop:return
    for alg in ['dqn','double_dqn']:
        cfg=DQNConfig.from_json(f'configs/{alg}_curriculum.json')
        train(cfg,output/f'{alg}_complete',resume=output/f'{alg}_level1/last.resume.pt',**kwargs)
if __name__=='__main__':main()
