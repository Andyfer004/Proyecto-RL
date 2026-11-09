import runpy
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from thruster_landing.agents.dqn import DQNAgent,DQNConfig
from thruster_landing.agents.exploration import WeightedExplorationAgent
from thruster_landing.train import train

PROTOCOL=json.loads(Path('configs/exploration_balance_v1_protocol.json').read_text())


def test_mode_instrumentation_preserves_original_uniform_actions_and_rng():
    namespace=runpy.run_path(str(Path(__file__).parent/'fixtures/act_uniform_before_telemetry.py'))
    config=replace(DQNConfig(),epsilon_decay_steps=80000)
    observed=DQNAgent(config);original=DQNAgent(config);original.act=MethodType(namespace['act'],original)
    observation=np.zeros(9,np.float32)
    for step in [0,999,1000,10000,50000,81000,100000]:
        observed.env_steps=original.env_steps=step
        for _ in range(20):
            assert observed.act(observation)==original.act(observation)
            assert observed.rng.bit_generator.state==original.rng.bit_generator.state
    # A random draw equal to argmax must still be classified as exploration.
    observed=DQNAgent(config)
    with torch.no_grad():
        for p in observed.online.parameters():p.zero_()
    assert observed.act(observation)==0 and observed.last_selection_mode=='exploration'
    before=copy.deepcopy(observed.rng.bit_generator.state)
    assert observed.act(observation,explore=False)==0 and observed.last_selection_mode=='evaluation'
    assert observed.rng.bit_generator.state==before


def test_weighted_greedy_mode_and_diagnostic_state_do_not_change_policy():
    agent=WeightedExplorationAgent(DQNConfig(),PROTOCOL);obs=np.zeros(9,np.float32)
    for _ in range(20):agent.act(obs);assert agent.last_selection_mode=='exploration'
    agent.env_steps=100000;agent.last_selection_mode='arbitrary_diagnostic_value'
    state=copy.deepcopy(agent.rng.bit_generator.state);agent.act(obs,explore=False)
    assert agent.last_selection_mode=='evaluation' and agent.rng.bit_generator.state==state


def test_telemetry_resume_preserves_live_episode_weights_replay_and_mode_counts(tmp_path):
    config=replace(DQNConfig(),total_steps=24,learning_starts=4,batch_size=4,buffer_capacity=32,validation_frequency=100000,checkpoint_frequency=100000,detailed_validation=False)
    kwargs=dict(train_path='configs/level1_train_scenarios.json',validation_path='configs/curriculum_validation_scenarios.json',exploration=PROTOCOL,run_protocol=dict(decision_logging=True,validation_selection_level=1))
    full,full_runner=train(config,tmp_path/'full',**kwargs)
    train(config,tmp_path/'first',stop_steps=8,**kwargs)
    resumed,resume_runner=train(config,tmp_path/'resume',resume=tmp_path/'first/last.resume.pt',**kwargs)
    for name,value in full.online.state_dict().items():assert torch.equal(value,resumed.online.state_dict()[name])
    for name,value in full.replay.arrays.items():np.testing.assert_array_equal(value,resumed.replay.arrays[name])
    assert full_runner['decision_telemetry']==resume_runner['decision_telemetry']
    assert sum(sum(v) for v in resume_runner['decision_telemetry']['histograms'].values())==24
    assert full.rng.bit_generator.state==resumed.rng.bit_generator.state
