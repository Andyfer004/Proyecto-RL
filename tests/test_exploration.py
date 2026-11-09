from dataclasses import replace
import json
from pathlib import Path
import copy
import numpy as np
import pytest
import torch
from thruster_landing.agents.dqn import DQNAgent,DQNConfig
from thruster_landing.agents.exploration import WeightedExplorationAgent,action_probabilities

PROTOCOL=json.loads(Path('configs/exploration_balance_v1_protocol.json').read_text())


def test_exploration_changes_only_exploratory_action_distribution_and_preserves_greedy():
    cfg=replace(DQNConfig(),learning_starts=1000)
    original=DQNAgent(cfg);weighted=WeightedExplorationAgent(cfg,PROTOCOL);obs=np.zeros(9,np.float32)
    for key,value in original.online.state_dict().items():torch.testing.assert_close(weighted.online.state_dict()[key],value,rtol=0,atol=0)
    rng=copy.deepcopy(weighted.rng.bit_generator.state)
    assert weighted.act(obs,explore=False)==original.act(obs,explore=False)
    assert weighted.rng.bit_generator.state==rng and weighted.updates==0 and len(weighted.replay)==0
    counts=np.bincount([weighted.act(obs) for _ in range(12000)],minlength=8)/12000
    np.testing.assert_allclose(counts,PROTOCOL['action_probabilities'],atol=.015,rtol=0)
    assert all(counts>0)
    assert weighted.epsilon==original.epsilon
    weighted.env_steps=original.env_steps=50000
    assert weighted.epsilon==original.epsilon
    # Greedy calls do not consume the exploration RNG even after warmup.
    rng=copy.deepcopy(weighted.rng.bit_generator.state);weighted.act(obs,explore=False)
    assert weighted.rng.bit_generator.state==rng


def test_weighted_rng_resume_and_protocol_geometry():
    weighted=WeightedExplorationAgent(DQNConfig(),PROTOCOL);obs=np.zeros(9,np.float32)
    for _ in range(17):weighted.act(obs)
    state=weighted.state_dict();expected=[weighted.act(obs) for _ in range(20)]
    restored=WeightedExplorationAgent(DQNConfig(),PROTOCOL);restored.load_state_dict(state)
    assert [restored.act(obs) for _ in range(20)]==expected
    p=action_probabilities(PROTOCOL)
    assert p@np.array([0,1800,600,600,1200,2400,2400,3000])==pytest.approx(990)
    assert p@np.array([0,0,-360,360,0,-360,360,0])==pytest.approx(0)
    assert p[[0,1,4,7]].sum()==pytest.approx(.9)
    with pytest.raises(ValueError):action_probabilities({**PROTOCOL,'action_probabilities':[1]+[0]*7})

def test_prepared_experiment_keeps_reward_hyperparameters_and_level1_cases():
    for alg in ['dqn','double_dqn']:
        old=json.loads(Path(f'configs/{alg}_curriculum.json').read_text())
        new=json.loads(Path(f'configs/{alg}_exploration_v1.json').read_text())
        assert {key for key in new if new[key]!=old[key]}=={'total_steps'}
        assert new['total_steps']==100000 and new['epsilon_decay_steps']==80000
        assert new['learning_rate']==.0001 and new['smooth_l1_beta']==10
    for split,count in [('train',128),('validation',32)]:
        full=json.loads(Path(f'configs/curriculum_{split}_scenarios.json').read_text())
        subset=json.loads(Path(f'configs/level1_{split}_scenarios.json').read_text())
        assert subset['scenarios']==[case for case in full['scenarios'] if case['source_level']==1]
        assert len(subset['scenarios'])==count
    assert PROTOCOL['reward_version']=='v1' and PROTOCOL['status']=='prepared_not_run'
