from dataclasses import replace
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from thruster_landing.agents.dqn import DQNConfig
from thruster_landing.agents.exploration import WeightedExplorationAgent
from thruster_landing.replay_exposure import absolute_transition_steps,ReplayExposure,SAMPLE_DTYPE

PROTOCOL=json.loads(Path('configs/exploration_balance_v1_protocol.json').read_text())


@pytest.mark.parametrize('algorithm',['dqn','double_dqn'])
def test_capacity_preserves_initialization_rng_samples_and_updates_before_wrap(algorithm,tmp_path):
    c=replace(DQNConfig(),algorithm=algorithm,learning_rate=.0001,smooth_l1_beta=10.,epsilon_decay_steps=80000)
    small=WeightedExplorationAgent(replace(c,buffer_capacity=50000),PROTOCOL)
    large=WeightedExplorationAgent(replace(c,buffer_capacity=150000),PROTOCOL)
    for name,value in small.online.state_dict().items():assert torch.equal(value,large.online.state_dict()[name])
    assert small.rng.bit_generator.state==large.rng.bit_generator.state
    assert small.replay.rng.bit_generator.state==large.replay.rng.bit_generator.state
    for i in range(2000):
        obs=np.array([i%7]*9,dtype=np.float32)
        for a in [small,large]:a.replay.add(obs,i%8,float(i%3),obs+1,i%31==0,False)
    runner={};observer=ReplayExposure(large,runner,tmp_path);large.learning_observer=observer
    for t in range(2000,2040,4):
        # Same population; only the passive observer differs in addition to capacity.
        small.env_steps=large.env_steps=2000
        assert small.act(np.zeros(9,np.float32))==large.act(np.zeros(9,np.float32))
        assert small.learn()==large.learn()
        np.testing.assert_array_equal(small.replay.last_sample_indices,large.replay.last_sample_indices)
        assert small.replay.rng.bit_generator.state==large.replay.rng.bit_generator.state
        for name,value in small.online.state_dict().items():assert torch.equal(value,large.online.state_dict()[name])
    observer.checkpoint();observer.close(tmp_path)
    rows=np.fromfile(tmp_path/'learning_samples.bin',dtype=SAMPLE_DTYPE)
    assert len(rows)==10*64 and sum(runner['replay_exposure']['sample_counts'])==len(rows)
    assert np.isfinite(rows['q']).all() and np.isfinite(rows['target']).all()


def test_slot_to_transition_mapping_at_wrap_and_after_many_wraps():
    np.testing.assert_array_equal(absolute_transition_steps(5,[0,1,2,3,4],5),[1,2,3,4,5])
    np.testing.assert_array_equal(absolute_transition_steps(6,[0,1,2,3,4],5),[6,2,3,4,5])
    np.testing.assert_array_equal(absolute_transition_steps(13,[0,1,2,3,4],5),[11,12,13,9,10])


def test_exposure_resume_matches_uninterrupted_sampling_and_q_targets(tmp_path):
    from thruster_landing.train import train
    c=replace(DQNConfig(),total_steps=24,learning_starts=4,batch_size=4,buffer_capacity=32,validation_frequency=100000,checkpoint_frequency=100000,detailed_validation=False)
    kwargs=dict(train_path='configs/level1_train_scenarios.json',validation_path='configs/curriculum_validation_scenarios.json',exploration=PROTOCOL,run_protocol=dict(decision_logging=True,replay_capacity_audit=True))
    full,runner=train(c,tmp_path/'full',**kwargs)
    train(c,tmp_path/'prefix',stop_steps=8,**kwargs)
    resumed,other=train(c,tmp_path/'continued',resume=tmp_path/'prefix/last.resume.pt',**kwargs)
    assert runner['replay_exposure']==other['replay_exposure']
    for key,value in full.online.state_dict().items():assert torch.equal(value,resumed.online.state_dict()[key])
    expected=np.fromfile(tmp_path/'full/learning_samples.bin',dtype=SAMPLE_DTYPE)
    actual=np.concatenate([np.fromfile(tmp_path/p/'learning_samples.bin',dtype=SAMPLE_DTYPE) for p in ['prefix','continued']])
    np.testing.assert_array_equal(expected,actual)
