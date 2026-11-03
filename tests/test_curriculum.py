import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from thruster_landing.agents.replay import ReplayBuffer
from thruster_landing.agents.dqn import DQNConfig
from thruster_landing.curriculum import phase_at, choose_scenario, stop_after_level1
from thruster_landing.train import train
from thruster_landing.checkpoints import read_checkpoint

PROTOCOL=json.loads(Path('configs/curriculum_protocol.json').read_text())

def test_origin_survives_ring_checkpoint_and_legacy():
    replay=ReplayBuffer(2,1)
    for i in range(3):replay.add([i],i,0,[i+1],False,False,source_level=i+1)
    state=replay.state_dict();loaded=ReplayBuffer(2,1);loaded.load_state_dict(state)
    assert set(loaded.sample(2)['source_levels'].tolist())=={2,3}
    state['arrays'].pop('source_levels');loaded.load_state_dict(state)
    assert loaded.arrays['source_levels'].tolist()==[0,0]

def test_calendar_and_provenance():
    assert [phase_at(s,PROTOCOL)[0] for s in [0,49999,50000,149999,150000,299999]]==[1,1,2,2,3,3]
    cases=[dict(source_level=i,id=i) for i in [1,2,3]]
    rng=np.random.default_rng(42)
    origins=[choose_scenario(cases,rng,50000,PROTOCOL)['source_level'] for _ in range(2000)]
    assert .17<origins.count(1)/len(origins)<.23 and 3 not in origins
    empty={'episodes':[],'validation':[]}
    assert stop_after_level1([empty,empty])
    for key in ['episodes','validation']:
        nonempty=copy.deepcopy(empty);nonempty[key]=[{'success':True}]
        assert not stop_after_level1([empty,nonempty])

def test_curriculum_segment_resume_matches_uninterrupted(tmp_path):
    # A live episode crosses both segment stop and phase boundary without reset.
    protocol={**PROTOCOL,'phases':[dict(until=8,weights={'1':1.}),dict(until=100,weights={'1':.2,'2':.8})]}
    cfg=replace(DQNConfig(),total_steps=24,learning_starts=4,batch_size=4,validation_frequency=24,checkpoint_frequency=24,detailed_validation=False)
    kwargs=dict(train_path='configs/curriculum_train_scenarios.json',validation_path='configs/curriculum_validation_scenarios.json',curriculum=protocol)
    full,_=train(cfg,tmp_path/'full',**kwargs)
    first,_=train(cfg,tmp_path/'first',stop_steps=8,**kwargs)
    part=read_checkpoint(tmp_path/'first/last.resume.pt','resume')
    assert part['runner']['observation'] is not None
    resumed,_=train(cfg,tmp_path/'resumed',resume=tmp_path/'first/last.resume.pt',**kwargs)
    for k,v in full.online.state_dict().items():assert torch.equal(v,resumed.online.state_dict()[k])
    for k,v in full.replay.arrays.items():np.testing.assert_array_equal(v,resumed.replay.arrays[k])
    assert full.updates==resumed.updates

def test_only_budget_changes_and_all_new_resets_are_clear():
    from thruster_landing.envs import EnvConfig,ThrusterLandingEnv
    from thruster_landing.envs.physics import geometry
    from thruster_landing.experiments import assert_disjoint,load_manifest
    for alg in ['dqn','double_dqn']:
        baseline=json.loads(Path(f'configs/{alg}_beta10.json').read_text())
        new=json.loads(Path(f'configs/{alg}_curriculum.json').read_text())
        assert {k for k in new if new[k]!=baseline[k]}=={'total_steps'}
        assert new['total_steps']==300000
    train_manifest=load_manifest('configs/curriculum_train_scenarios.json','train')
    validation=load_manifest('configs/curriculum_validation_scenarios.json','validation')
    oldtrain=load_manifest('configs/rl_train_scenarios.json','train')
    assert_disjoint(train_manifest,validation,oldtrain)
    env=ThrusterLandingEnv(EnvConfig())
    for case in train_manifest['scenarios']+validation['scenarios']:
        _,info=env.reset(seed=case['seed'],options=case['options'])
        assert geometry(env.state,env.config)[:,1].min()>0
        assert env.state.gas==18000 and not info['contact'] and not info['is_success']
    env.close()
