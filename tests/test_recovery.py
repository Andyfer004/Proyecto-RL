import copy,json,time
from pathlib import Path
import numpy as np
import pytest
import torch
from thruster_landing.recovery_agent import AssistedDQN,landing_features,large_margin
from thruster_landing.recovery_budget import Budget,BudgetExceeded
from thruster_landing.experiments import assert_disjoint

CONFIG=json.loads(Path('configs/recovery_assisted_v1.json').read_text())


def fake_demos():
    return dict(observations=torch.zeros(2,9),actions=torch.tensor([0,0]),rewards=torch.tensor([1.,2.]),next_observations=torch.zeros(2,9),terminated=torch.tensor([False,True]),n_next_observations=torch.zeros(2,9),n_rewards=torch.tensor([10.,20.]),n_gamma=torch.tensor([.81,.81]),n_terminated=torch.tensor([False,True]),train_mask=torch.tensor([True,False]))


def test_landing_scales_are_fixed_invertible_and_batch_consistent():
    raw=np.array([1.5/12,2/25,.7/10,-.1,np.sin(.12),np.cos(.12),.25/3,.5,.4],np.float32)
    result=landing_features(raw,CONFIG)
    np.testing.assert_allclose(result,[1,1,1,-1,np.sin(.12)/.12,np.cos(.12),1,.5,.4],rtol=1e-6)
    np.testing.assert_array_equal(landing_features(np.stack([raw,raw]),CONFIG),np.stack([result,result]))
    np.testing.assert_allclose(result/np.asarray(CONFIG['observation_multipliers']),raw,rtol=1e-6)


def test_margin_enforces_expert_order_and_does_not_depend_on_q_offset():
    q=torch.tensor([[0.,1.]],requires_grad=True);a=torch.tensor([0]);loss=large_margin(q,a,.8)
    assert float(loss.detach())==pytest.approx(1.8)
    torch.testing.assert_close(torch.autograd.grad(loss,q)[0],torch.tensor([[-1.,1.]]))
    assert float(large_margin(q+100,a,.8).detach())==pytest.approx(1.8,abs=1e-5)
    assert float(large_margin(torch.tensor([[1.,0.]]),a,.8))==0


@pytest.mark.parametrize('algorithm',['dqn','double_dqn'])
def test_actual_assisted_one_and_multistep_targets_no_broadcast_or_target_grad(algorithm,monkeypatch):
    c={**CONFIG,'algorithm':algorithm,'batch_size':2,'gamma':.9}
    d=fake_demos();agent=AssistedDQN(c,d)
    agent.online=torch.nn.Linear(9,8);agent.target=torch.nn.Linear(9,8)
    with torch.no_grad():
        agent.online.weight.zero_();agent.online.bias.fill_(-10);agent.online.bias[0]=2;agent.online.bias[1]=4
        agent.target.weight.zero_();agent.target.bias.zero_();agent.target.bias[0]=5;agent.target.bias[1]=1
    agent.target.requires_grad_(False);agent.optimizer=torch.optim.SGD(agent.online.parameters(),lr=0.)
    monkeypatch.setattr(agent,'sample_demo',lambda n:d)
    result=agent.learn(offline=True)
    first_next=5 if algorithm=='dqn' else 1
    one_targets=np.array([1+.9*first_next,2.]);n_targets=np.array([10+.81*first_next,20.])
    def smooth(errors):return np.where(np.abs(errors)<10,errors**2/20,np.abs(errors)-5).mean()
    assert result['td_loss']==pytest.approx(smooth(one_targets-2),abs=1e-5)
    assert result['nstep_loss']==pytest.approx(smooth(n_targets-2),abs=1e-5)
    assert result['margin_loss']==pytest.approx(2.8)
    assert all(p.grad is None for p in agent.target.parameters())


def test_demo_sampling_excludes_grouped_training_holdout():
    d=fake_demos();d['observations'][1].fill_(100.)
    agent=AssistedDQN(CONFIG,d);batch=agent.sample_demo(1024)
    assert batch['observations'].eq(0).all()


def test_loaded_policy_requires_no_teacher_and_does_not_change_rng_or_weights(tmp_path,monkeypatch):
    from thruster_landing.agents.pid import PIDController
    def forbidden(*args,**kwargs):raise AssertionError('Teacher queried in inference')
    monkeypatch.setattr(PIDController,'act',forbidden)
    agent=AssistedDQN(CONFIG);obs=np.zeros(9,np.float32);expected=agent.act(obs);agent.save(tmp_path/'policy.pt',dict(assisted=True))
    loaded,meta=AssistedDQN.load(tmp_path/'policy.pt');rng=copy.deepcopy(loaded.rng.bit_generator.state)
    weights={k:v.clone() for k,v in loaded.online.state_dict().items()}
    assert loaded.act(obs)==expected and loaded.demos is None
    assert loaded.rng.bit_generator.state==rng
    for key,value in weights.items():assert torch.equal(value,loaded.online.state_dict()[key])


def test_new_development_and_check_cases_are_disjoint_from_training():
    sets=[json.loads(Path(f'configs/recovery_{name}_scenarios.json').read_text()) for name in ['train','development','check']]
    assert_disjoint(*sets)
    assert len(sets[1]['scenarios'])==len(sets[2]['scenarios'])==64


def test_budget_stops_before_next_environment_step_on_either_limit():
    b=object.__new__(Budget);b.data=dict(deadline_epoch=time.time()+100,actual_steps=5,step_limit=5)
    with pytest.raises(BudgetExceeded):b.guard()
    b.data.update(actual_steps=0,deadline_epoch=time.time()-1)
    with pytest.raises(BudgetExceeded):b.guard()


def test_closed_intervention_budget_rejects_further_steps():
    b=object.__new__(Budget);b.data=dict(status='complete',deadline_epoch=time.time()+100,actual_steps=0,step_limit=5000000)
    with pytest.raises(BudgetExceeded):b.guard()


def test_coverage_intervention_only_changes_training_support_and_grouped_protocol():
    c=json.loads(Path('configs/recovery_assisted_v2.json').read_text())
    allowed={'version','train_manifest','development_manifest','demo_directory','goal_groups'}
    assert {k:v for k,v in c.items() if k not in allowed}=={k:v for k,v in CONFIG.items() if k not in allowed}
    manifests=[json.loads(Path(f'configs/recovery_v2_{s}_scenarios.json').read_text()) for s in ['train','development','check']]
    assert_disjoint(*manifests)
    for manifest in manifests:
        cases=manifest['scenarios']
        assert sum(s['condition']=='level1' for s in cases)==sum(s['condition']=='original' for s in cases)
        assert all(s['options']['initial_state']['y']>1.5 for s in cases)
