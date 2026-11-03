import copy
from dataclasses import replace
import pytest

torch=pytest.importorskip('torch')
from thruster_landing.agents.dqn import DQNAgent,DQNConfig
from thruster_landing.offline_diagnostics import decompose,one_update_copy,targets_for


def fixture():
    agent=DQNAgent(replace(DQNConfig(),hidden_sizes=(2,2)))
    with torch.no_grad():
        for parameter in agent.online.parameters():parameter.fill_(.1)
    agent.target.load_state_dict(agent.online.state_dict())
    batch={'observations':torch.ones(6,9),'next_observations':torch.ones(6,9),
           'actions':torch.tensor([0,1,2,3,4,5]),'rewards':torch.tensor([-10.,1.,1.,1.,1.,1.]),
           'terminated':torch.tensor([True,False,False,False,False,False]),'truncated':torch.zeros(6,dtype=torch.bool)}
    return agent,batch


def test_proportional_gradient_decomposition_and_undefined_cosine():
    agent,batch=fixture()
    groups,full,_=decompose(agent,batch)
    assert [row['count'] for row in groups]==[1,5]
    assert groups[0]['proportion']==pytest.approx(1/6)
    assert groups[0]['shared_weighted_contribution_norm']==pytest.approx(groups[0]['shared_gradient_mean_norm']/6)
    assert groups[1]['shared_weighted_contribution_norm']==pytest.approx(groups[1]['shared_gradient_mean_norm']*5/6)
    assert full['weighted_sum_verification_error']<1e-6
    assert full['cosine_shared_groups']==pytest.approx(-1,abs=1e-6)
    assert sum(row['projection_share_full_gradient'] for row in groups)==pytest.approx(1,abs=1e-5)
    batch['terminated'].zero_()
    groups,full,_=decompose(agent,batch)
    assert groups[0]['count']==0 and groups[0]['shared_weighted_contribution_norm']==0
    assert full['cosine_shared_groups'] is None


def test_one_step_copies_preserve_weights_optimizer_and_proportions():
    agent,batch=fixture()
    # Inicializar momentos Adam no nulos y conservarlos para comparaciones pareadas.
    agent.optimizer.zero_grad(set_to_none=True)
    agent.online(batch['observations']).sum().backward();agent.optimizer.step()
    saved=copy.deepcopy(agent.optimizer.state_dict())
    weights={key:value.clone() for key,value in agent.online.state_dict().items()}
    targets=targets_for(agent,batch)
    for mode in ('saved_Adam_state','fresh_Adam_state'):
        first,metrics=one_update_copy(agent.online,saved,agent.config.learning_rate,mode,batch,targets,batch['terminated'],10)
        second,_=one_update_copy(agent.online,saved,agent.config.learning_rate,mode,batch,targets,batch['terminated'],10)
        assert all(torch.equal(value,second.state_dict()[key]) for key,value in first.state_dict().items())
        expected=torch.nn.functional.smooth_l1_loss(agent.online(batch['observations']).gather(1,batch['actions'][:,None]).squeeze(1)[batch['terminated']],targets[batch['terminated']],reduction='sum')/6
        assert metrics['update_loss']==pytest.approx(float(expected.detach()))
    assert all(torch.equal(value,agent.online.state_dict()[key]) for key,value in weights.items())
    for identifier,values in saved['state'].items():
        for key,value in values.items():assert torch.equal(value,agent.optimizer.state_dict()['state'][identifier][key])
