from dataclasses import replace
import pytest

torch=pytest.importorskip('torch')
from thruster_landing.agents.dqn import DQNAgent,DQNConfig
from thruster_landing.beta_diagnostics import measure_groups
from thruster_landing.offline_diagnostics import decompose


def test_beta_gradient_diagnostics_preserve_identity_and_beta1_reference():
    agent=DQNAgent(replace(DQNConfig(),hidden_sizes=(2,2)))
    with torch.no_grad():
        for p in agent.online.parameters():p.fill_(.1)
    agent.target.load_state_dict(agent.online.state_dict())
    batch={'observations':torch.ones(6,9),'next_observations':torch.ones(6,9),
        'actions':torch.tensor([0,1,2,3,4,5]),'rewards':torch.tensor([-20.,1.,1.,1.,1.,1.]),
        'terminated':torch.tensor([True,False,False,False,False,False]),'truncated':torch.zeros(6,dtype=torch.bool)}
    legacy,_,_=decompose(agent,batch)
    one,_,detail1=measure_groups(agent,batch,1.)
    ten,full10,detail10=measure_groups(agent,batch,10.)
    for old,new in zip(legacy,one):
        assert new['shared_weighted_gradient_norm']==pytest.approx(old['shared_weighted_contribution_norm'])
    assert full10['identity_error']<1e-6
    torch.testing.assert_close(detail1['q'],detail10['q'],rtol=0,atol=0)
    torch.testing.assert_close(detail1['target'],detail10['target'],rtol=0,atol=0)
    assert ten[0]['mean_abs_loss_derivative']==pytest.approx(1.)
    assert ten[1]['mean_abs_loss_derivative']==pytest.approx(one[1]['mean_abs_loss_derivative']/10)
