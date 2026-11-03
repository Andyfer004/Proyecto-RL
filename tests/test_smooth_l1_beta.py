from dataclasses import replace
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from thruster_landing.agents.dqn import DQNAgent,DQNConfig


def test_smooth_l1_beta10_values_and_slopes_are_not_huber_delta10():
    agent=DQNAgent(replace(DQNConfig(),smooth_l1_beta=10.0,buffer_capacity=8,batch_size=3))
    assert type(agent.loss_function) is torch.nn.SmoothL1Loss
    assert agent.loss_function.beta==10 and agent.loss_function.reduction=='mean'
    error=torch.tensor([-20.,-10.,-2.,-.5,0.,.5,2.,10.,20.],requires_grad=True)
    target=torch.zeros_like(error)
    loss=agent.loss_function(error,target)
    gradient=torch.autograd.grad(loss,error)[0]
    expected=torch.where(error.detach().abs()<10,error.detach().square()/20,error.detach().abs()-5)
    torch.testing.assert_close(loss,expected.mean())
    torch.testing.assert_close(gradient,torch.clamp(error.detach()/10,-1,1)/len(error))
    huber=torch.nn.HuberLoss(delta=10)(error,target)
    torch.testing.assert_close(huber,10*loss)
    assert not torch.allclose(loss,huber)


def test_actual_learn_beta10_has_no_compensating_multiplier(monkeypatch):
    agent=DQNAgent(replace(DQNConfig(),smooth_l1_beta=10.0,buffer_capacity=8,batch_size=3))
    agent.online=torch.nn.Linear(9,8)
    with torch.no_grad():
        agent.online.weight.zero_();agent.online.bias.zero_();agent.online.bias[0]=2;agent.online.bias[1]=20
    agent.optimizer=torch.optim.SGD(agent.online.parameters(),lr=0.)
    batch={'observations':torch.zeros(3,9),'next_observations':torch.zeros(3,9),
           'actions':torch.tensor([0,1,0]),'rewards':torch.zeros(3),
           'terminated':torch.ones(3,dtype=torch.bool),'truncated':torch.zeros(3,dtype=torch.bool)}
    monkeypatch.setattr(agent.replay,'sample',lambda size,device:batch)
    metrics=agent.learn()
    assert metrics['loss']==pytest.approx((.2+15+.2)/3)
    expected=torch.zeros(8);expected[0]=.4/3;expected[1]=1/3
    torch.testing.assert_close(agent.online.bias.grad,expected)


def test_beta1_backwards_compatible_with_original_functional_loss():
    assert DQNConfig(**{'algorithm':'dqn'}).smooth_l1_beta==1
    errors=torch.tensor([-20.,-2.,-.5,0.,.5,2.,20.],requires_grad=True)
    target=torch.zeros_like(errors)
    legacy=torch.nn.functional.smooth_l1_loss(errors,target)
    current=torch.nn.SmoothL1Loss(beta=1)(errors,target)
    torch.testing.assert_close(current,legacy,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(current,errors,retain_graph=True)[0],torch.autograd.grad(legacy,errors)[0],rtol=0,atol=0)
