"""Pruebas de integración nuevas: gather/broadcast y transición a través de reset."""
from dataclasses import replace
import importlib
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from thruster_landing.agents.dqn import DQNAgent, DQNConfig
from thruster_landing.envs import EnvConfig, ThrusterLandingEnv
from thruster_landing.experiments import ROOT, load_manifest
from thruster_landing.rl_evaluation import evaluate


def test_actual_learn_gathers_stored_actions_and_vector_targets(monkeypatch):
    config = replace(DQNConfig(), buffer_capacity=8, batch_size=3, gamma=0.5)
    agent = DQNAgent(config)
    agent.online = torch.nn.Linear(9, 8)
    agent.target = torch.nn.Linear(9, 8)
    with torch.no_grad():
        for model in (agent.online, agent.target):
            model.weight.zero_()
            model.bias.copy_(torch.arange(8, dtype=torch.float32))
    agent.target.requires_grad_(False)
    agent.optimizer = torch.optim.SGD(agent.online.parameters(), lr=0.0)
    batch = {'observations': torch.zeros(3,9), 'next_observations': torch.ones(3,9),
             'actions': torch.tensor([7,0,3]), 'rewards': torch.tensor([10.,1.,-2.]),
             'terminated': torch.tensor([True,False,True]), 'truncated': torch.tensor([False,True,False])}
    monkeypatch.setattr(agent.replay, 'sample', lambda size, device: batch)
    import thruster_landing.agents.dqn as module
    original_loss = module.F.smooth_l1_loss
    seen = {}
    def loss(q, targets, **kwargs):
        seen.update(q=q.detach().clone(), targets=targets.detach().clone())
        assert q.shape == targets.shape == batch['rewards'].shape == batch['terminated'].shape == (3,)
        return original_loss(q, targets, **kwargs)
    monkeypatch.setattr(module.F, 'smooth_l1_loss', loss)
    metrics = agent.learn()
    torch.testing.assert_close(seen['q'], torch.tensor([7.,0.,3.]))
    torch.testing.assert_close(seen['targets'], torch.tensor([10.,4.5,-2.]))
    expected = torch.zeros(8)
    expected[7], expected[0], expected[3] = -1/3, -1/3, 1/3
    torch.testing.assert_close(agent.online.bias.grad, expected)
    assert metrics['loss'] == pytest.approx(11/3)
    assert all(p.grad is None for p in agent.target.parameters())


@pytest.mark.parametrize('ending', ['terminal','truncated'])
def test_training_stores_final_observation_before_reset_and_syncs_on_schedule(monkeypatch,tmp_path,ending):
    module = importlib.import_module('thruster_landing.train')
    real_agent, real_env = module.DQNAgent, module.ThrusterLandingEnv
    events, final_observations, reset_observations, actions = [], [], [], []
    class AuditEnv(real_env):
        def __init__(self, config):
            super().__init__(replace(config,max_time=0.001) if ending=='truncated' else config)
        def reset(self, *,seed=None,options=None):
            # Solo fixture: forzar un contacto o truncamiento por paso para probar resets.
            state = {'x':0.,'y':1.001,'vx':0.,'vy':-2.,'theta':0.,'omega':0.} if ending=='terminal' else {'x':0.,'y':10.,'vx':0.,'vy':0.,'theta':0.,'omega':0.}
            result = super().reset(seed=seed,options={'initial_state':state,'wind_force':0.})
            reset_observations.append(result[0].copy())
            return result
        def step(self,action):
            actions.append(action)
            result = super().step(action)
            assert result[2] or result[3]
            final_observations.append(result[0].copy())
            return result
    class AuditAgent(real_agent):
        def __init__(self,config):
            super().__init__(config)
            assert all(torch.equal(v,self.target.state_dict()[k]) for k,v in self.online.state_dict().items())
        def sync_target(self):
            events.append(self.env_steps)
            super().sync_target()
            assert all(torch.equal(v,self.target.state_dict()[k]) for k,v in self.online.state_dict().items())
    monkeypatch.setattr(module,'DQNAgent',AuditAgent)
    monkeypatch.setattr(module,'ThrusterLandingEnv',AuditEnv)
    config = replace(DQNConfig(),total_steps=6,buffer_capacity=8,batch_size=1,learning_starts=1,
                     train_frequency=1,target_update_frequency=2,validation_frequency=100,checkpoint_frequency=6)
    agent,_ = module.train(config,tmp_path/'audit',train_path=ROOT/'configs/rl_train_scenarios.json',
                           validation_path=ROOT/'configs/rl_validation_scenarios.json')
    assert len(final_observations)==len(reset_observations)==6
    np.testing.assert_array_equal(agent.replay.arrays['next_observations'][:6],final_observations)
    np.testing.assert_array_equal(agent.replay.arrays['observations'][:6],reset_observations)
    np.testing.assert_array_equal(agent.replay.arrays['actions'][:6],actions)
    assert all(not np.array_equal(a,b) for a,b in zip(reset_observations,final_observations))
    assert agent.replay.arrays['terminated'][:6].all() == (ending=='terminal')
    assert agent.replay.arrays['truncated'][:6].all() == (ending=='truncated')
    assert events == [2,4,6] and agent.target_syncs==3


def test_training_and_greedy_evaluation_receive_identical_observation():
    scenario=load_manifest(ROOT/'configs/rl_validation_scenarios.json','validation')['scenarios'][16]
    env=ThrusterLandingEnv()
    observation,_=env.reset(seed=scenario['seed'],options=scenario['options'])
    captured=[]
    class Reader:
        config=SimpleNamespace(algorithm='audit')
        def act(self,obs,explore):
            assert not explore
            captured.append(obs.copy())
            return 0
    evaluate(Reader(),EnvConfig(),{'split':'validation','scenarios':[scenario]})
    np.testing.assert_array_equal(captured[0],observation)
    assert captured[0].shape==(9,) and captured[0].dtype==np.float32
    env.close()
