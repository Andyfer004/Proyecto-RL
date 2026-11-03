from dataclasses import replace
import copy
import random
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from thruster_landing.agents.dqn import DQNAgent, DQNConfig, QNetwork, compute_targets
from thruster_landing.agents.replay import ReplayBuffer
from thruster_landing.checkpoints import (load_inference, read_checkpoint, restore_environment,
    restore_global_rng, save_inference, save_resume)
from thruster_landing.envs import EnvConfig, ThrusterLandingEnv
from thruster_landing.experiments import ROOT, assert_disjoint, load_manifest, prepare_experiment
from thruster_landing.rl_evaluation import evaluate
from thruster_landing.train import train


def tiny_config(**overrides):
    values = dict(buffer_capacity=128, batch_size=4, learning_starts=4, train_frequency=2,
                  target_update_frequency=8, total_steps=24, validation_frequency=100,
                  checkpoint_frequency=12, log_frequency=4)
    return DQNConfig(**{**values, **overrides})


def fill(agent):
    for i in range(8):
        agent.replay.add(np.ones(9)*i, i, float(i), np.ones(9)*(i+1), i == 7, i == 6)


def test_network_architecture_and_action_count():
    model = QNetwork(9, 8)
    assert model(torch.zeros(3, 9)).shape == (3, 8)
    assert [layer.out_features for layer in model if isinstance(layer, torch.nn.Linear)] == [128, 128, 8]


def test_replay_copies_ring_and_samples_without_replacement():
    replay = ReplayBuffer(4, 2, seed=3)
    observation = np.array([1., 2.])
    replay.add(observation, 0, 0, observation, False, False)
    observation[:] = 100
    assert replay.arrays['observations'][0, 0] == 1
    for i in range(1, 6):
        replay.add([i, i], i, i, [i+1, i+1], i == 5, i == 4)
    assert len(replay) == 4
    batch = replay.sample(4)
    assert set(batch['actions'].tolist()) == {2, 3, 4, 5}
    assert batch['terminated'].dtype == torch.bool
    with pytest.raises(ValueError):
        replay.sample(5)
    saved = replay.state_dict()
    clone = ReplayBuffer(4, 2)
    clone.load_state_dict(saved)
    a, b = replay.sample(3), clone.sample(3)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_known_targets_demonstrate_dqn_double_and_termination():
    rewards = torch.tensor([1., 1., 1.], requires_grad=True)
    # Principal elige acción 0; objetivo prefiere acción 1.
    online = torch.tensor([[10., 2.]] * 3, requires_grad=True)
    target = torch.tensor([[3., 8.]] * 3, requires_grad=True)
    terminated = torch.tensor([False, True, False])
    truncated = torch.tensor([False, False, True])
    dqn = compute_targets(rewards, terminated, online, target, .5)
    double = compute_targets(rewards, terminated, online, target, .5, True)
    torch.testing.assert_close(dqn, torch.tensor([5., 1., 5.]))
    torch.testing.assert_close(double, torch.tensor([2.5, 1., 2.5]))
    assert truncated[-1] and dqn[-1] > rewards[-1]  # truncamiento conserva bootstrap
    assert not dqn.requires_grad and not double.requires_grad
    assert online.grad is None and target.grad is None and rewards.grad is None


def test_learning_no_target_gradients_and_hard_sync():
    agent = DQNAgent(tiny_config())
    fill(agent)
    old_target = {k: v.clone() for k, v in agent.target.state_dict().items()}
    metrics = agent.learn()
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(p.grad is None and not p.requires_grad for p in agent.target.parameters())
    assert all(torch.equal(old_target[k], v) for k, v in agent.target.state_dict().items())
    assert any(not torch.equal(old_target[k], v) for k, v in agent.online.state_dict().items())
    agent.sync_target()
    assert all(torch.equal(v, agent.online.state_dict()[k]) for k, v in agent.target.state_dict().items())


def test_actions_and_epsilon_schedule():
    agent = DQNAgent(tiny_config(epsilon_decay_steps=10))
    assert agent.epsilon == 1
    for _ in range(100):
        assert 0 <= agent.act(np.zeros(9)) < 8
    agent.env_steps = agent.config.learning_starts + 10
    assert agent.epsilon == pytest.approx(agent.config.epsilon_end)
    assert 0 <= agent.act(np.zeros(9), explore=False) < 8


def test_equal_initialization_and_hyperparameters():
    from pathlib import Path
    a = DQNConfig.from_json(ROOT / 'configs/dqn.json')
    b = DQNConfig.from_json(ROOT / 'configs/double_dqn.json')
    assert replace(a, algorithm='double_dqn') == b
    dqn, double = DQNAgent(a), DQNAgent(b)
    assert all(torch.equal(v, double.online.state_dict()[k]) for k, v in dqn.online.state_dict().items())


def test_inference_checkpoint_and_kind_safety(tmp_path):
    agent = DQNAgent(tiny_config())
    fill(agent)
    agent.learn()
    path = tmp_path / 'policy.pt'
    save_inference(path, agent, {'env_config': EnvConfig().to_dict()})
    restored, experiment = load_inference(path)
    assert all(torch.equal(v, restored.online.state_dict()[k]) for k, v in agent.online.state_dict().items())
    assert len(restored.replay) == 0
    assert restored.act(np.ones(9), explore=False) == agent.act(np.ones(9), explore=False)
    with pytest.raises(ValueError):
        read_checkpoint(path, 'resume')


def test_complete_checkpoint_restores_rng_and_mid_episode(tmp_path):
    agent = DQNAgent(tiny_config())
    fill(agent)
    env = ThrusterLandingEnv()
    env.reset(seed=200)
    env.step(6)
    path = tmp_path / 'resume.pt'
    save_resume(path, agent, env, {'partial_return': 3.5}, {})
    expected_action = agent.act(env._observation())
    expected_transition = env.step(expected_action)
    expected_sample = agent.replay.sample(4)
    expected_random = (random.random(), np.random.random(), torch.rand(2))
    saved = read_checkpoint(path, 'resume')
    clone = DQNAgent(tiny_config())
    clone.load_state_dict(saved['agent'])
    restored_env = ThrusterLandingEnv()
    restore_environment(restored_env, saved['environment'])
    restore_global_rng(saved['global_rng'])
    assert clone.act(restored_env._observation()) == expected_action
    result = restored_env.step(expected_action)
    np.testing.assert_array_equal(result[0], expected_transition[0])
    assert result[1:] == expected_transition[1:]
    actual = clone.replay.sample(4)
    for key in expected_sample:
        assert torch.equal(expected_sample[key], actual[key])
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])


def test_evaluation_is_greedy_without_mutation():
    agent = DQNAgent(tiny_config())
    fill(agent)
    before = copy.deepcopy(agent.state_dict())
    torch_rng = torch.get_rng_state().clone()
    scenario = {'id': 'validation', 'seed': 30000, 'condition': 'vertical', 'options': {}}
    manifest = {'schema_version': 1, 'split': 'validation', 'scenarios': [scenario]}
    rows, _ = evaluate(agent, EnvConfig(), manifest)
    second, _ = evaluate(agent, EnvConfig(), manifest)
    assert rows == second and len(rows) == 1
    after = agent.state_dict()
    for name in ('online', 'target'):
        assert all(torch.equal(value, after[name][key]) for key, value in before[name].items())
    assert before['optimizer'] == after['optimizer']
    assert before['rng'] == after['rng']
    assert before['replay']['rng'] == after['replay']['rng']
    assert all(torch.equal(value, after['replay']['arrays'][key])
               for key, value in before['replay']['arrays'].items())
    assert before['updates'] == after['updates'] == 0
    assert torch.equal(torch_rng, torch.get_rng_state())


def test_splits_exclude_final_test_and_pid_validation():
    experiment = prepare_experiment(EnvConfig(), ROOT / 'configs/rl_train_scenarios.json', ROOT / 'configs/rl_validation_scenarios.json')
    final_test = load_manifest(ROOT / 'configs/pid_test_scenarios.json', 'test')
    assert_disjoint(experiment['train_manifest'], experiment['validation_manifest'], final_test)
    with pytest.raises(ValueError):
        load_manifest(ROOT / 'configs/pid_test_scenarios.json', 'train')
    with pytest.raises(ValueError):
        assert_disjoint(final_test, final_test)
    with pytest.raises(ValueError):
        prepare_experiment(replace(EnvConfig(), gravity=10), ROOT / 'configs/rl_train_scenarios.json', ROOT / 'configs/rl_validation_scenarios.json')


@pytest.mark.parametrize('algorithm', ['dqn', 'double_dqn'])
def test_training_resume_matches_continuous_execution(tmp_path, algorithm):
    paths = dict(train_path=ROOT / 'configs/rl_train_scenarios.json', validation_path=ROOT / 'configs/rl_validation_scenarios.json')
    config = tiny_config(algorithm=algorithm, total_steps=64, validation_frequency=32)
    continuous, full = train(config, tmp_path / 'full', **paths)
    train(replace(config, total_steps=32), tmp_path / 'partial', **paths)
    resumed, continued = train(config, tmp_path / 'continued', resume=tmp_path / 'partial/last.resume.pt', **paths)
    assert continuous.env_steps == resumed.env_steps == 64
    assert full['episodes'] and full['validation']
    assert continuous.updates == resumed.updates
    assert full == continued
    for key, value in continuous.online.state_dict().items():
        assert torch.equal(value, resumed.online.state_dict()[key])
    a, b = continuous.replay.sample(4), resumed.replay.sample(4)
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert continuous.optimizer.state_dict()['param_groups'] == resumed.optimizer.state_dict()['param_groups']
    for key, values in continuous.optimizer.state_dict()['state'].items():
        assert all(torch.equal(value, resumed.optimizer.state_dict()['state'][key][name]) for name, value in values.items())


def test_best_checkpoint_criterion_excludes_failed_contacts():
    from thruster_landing.train import validation_score, validation_rank
    zero = validation_score([{'success': False, 'contact_error_x_m': 0.0}])
    assert zero['landing_error_m'] is None
    bad = validation_score([{'success': True, 'contact_error_x_m': 1.0},
                            {'success': False, 'contact_error_x_m': 0.0}])
    good = validation_score([{'success': True, 'contact_error_x_m': 0.2},
                             {'success': False, 'contact_error_x_m': None}])
    more = validation_score([{'success': True, 'contact_error_x_m': 1.4}] * 2)
    assert validation_rank(more) > validation_rank(good) > validation_rank(bad) > validation_rank(zero)
    assert validation_rank(zero) == validation_rank(validation_score([{'success': False, 'contact_error_x_m': None}]))


def test_detailed_validation_exports_initial_best_and_final(tmp_path):
    from thruster_landing.train import validation_score, validation_rank
    config = tiny_config(total_steps=16, validation_frequency=8, detailed_validation=True)
    output = tmp_path / 'detailed'
    agent, runner = train(config, output, train_path=ROOT / 'configs/rl_train_scenarios.json',
                          validation_path=ROOT / 'configs/rl_validation_scenarios.json')
    assert agent.env_steps == 16
    assert read_checkpoint(output / 'initial.policy.pt')['env_steps'] == 0
    for prefix in ('initial', 'best', 'final'):
        assert (output / f'{prefix}_validation.csv').is_file()
        assert (output / f'{prefix}_validation.traces.csv').is_file()
    groups = [[row for row in runner['validation'] if row['env_steps'] == step] for step in (8,16)]
    ranks = [validation_rank(validation_score(rows)) for rows in groups]
    expected_step = (8,16)[ranks.index(max(ranks))]
    assert runner['best_validation']['env_steps'] == expected_step
    assert read_checkpoint(output / 'best.policy.pt')['env_steps'] == expected_step
    assert read_checkpoint(output / 'best.resume.pt')['agent']['env_steps'] == expected_step
    assert read_checkpoint(output / 'last.resume.pt')['agent']['env_steps'] == 16
