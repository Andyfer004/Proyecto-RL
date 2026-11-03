from dataclasses import replace
import json
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
import thruster_landing
from thruster_landing.envs import EnvConfig, ThrusterLandingEnv


def env_at(**state):
    env = ThrusterLandingEnv()
    env.reset(seed=4, options={"initial_state": {"x": 0, "y": 10, "vx": 0, "vy": 0,
                                              "theta": 0, "omega": 0, **state}})
    return env


def test_gymnasium_contract():
    env = gym.make("ThrusterLanding-v0")
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_seed_reproducibility_with_wind():
    config = replace(EnvConfig(), wind_amplitude=80)
    first, second = ThrusterLandingEnv(config), ThrusterLandingEnv(config)
    a, ia = first.reset(seed=123)
    b, ib = second.reset(seed=123)
    np.testing.assert_array_equal(a, b)
    assert ia == ib
    for action in [0, 1, 2, 3, 4, 5, 6, 7] * 3:
        a, ra, ta, xa, ia = first.step(action)
        b, rb, tb, xb, ib = second.step(action)
        np.testing.assert_array_equal(a, b)
        assert (ra, ta, xa, ia) == (rb, tb, xb, ib)
        if ta or xa:
            break


def test_observation_scaling_and_copies():
    env = env_at(x=2, y=10, gas=9000)
    before = env._observation()
    assert before[0] == pytest.approx(2 / 12)
    assert before[1] == pytest.approx(9 / 25)
    assert before[7] == 0.5
    old = before.copy()
    env.step(0)
    np.testing.assert_array_equal(before, old)
    assert env.observation_space.contains(env._observation())


def test_success_uses_feet_before_center():
    env = env_at(y=1.001, vy=-0.1)
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["is_success"]
    assert info["state"]["y"] == pytest.approx(1, abs=1e-7)
    assert info["time"] < env.config.decision_dt
    assert info["contact_metrics"] is not None


@pytest.mark.parametrize("state,outcome", [({"y": 1.001, "vy": -3}, "invalid_contact"),
                                         ({"y": 1.001, "x": 3}, "missed_platform"),
                                         ({"y": 0.701, "theta": np.pi / 2}, "body_contact"),
                                         ({"x": 11.29, "vx": 4}, "out_of_bounds")])
def test_failure_types(state, outcome):
    env = env_at(**state)
    for _ in range(10):
        _, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["outcome"] == outcome
    assert reward < 0


def test_platform_edge_requires_both_supports():
    env = env_at(x=2.0, y=1.001)
    assert env.step(0)[4]["outcome"] == "missed_platform"


def test_angle_and_angular_velocity_thresholds():
    for state in ({"theta": 0.13, "y": 1.08}, {"omega": 0.5, "y": 1.001}):
        env = env_at(**state)
        for _ in range(20):
            _, _, terminated, _, info = env.step(0)
            if terminated:
                break
        assert info["outcome"] == "invalid_contact"


def test_gas_depletion_is_not_terminal():
    env = env_at(gas=1)
    _, _, terminated, truncated, info = env.step(7)
    assert info["state"]["gas"] == 0
    assert not terminated and not truncated
    assert info["gas_consumed"] == pytest.approx(1)
    assert info["reward_components"]["gas"] == pytest.approx(-5 / 18000)


def test_truncation_and_no_more_steps():
    env = ThrusterLandingEnv(replace(EnvConfig(), max_time=0.013))
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and truncated
    assert info["outcome"] == "time_limit"
    assert info["time"] == pytest.approx(0.013)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_reward_components_sum_and_no_charge_for_remaining_gas():
    env = env_at()
    _, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(sum(info["reward_components"].values()))
    assert info["reward_components"]["gas"] == 0
    assert info["reward_components"]["time"] < 0


def test_partial_contact_step_only_spends_available_time():
    env = env_at(y=1.001, vy=-0.5)
    initial = env.state.gas
    _, _, _, _, info = env.step(1)
    assert info["outcome"] == "success"
    assert initial - env.state.gas == pytest.approx(env.config.main_thrust * info["time"])


def test_reset_options_and_invalid_actions():
    env = env_at()
    with pytest.raises(ValueError):
        env.step(8)
    with pytest.raises(ValueError):
        env.reset(options={"initial_state": {"y": 0}})
    with pytest.raises(ValueError):
        env.reset(options={"initial_state": {"gas": -1}})
    with pytest.raises(ValueError):
        env.reset(options={"typo": 1})


def test_config_roundtrip(tmp_path):
    config = replace(EnvConfig(), wind_amplitude=80)
    path = tmp_path / "env.json"
    path.write_text(json.dumps(config.to_dict()))
    loaded = EnvConfig.from_json(path)
    assert loaded.to_dict() == json.loads(json.dumps(config.to_dict()))


def test_contact_time_matches_ballistic_solution():
    env = env_at(y=1.005, vy=-0.2)
    _, _, terminated, _, info = env.step(0)
    expected = (-0.2 + np.sqrt(0.2**2 + 2 * env.config.gravity * 0.005)) / env.config.gravity
    assert terminated
    assert info["time"] == pytest.approx(expected, abs=1e-9)
    assert info["state"]["vy"] == pytest.approx(-0.2 - env.config.gravity * expected)


def test_empty_vehicle_continues_to_impact():
    env = env_at(y=10, gas=0)
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(7)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["outcome"] == "invalid_contact"
    assert info["state"]["gas"] == 0
