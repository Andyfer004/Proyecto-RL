from dataclasses import replace
import numpy as np
import pytest
from thruster_landing.agents.pid import PIDConfig, PIDController, PIDLoop
from thruster_landing.envs import EnvConfig, ThrusterLandingEnv
from thruster_landing.envs.physics import State, wrench


@pytest.mark.parametrize("action,force,torque", [(0, 0, 0), (1, 1800, 0), (2, 600, -360),
                                                (3, 600, 360), (4, 1200, 0), (5, 2400, -360),
                                                (6, 2400, 360), (7, 3000, 0)])
def test_complete_catalog(action, force, torque):
    f, t, rate = wrench(action, 0, EnvConfig())
    np.testing.assert_allclose(f, [0, force])
    assert t == torque and rate == force


def test_horizontal_and_attitude_signs():
    p = PIDController(EnvConfig())
    p.act(State(x=1, y=10))
    assert p.diagnostics["target_vx"] < 0
    assert p.diagnostics["target_theta"] > 0
    assert p.diagnostics["requested_torque"] > 0
    p.reset()
    action = p.act(State(x=0, y=10, theta=0.2))
    assert p.diagnostics["requested_torque"] < 0
    assert wrench(action, 0.2, p.env_config)[1] < 0
    p.reset()
    p.act(State(x=0, y=10, omega=0.5))
    assert p.diagnostics["requested_torque"] < 0


def test_gravity_compensation_and_descent_profile():
    p = PIDController(EnvConfig())
    s = State(0, 10, vy=-p.config.descent_speed_limit)
    p.act(s)
    assert p.diagnostics["requested_fy"] == pytest.approx(p.env_config.mass * p.env_config.gravity)
    far = p.diagnostics["target_vy"]
    p.act(replace(s, y=1.01))
    assert far < p.diagnostics["target_vy"] < 0
    assert abs(p.diagnostics["target_theta"]) <= p.config.tilt_limit


def test_saturation_and_anti_windup():
    loop = PIDLoop(2, 1, 0, 0.5)
    for _ in range(100):
        assert loop.update(100, 0, 0.05, -1, 1) == 1
    assert loop.integral == 0  # No acumula al empujar contra el límite.
    for _ in range(100):
        loop.update(0.1, 0, 0.05, -1, 1)
    assert loop.integral == pytest.approx(0.5)
    loop.update(-0.1, 0, 0.05, -1, 1)
    assert loop.integral < 0.5  # Puede descargarse.
    with pytest.raises(ValueError):
        loop.update(0, 0, 0, -1, 1)


def test_controller_saturation_and_reset():
    p = PIDController(EnvConfig())
    initial = State(0.5, 10, vx=0.1, vy=-1)
    first = p.act(initial)
    for _ in range(10):
        p.act(initial)
    assert p.horizontal.integral != 0
    p.reset()
    assert p.previous_velocity is None and p.diagnostics == {}
    assert all(loop.integral == 0 for loop in (p.horizontal, p.vertical, p.attitude))
    assert p.act(initial) == first
    action = p.act(State(100, 10, vx=100, vy=-100, theta=0.8, omega=100))
    assert 0 <= action < 8
    assert abs(p.diagnostics["target_theta"]) <= p.config.tilt_limit
    assert 0 <= p.diagnostics["requested_fy"] <= p.max_force
    assert abs(p.diagnostics["requested_torque"]) <= p.max_torque


@pytest.mark.parametrize("theta", [0.0, 0.25, -0.3])
def test_allocator_recovers_each_exact_action(theta):
    p = PIDController(EnvConfig())
    for action in range(8):
        force, torque, _ = wrench(action, theta, p.env_config)
        assert p.allocate(force, torque, theta, 18000) == action


def test_allocator_prorates_low_gas_and_respects_weights():
    c = EnvConfig()
    p = PIDController(c)
    force, torque, rate = wrench(6, 0.2, c)
    gas = 12
    fraction = gas / (rate * c.decision_dt)
    assert p.allocate(force * fraction, torque * fraction, 0.2, gas) == 6
    assert p.allocate(force, torque, 0.2, 0) == 0
    # Solicitud incompatible: bajo peso de momento prima fuerza, alto peso prima giro.
    low = PIDController(c, replace(PIDConfig(), torque_weight=0.001))
    high = PIDController(c, replace(PIDConfig(), torque_weight=100))
    assert wrench(low.allocate([0, 1200], 360, 0, 18000), 0, c)[1] == 0
    assert wrench(high.allocate([0, 1200], 360, 0, 18000), 0, c)[1] == 360


def test_pid_does_not_mutate_state_and_empty_gas_resets():
    p = PIDController(EnvConfig())
    s = State(0.5, 10, gas=0)
    saved = s.to_dict()
    assert p.act(s) == 0
    assert s.to_dict() == saved
    assert p.vertical.integral == 0


def test_vertical_landing_in_validation_case():
    env = ThrusterLandingEnv()
    p = PIDController(env.config)
    _, info = env.reset(seed=1000, options={"initial_state": {"x": 0, "vx": 0, "y": 11,
                                                          "vy": -0.5, "theta": 0, "omega": 0}})
    for _ in range(500):
        action = p.act(State(**info["state"]), info["wind_force"])
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["outcome"] == "success"
    assert info["gas_consumed"] < env.config.gas_capacity
    env.close()


@pytest.mark.parametrize("change", [{"vertical_ki": -1}, {"tilt_limit": 2},
                                   {"force_y_weight": 0}, {"touchdown_speed": 3}])
def test_invalid_pid_config(change):
    with pytest.raises(ValueError):
        replace(PIDConfig(), **change)
