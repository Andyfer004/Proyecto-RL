from dataclasses import replace
import numpy as np
import pytest
from thruster_landing.envs.config import EnvConfig
from thruster_landing.envs.physics import State, geometry, integrate, wrench

C = EnvConfig()


def test_free_fall_analytic():
    state = State(0, 12, vy=-1, gas=C.gas_capacity)
    initial = state
    dt = C.decision_dt / C.substeps
    for _ in range(200):
        state = integrate(state, 0, dt, 0, C)
    time = 200 * dt
    assert state.y == pytest.approx(initial.y + initial.vy * time - C.gravity * time**2 / 2)
    assert state.vy == pytest.approx(initial.vy - C.gravity * time)
    assert state.gas == initial.gas


def test_force_rotation_and_torque_signs():
    f, torque, _ = wrench(1, np.pi / 2, C)
    np.testing.assert_allclose(f, [-C.main_thrust, 0], atol=1e-10)
    assert torque == 0
    assert wrench(2, 0, C)[1] == pytest.approx(-C.side_arm * C.side_thrust)
    assert wrench(3, 0, C)[1] == pytest.approx(C.side_arm * C.side_thrust)
    assert wrench(4, 0, C)[1] == 0
    assert wrench(7, 0, C)[0][1] > C.mass * C.gravity


def test_angular_dynamics():
    s = integrate(State(0, 10, gas=100), 3, 0.01, 0, C)
    alpha = C.side_arm * C.side_thrust / C.inertia
    assert s.omega == pytest.approx(alpha * 0.01)
    assert s.theta == pytest.approx(alpha * 0.01**2 / 2)


def test_gas_exhaustion_limits_impulse_and_splits_burn():
    s = State(0, 10, gas=9.0)
    final = integrate(s, 1, 0.05, 0, C)
    assert final.gas == 0
    assert final.vy == pytest.approx(9 / C.mass - C.gravity * 0.05)
    # Impulso de 9 N s a lo largo de los primeros 0.005 s, luego caída libre.
    expected_y = 10 - 0.5 * C.gravity * 0.05**2 + 9 / C.mass * (0.05 - 0.005 / 2)
    assert final.y == pytest.approx(expected_y)


def test_consumption_sums_all_thrusters():
    s = integrate(State(0, 10, gas=1000), 7, 0.05, 0, C)
    assert s.gas == pytest.approx(850)
    assert s.omega == 0


def test_no_gas_no_thrust():
    s = State(0, 10, gas=0)
    assert integrate(s, 7, 0.1, 0, C) == integrate(s, 0, 0.1, 0, C)


def test_wind_is_force_at_center_of_mass():
    s = integrate(State(0, 10), 0, 0.1, 80, C)
    assert s.vx == pytest.approx(0.08)
    assert s.omega == 0


def test_support_geometry():
    points = geometry(State(0, C.foot_drop), C)
    np.testing.assert_allclose(points[4:, 1], 0)
    assert points[:4, 1].min() > 0


def test_braking_authority_for_training_envelope():
    # Máxima velocidad inicial descendente: 1 m/s; demora de decisión 0.05 s.
    speed = abs(C.initial_vy[0]) + C.gravity * C.decision_dt
    deceleration = (C.main_thrust + 2 * C.side_thrust) / C.mass - C.gravity
    stop_distance = speed**2 / (2 * deceleration)
    available = C.initial_y[0] - C.foot_drop - abs(C.initial_vy[0]) * C.decision_dt
    assert deceleration > 0 and stop_distance < available


@pytest.mark.parametrize("field,value", [("mass", 0), ("substeps", 0), ("substeps", 1.5),
                                          ("gas_capacity", -1), ("wind_mean", float("nan"))])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        replace(C, **{field: value})


def test_rotating_thrust_converges_with_substeps():
    def trajectory(dt):
        state = State(0, 10, gas=C.gas_capacity)
        for _ in range(round(0.2 / dt)):
            state = integrate(state, 6, dt, 0, C)
        return np.array([state.x, state.y, state.vx, state.vy])

    reference = trajectory(0.0001)
    coarse_error = np.linalg.norm(trajectory(0.005) - reference)
    fine_error = np.linalg.norm(trajectory(0.0025) - reference)
    assert fine_error < 0.6 * coarse_error


def test_gas_budget_over_many_actions():
    state = State(0, 10, gas=1000)
    spent = 0.0
    rng = np.random.default_rng(8)
    for action in rng.integers(0, 8, size=100):
        rate = wrench(action, state.theta, C)[2]
        expected = min(state.gas, rate * 0.005)
        updated = integrate(state, action, 0.005, 0, C)
        spent += state.gas - updated.gas
        assert state.gas - updated.gas == pytest.approx(expected)
        assert updated.gas >= 0
        state = updated
    assert spent == pytest.approx(1000 - state.gas)
