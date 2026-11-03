"""Dinámica de cuerpo rígido, sin dependencias de Gymnasium ni gráficos."""
from dataclasses import asdict, dataclass, replace
import numpy as np
from .config import EnvConfig

# Orden: central, lateral izquierdo, lateral derecho.
ACTIONS = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                    (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)], dtype=np.float64)
ACTIONS.setflags(write=False)
ACTION_NAMES = ("apagado", "central", "izquierdo", "derecho", "laterales",
                "central+izquierdo", "central+derecho", "todos")


@dataclass(frozen=True)
class State:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0
    omega: float = 0.0
    gas: float = 18000.0

    def to_dict(self):
        return asdict(self)


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def thrusters(config: EnvConfig):
    points = np.array([[0.0, -config.height / 2],
                       [-config.side_arm, -config.height / 2],
                       [config.side_arm, -config.height / 2]])
    forces = np.array([[0.0, config.main_thrust], [0.0, config.side_thrust],
                       [0.0, config.side_thrust]])
    return points, forces


def wrench(action, theta, config):
    points, forces = thrusters(config)
    active = forces * ACTIONS[action, :, None]
    local = active.sum(axis=0)
    torque = np.sum(points[:, 0] * active[:, 1] - points[:, 1] * active[:, 0])
    return rotation(theta) @ local, float(torque), float(np.linalg.norm(active, axis=1).sum())


def geometry(state, config):
    """Cuatro esquinas del cuerpo y dos apoyos; índices 4 y 5 son apoyos."""
    w, h = config.width / 2, config.height / 2
    local = np.array([[-w, -h], [w, -h], [w, h], [-w, h],
                      [-config.foot_half_span, -config.foot_drop],
                      [config.foot_half_span, -config.foot_drop]])
    return local @ rotation(state.theta).T + [state.x, state.y]


def _segment(state, force, torque, duration, wind, config, gas):
    acceleration = (force + [wind, -config.mass * config.gravity]) / config.mass
    alpha = torque / config.inertia
    return State(x=state.x + state.vx * duration + 0.5 * acceleration[0] * duration**2,
                 y=state.y + state.vy * duration + 0.5 * acceleration[1] * duration**2,
                 vx=state.vx + acceleration[0] * duration,
                 vy=state.vy + acceleration[1] * duration,
                 theta=state.theta + state.omega * duration + 0.5 * alpha * duration**2,
                 omega=state.omega + alpha * duration, gas=gas)


def integrate(state: State, action: int, dt: float, wind: float, config: EnvConfig):
    """Fuerza global congelada por subpaso. Separar combustión y caída si se agota gas."""
    force, torque, rate = wrench(action, state.theta, config)
    burn_dt = min(dt, state.gas / rate) if rate > 0 else 0.0
    current = state
    if burn_dt > 0:
        current = _segment(state, force, torque, burn_dt, wind, config,
                           max(0.0, state.gas - rate * burn_dt))
    if burn_dt < dt:
        current = _segment(current, np.zeros(2), 0.0, dt - burn_dt, wind, config, current.gas)
    return replace(current, theta=float((current.theta + np.pi) % (2 * np.pi) - np.pi))
