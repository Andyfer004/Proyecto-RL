"""PID en cascada y asignación discreta de fuerza/momento; sin acceso al entorno."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np
from ..envs.config import EnvConfig
from ..envs.physics import ACTIONS, State, wrench


@dataclass(frozen=True)
class PIDConfig:
    position_kp: float = 0.5  # s^-1: error x -> velocidad objetivo
    horizontal_kp: float = 1.5  # s^-1: error vx -> aceleración
    horizontal_ki: float = 0.05  # s^-2
    horizontal_kd: float = 0.0
    vertical_kp: float = 4.0  # s^-1: error vy -> aceleración
    vertical_ki: float = 2.0
    vertical_kd: float = 0.0
    attitude_kp: float = 80.0  # s^-2: error theta -> aceleración angular
    attitude_ki: float = 0.0
    attitude_kd: float = 16.0  # s^-1: amortiguamiento omega medido
    integral_limit: float = 5.0  # cota de contribución integral en unidades de salida
    horizontal_speed_limit: float = 1.0
    horizontal_accel_limit: float = 2.0
    tilt_limit: float = 0.2
    landing_tilt_limit: float = 0.04
    leveling_height: float = 3.0
    descent_speed_limit: float = 2.0
    touchdown_speed: float = 0.4
    descent_braking_accel: float = 0.8
    force_x_weight: float = 0.5
    force_y_weight: float = 1.0
    torque_weight: float = 4.0
    wind_compensation: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "wind_compensation":
                if not isinstance(value, bool):
                    raise ValueError("wind_compensation debe ser booleano")
            elif not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} debe ser no negativo y finito")
        for name in ("horizontal_speed_limit", "horizontal_accel_limit", "tilt_limit",
                     "leveling_height", "descent_speed_limit", "descent_braking_accel",
                     "force_y_weight", "torque_weight"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")
        if not self.landing_tilt_limit <= self.tilt_limit < math.pi / 2:
            raise ValueError("Límites de inclinación inválidos")
        if not 0 < self.touchdown_speed <= self.descent_speed_limit:
            raise ValueError("touchdown_speed debe estar en (0, descent_speed_limit]")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class PIDLoop:
    """Derivada de la medición; integral acotada con integración condicional.

    `integral` es la contribución ki*integral(error dt), en unidades de salida.
    Congela la integral si el error empuja más hacia una saturación, pero permite
    descargarla. La derivada se recibe ya medida, evitando kicks de consigna.
    """
    def __init__(self, kp, ki, kd, integral_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0

    def update(self, error, measurement_rate, dt, low, high, feedforward=0.0):
        if dt <= 0 or low > high:
            raise ValueError("Intervalo PID inválido")
        candidate = float(np.clip(self.integral + self.ki * error * dt,
                                  -self.integral_limit, self.integral_limit))
        base = feedforward + self.kp * error - self.kd * measurement_rate
        unsaturated = base + candidate
        if not ((unsaturated > high and error > 0) or (unsaturated < low and error < 0)):
            self.integral = candidate
        return float(np.clip(base + self.integral, low, high))


class PIDController:
    def __init__(self, env_config: EnvConfig, config: PIDConfig | None = None):
        self.env_config, self.config = env_config, config or PIDConfig()
        p = self.config
        self.horizontal = PIDLoop(p.horizontal_kp, p.horizontal_ki, p.horizontal_kd, p.integral_limit)
        self.vertical = PIDLoop(p.vertical_kp, p.vertical_ki, p.vertical_kd, p.integral_limit)
        self.attitude = PIDLoop(p.attitude_kp, p.attitude_ki, p.attitude_kd, p.integral_limit)
        self.max_force = env_config.main_thrust + 2 * env_config.side_thrust
        self.max_torque = env_config.side_arm * env_config.side_thrust
        self.reset()

    def reset(self):
        for loop in (self.horizontal, self.vertical, self.attitude):
            loop.reset()
        self.previous_velocity = None
        self.diagnostics = {}

    def allocate(self, force, torque, theta, gas):
        """Error de wrench global medio en un período; combustible prorratea cada acción.

        El denominador permanece fijo en la autoridad nominal (N y N m). No se
        amplifican errores al agotarse gas. Los empates eligen el índice menor.
        """
        if gas <= 0:
            return 0
        c, p = self.env_config, self.config
        scores = []
        for action in range(len(ACTIONS)):
            f, t, rate = wrench(action, theta, c)
            fraction = min(1.0, gas / (rate * c.decision_dt)) if rate else 1.0
            residual = (f * fraction - force) / self.max_force
            scores.append(p.force_x_weight * residual[0]**2 + p.force_y_weight * residual[1]**2
                          + p.torque_weight * ((t * fraction - torque) / self.max_torque)**2)
        return int(np.argmin(scores))

    def act(self, state: State, wind_force=0.0):
        """Lee solo variables observables en SI y devuelve un índice Discrete(8)."""
        c, p = self.env_config, self.config
        dt = c.decision_dt
        if state.gas <= 0:
            self.reset()  # Evitar integración sin autoridad de control.
            return 0
        velocity = np.array([state.vx, state.vy])
        acceleration = (velocity - self.previous_velocity) / dt if self.previous_velocity is not None else np.zeros(2)
        self.previous_velocity = velocity
        height = max(0.0, state.y - c.foot_drop)
        target_vx = float(np.clip(-p.position_kp * state.x, -p.horizontal_speed_limit, p.horizontal_speed_limit))
        ax = self.horizontal.update(target_vx - state.vx, acceleration[0], dt,
                                    -p.horizontal_accel_limit, p.horizontal_accel_limit)
        target_vy = -min(p.descent_speed_limit,
                         math.sqrt(p.touchdown_speed**2 + 2 * p.descent_braking_accel * height))
        # Fuerza global vertical, limitada por la autoridad a la orientación actual.
        max_fy = self.max_force * max(0.0, math.cos(state.theta))
        ay = self.vertical.update(target_vy - state.vy, acceleration[1], dt,
                                  -c.gravity, max_fy / c.mass - c.gravity)
        fy = c.mass * (c.gravity + ay)
        fx = c.mass * ax - (wind_force if p.wind_compensation else 0.0)
        tilt_limit = p.landing_tilt_limit + (p.tilt_limit - p.landing_tilt_limit) * min(1.0, height / p.leveling_height)
        target_theta = float(np.clip(-math.atan2(fx, max(fy, 0.1)), -tilt_limit, tilt_limit))
        error = (target_theta - state.theta + math.pi) % (2 * math.pi) - math.pi
        alpha = self.attitude.update(error, state.omega, dt,
                                     -self.max_torque / c.inertia, self.max_torque / c.inertia)
        torque = c.inertia * alpha
        force = np.array([fx, fy])
        action = self.allocate(force, torque, state.theta, state.gas)
        self.diagnostics = {"target_vx": target_vx, "target_vy": target_vy,
                            "target_theta": target_theta, "requested_fx": fx,
                            "requested_fy": fy, "requested_torque": torque}
        return action
