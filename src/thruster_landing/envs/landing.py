"""Entorno Gymnasium de aterrizaje 2D con contacto geométrico."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .config import EnvConfig
from .physics import ACTIONS, State, geometry, integrate
from .rewards import components


class ThrusterLandingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config=None, render_mode=None):
        self.config = config or EnvConfig()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"render_mode inválido: {render_mode}")
        self.render_mode = render_mode
        self.metadata = {**type(self).metadata, "render_fps": round(1 / self.config.decision_dt)}
        self.action_space = spaces.Discrete(len(ACTIONS))
        # No recortar: las escalas no son cotas físicas; velocidades pueden superarlas.
        low = np.array([-np.inf] * 4 + [-1, -1, -np.inf, 0, -np.inf], dtype=np.float32)
        high = np.array([np.inf] * 4 + [1, 1, np.inf, 1, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None
        self._renderer = None
        self._done = True
        self.last_action = 0

    def _observation(self):
        s, c = self.state, self.config
        return np.array([s.x / c.position_scale, (s.y - c.foot_drop) / c.height_scale,
                         s.vx / c.velocity_scale, s.vy / c.velocity_scale,
                         np.sin(s.theta), np.cos(s.theta), s.omega / c.omega_scale,
                         s.gas / c.gas_capacity, self.wind / c.wind_scale], dtype=np.float32)

    def _info(self, outcome="flying", reward_components=None):
        contact = geometry(self.state, self.config)[:, 1].min() <= 1e-7
        return {"state": self.state.to_dict(), "wind_force": self.wind, "time": self.time,
                "outcome": outcome, "is_success": outcome == "success", "contact": bool(contact),
                "gas_consumed": self.initial_gas - self.state.gas,
                "reward_components": dict(reward_components or {}),
                "contact_metrics": {"horizontal_error": abs(self.state.x), "vx": self.state.vx,
                                    "vy": self.state.vy, "theta": self.state.theta,
                                    "omega": self.state.omega} if contact else None}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        options = options or {}
        unknown = set(options) - {"initial_state", "wind_force"}
        if unknown:
            raise ValueError(f"Opciones desconocidas: {unknown}")
        values = {name: float(self.np_random.uniform(*getattr(c, "initial_" + name)))
                  for name in ("x", "y", "vx", "vy", "theta", "omega")}
        values["gas"] = c.gas_capacity
        overrides = options.get("initial_state", {})
        if set(overrides) - set(values):
            raise ValueError("initial_state contiene campos desconocidos")
        values.update(overrides)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Estado inicial no finito")
        if not 0 <= values["gas"] <= c.gas_capacity or not -np.pi <= values["theta"] <= np.pi:
            raise ValueError("Gas o ángulo inicial fuera de rango")
        self.state = State(**values)
        self.wind = float(options.get("wind_force", c.wind_mean +
                          self.np_random.uniform(-c.wind_amplitude, c.wind_amplitude)))
        if not np.isfinite(self.wind):
            raise ValueError("Viento no finito")
        if self._boundary(self.state) is not None:
            raise ValueError("El estado inicial debe estar en vuelo dentro del área")
        self.time = 0.0
        self.initial_gas = self.state.gas
        self._done = False
        self.last_action = 0
        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def _boundary(self, state):
        points = geometry(state, self.config)
        if points[:, 1].min() <= 0:
            return "contact"
        if np.max(np.abs(points[:, 0])) >= self.config.x_limit or points[:, 1].max() >= self.config.ceiling:
            return "out_of_bounds"
        return None

    def _contact_outcome(self):
        s, c = self.state, self.config
        points = geometry(s, c)
        feet = points[4:]
        if points[:4, 1].min() <= 1e-6:
            return "body_contact"
        if not np.all(np.abs(feet[:, 0]) <= c.platform_half_width):
            return "missed_platform"
        valid = (abs(s.x) <= c.success_x and abs(s.vx) <= c.success_vx
                 and -c.success_vy <= s.vy <= 0 and abs(s.theta) <= c.success_angle
                 and abs(s.omega) <= c.success_omega
                 and feet[:, 1].max() <= c.support_height_tolerance)
        return "success" if valid else "invalid_contact"

    def step(self, action):
        if self._done:
            raise RuntimeError("Llamar reset() antes de step() o después de finalizar")
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")
        c, before = self.config, self.state
        self.last_action = int(action)
        start = self.time
        outcome = "flying"
        for _ in range(c.substeps):
            dt = min(c.decision_dt / c.substeps, c.max_time - self.time)
            if dt <= 1e-12:
                break
            previous = self.state
            candidate = integrate(previous, action, dt, self.wind, c)
            event = self._boundary(candidate)
            if event:
                # Localizar primer cruce dentro del subpaso y consumir solo el gas hasta él.
                low, high = 0.0, dt
                for _ in range(35):
                    middle = (low + high) / 2
                    if self._boundary(integrate(previous, action, middle, self.wind, c)):
                        high = middle
                    else:
                        low = middle
                dt = high
                candidate = integrate(previous, action, dt, self.wind, c)
                event = self._boundary(candidate)
            self.state = candidate
            self.time += dt
            if event:
                outcome = self._contact_outcome() if event == "contact" else event
                break
        terminated = outcome != "flying"
        truncated = not terminated and self.time >= c.max_time - 1e-10
        if truncated:
            outcome = "time_limit"
        self._done = terminated or truncated
        reward_parts = components(before, self.state, self.time - start, outcome, c)
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(sum(reward_parts.values())), terminated, truncated, self._info(outcome, reward_parts)

    def render(self):
        if self.render_mode is None or self.state is None:
            return None
        if self._renderer is None:
            from .rendering import Renderer
            self._renderer = Renderer(self.render_mode, self.config)
        return self._renderer.draw(self.state, self.last_action, self.time, self.wind)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
