"""Configuración SI, inmutable y serializable en JSON."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class RewardConfig:
    terminal_success: float = 100.0
    terminal_failure: float = -100.0
    progress: float = 2.0
    horizontal_distance: float = 1.0
    height: float = 0.3
    speed: float = 0.08
    angle: float = 0.3
    angular_speed: float = 0.05
    gas: float = 5.0
    time: float = 0.2


@dataclass(frozen=True)
class EnvConfig:
    mass: float = 100.0
    width: float = 1.4
    height: float = 1.6
    foot_half_span: float = 0.6
    foot_drop: float = 1.0
    main_thrust: float = 1800.0
    side_thrust: float = 600.0
    side_arm: float = 0.6
    gravity: float = 9.81
    gas_capacity: float = 18000.0
    decision_dt: float = 0.05
    substeps: int = 10
    max_time: float = 25.0
    x_limit: float = 12.0
    ceiling: float = 25.0
    platform_half_width: float = 2.5
    success_x: float = 1.5
    success_vx: float = 0.7
    success_vy: float = 1.0
    success_angle: float = 0.12
    success_omega: float = 0.25
    support_height_tolerance: float = 0.15
    wind_mean: float = 0.0
    wind_amplitude: float = 0.0
    initial_x: tuple[float, float] = (-1.0, 1.0)
    initial_y: tuple[float, float] = (10.0, 12.0)
    initial_vx: tuple[float, float] = (-0.2, 0.2)
    initial_vy: tuple[float, float] = (-1.0, 0.0)
    initial_theta: tuple[float, float] = (-0.08, 0.08)
    initial_omega: tuple[float, float] = (-0.02, 0.02)
    position_scale: float = 12.0
    height_scale: float = 25.0
    velocity_scale: float = 10.0
    omega_scale: float = 3.0
    wind_scale: float = 100.0
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        positive = ("mass", "width", "height", "foot_half_span", "foot_drop",
                    "main_thrust", "side_thrust", "side_arm", "gravity", "gas_capacity",
                    "decision_dt", "max_time", "x_limit", "ceiling", "platform_half_width",
                    "position_scale", "height_scale", "velocity_scale", "omega_scale", "wind_scale")
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo y finito")
        if isinstance(self.substeps, bool) or not isinstance(self.substeps, int) or self.substeps < 1:
            raise ValueError("substeps debe ser un entero positivo")
        if self.foot_drop <= self.height / 2 or self.foot_half_span > self.width / 2:
            raise ValueError("Los apoyos deben estar debajo y dentro del ancho del cuerpo")
        for name in ("success_x", "success_vx", "success_vy", "success_angle",
                     "success_omega", "support_height_tolerance", "wind_amplitude"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} debe ser no negativo y finito")
        if not math.isfinite(self.wind_mean):
            raise ValueError("wind_mean debe ser finito")
        for name in ("initial_x", "initial_y", "initial_vx", "initial_vy", "initial_theta", "initial_omega"):
            values = getattr(self, name)
            if len(values) != 2 or not all(math.isfinite(x) for x in values) or values[0] > values[1]:
                raise ValueError(f"Rango inválido: {name}")
        if not all(math.isfinite(value) and value >= 0
                   for name, value in asdict(self.reward).items() if name != "terminal_failure"):
            raise ValueError("Pesos de recompensa inválidos")
        if not math.isfinite(self.reward.terminal_failure) or self.reward.terminal_failure > 0:
            raise ValueError("terminal_failure debe ser no positivo y finito")

    @property
    def inertia(self):
        return self.mass * (self.width**2 + self.height**2) / 12.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        data["reward"] = RewardConfig(**data.get("reward", {}))
        return cls(**data)
