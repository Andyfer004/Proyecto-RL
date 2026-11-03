"""Recompensa separada de los criterios físicos y de las métricas."""
import numpy as np


def potential(state, config):
    r = config.reward
    return -(r.horizontal_distance * abs(state.x) / config.position_scale
             + r.height * max(0.0, state.y - config.foot_drop) / config.height_scale)


def components(before, after, elapsed, outcome, config):
    r = config.reward
    return {
        "terminal": r.terminal_success if outcome == "success" else
                    (r.terminal_failure if outcome not in ("flying", "time_limit") else 0.0),
        "progress": r.progress * (potential(after, config) - potential(before, config)),
        "speed": -r.speed * elapsed * np.hypot(after.vx, after.vy) / config.velocity_scale,
        "orientation": -elapsed * (r.angle * abs(after.theta) / np.pi
                                    + r.angular_speed * abs(after.omega) / config.omega_scale),
        "gas": -r.gas * (before.gas - after.gas) / config.gas_capacity,
        "time": -r.time * elapsed,
    }
