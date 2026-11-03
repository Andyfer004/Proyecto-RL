"""Evaluación greedy sin cambiar pesos, optimizador, replay ni RNG del agente."""
import json
import numpy as np
from .envs import EnvConfig, ThrusterLandingEnv
from .envs.config import RewardConfig
from .evaluation import summarize, write_csv


def config_from_dict(data):
    return EnvConfig(**{**data, "reward": RewardConfig(**data["reward"])})


def episode_metrics(info, initial, total_return, steps, scenario, algorithm, split):
    contact = info["contact_metrics"] or {}
    return {"controller": algorithm, "seed": scenario["seed"], "scenario_id": scenario["id"],
            "condition": scenario["condition"], "split": split, "steps": steps,
            "outcome": info["outcome"], "success": info["is_success"],
            "duration_s": info["time"], "return": total_return,
            "gas_consumed_Ns": info["gas_consumed"], "wind_force_N": info["wind_force"],
            "initial_state": json.dumps(initial, sort_keys=True), "final_state": json.dumps(info["state"], sort_keys=True),
            "final_error_x_m": abs(info["state"]["x"]), "final_center_height_m": info["state"]["y"],
            "contact_error_x_m": contact.get("horizontal_error"), "contact_vx_m_s": contact.get("vx"),
            "contact_vy_m_s": contact.get("vy"), "contact_theta_rad": contact.get("theta"),
            "contact_omega_rad_s": contact.get("omega")}


def evaluate(agent, env_config, manifest, render=False, trace=False):
    rows, traces = [], []
    env = ThrusterLandingEnv(env_config, render_mode="human" if render else None)
    try:
        for scenario in manifest["scenarios"]:
            observation, info = env.reset(seed=scenario["seed"], options=scenario["options"])
            initial, total, steps = info["state"], 0.0, 0
            parts = {}
            if trace:
                traces.append({"scenario_id": scenario["id"], "condition": scenario["condition"],
                               "time": 0.0, **initial, "action": "", "outcome": "flying"})
            while True:
                if env._renderer and env._renderer.close_requested:
                    return rows, traces
                action = agent.act(observation, explore=False)
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                for key, value in info["reward_components"].items():
                    parts[key] = parts.get(key, 0.0) + float(value)
                if trace:
                    traces.append({"scenario_id": scenario["id"], "condition": scenario["condition"],
                                   "time": info["time"], **info["state"], "action": action, "outcome": info["outcome"]})
                if terminated or truncated:
                    row = episode_metrics(info, initial, total, steps, scenario, agent.config.algorithm, manifest["split"])
                    row.update(terminated=terminated, truncated=truncated)
                    row.update({"reward_" + key: value for key, value in parts.items()})
                    rows.append(row)
                    break
    finally:
        env.close()
    return rows, traces


def export_evaluation(output, rows, traces=None):
    write_csv(output, rows)
    write_csv(output.with_suffix(".summary.csv"), summarize(rows))
    if traces is not None:
        write_csv(output.with_suffix(".traces.csv"), traces)
