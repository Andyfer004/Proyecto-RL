"""Checkpoints tipados, carga restringida y estado explícito del simulador v0."""
from dataclasses import replace
import json
from pathlib import Path
import random
import numpy as np
import torch
from .agents.dqn import DQNAgent, DQNConfig
from .envs.physics import State

FORMAT_VERSION = 1


def plain(value):
    return json.loads(json.dumps(value, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()))


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def global_rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_global_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"].cpu())


def environment_state(env):
    return plain({"state": env.state.to_dict(), "wind": env.wind, "time": env.time,
                  "initial_gas": env.initial_gas, "done": env._done, "last_action": env.last_action,
                  "rng": env.np_random.bit_generator.state, "seed": env.np_random_seed,
                  "action_rng": env.action_space.np_random.bit_generator.state})


def restore_environment(env, state):
    env.reset(seed=state["seed"])
    env.state = State(**state["state"])
    env.wind, env.time = state["wind"], state["time"]
    env.initial_gas, env._done, env.last_action = state["initial_gas"], state["done"], state["last_action"]
    env.np_random.bit_generator.state = state["rng"]
    env.action_space.np_random.bit_generator.state = state["action_rng"]


def save_inference(path, agent, experiment):
    atomic_save({"format_version": FORMAT_VERSION, "kind": "inference", "config": agent.config.to_dict(),
                 "observation_dim": agent.observation_dim, "action_count": agent.action_count,
                 "online": agent.online.state_dict(), "env_steps": agent.env_steps,
                 "experiment": experiment}, path)


def save_resume(path, agent, env, runner, experiment):
    # Inicializar el RNG de action_space antes de capturar los RNG globales.
    snapshot = environment_state(env)
    atomic_save({"format_version": FORMAT_VERSION, "kind": "resume", "agent": agent.state_dict(),
                 "environment": snapshot, "runner": plain(runner), "experiment": experiment,
                 "global_rng": global_rng_state()}, path)


def read_checkpoint(path, expected_kind=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Versión de checkpoint no soportada")
    if expected_kind and payload.get("kind") != expected_kind:
        raise ValueError(f"Se requiere checkpoint {expected_kind}")
    return payload


def load_inference(path, device="cpu"):
    payload = read_checkpoint(path, "inference")
    config = replace(DQNConfig(**payload["config"]), device=device)
    agent = DQNAgent(config, payload["observation_dim"], payload["action_count"])
    agent.online.load_state_dict(payload["online"])
    agent.target.load_state_dict(payload["online"])
    agent.env_steps = payload["env_steps"]
    return agent, payload["experiment"]
