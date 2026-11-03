"""Materializa conjuntos RL independientes; nunca lee resultados PID ni prueba."""
import json
from pathlib import Path
import numpy as np


def generate(split, first_seed, count):
    scenarios = []
    for i in range(count):
        seed = first_seed + i
        rng = np.random.default_rng(seed)
        state = {"x": 0.0, "y": float(rng.uniform(10, 12)), "vx": 0.0,
                 "vy": float(rng.uniform(-1, 0)), "theta": 0.0, "omega": 0.0}
        condition = "vertical"
        if split == "validation" and i >= count // 2:
            condition = "small_offset"
            state.update(x=float(rng.uniform(-0.2, 0.2)), vx=float(rng.uniform(-0.05, 0.05)),
                         theta=float(rng.uniform(-0.02, 0.02)))
        scenarios.append({"id": f"rl-{split}-{i:03d}", "condition": condition, "seed": seed,
                          "options": {"initial_state": state, "wind_force": 0.0}})
    return {"schema_version": 1, "split": split, "scenarios": scenarios}


if __name__ == "__main__":
    for split, first_seed, count in (("train", 20000, 128), ("validation", 30000, 32)):
        path = Path(f"configs/rl_{split}_scenarios.json")
        if path.exists():
            raise SystemExit(f"No se sobrescribe: {path}")
        path.write_text(json.dumps(generate(split, first_seed, count), indent=2) + "\n")
