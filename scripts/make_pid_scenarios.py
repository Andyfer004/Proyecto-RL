"""Materializa splits antes del ajuste; sin ejecutar el controlador."""
import json
from pathlib import Path
import numpy as np


def generate(split, base_seed, count):
    scenarios = []
    for condition_index, condition in enumerate(("vertical", "offset", "tilted", "wind")):
        for index in range(count):
            seed = base_seed + condition_index * 100 + index
            rng = np.random.default_rng(seed)
            state = {"x": 0.0, "y": float(rng.uniform(10, 12)), "vx": 0.0,
                     "vy": float(rng.uniform(-1, 0)), "theta": 0.0, "omega": 0.0}
            if condition != "vertical":
                state.update(x=float(rng.uniform(-0.8, 0.8)), vx=float(rng.uniform(-0.15, 0.15)))
            if condition in ("tilted", "wind"):
                state.update(theta=float(rng.uniform(-0.18, 0.18)), omega=float(rng.uniform(-0.03, 0.03)))
            wind = float(rng.uniform(-40, 40)) if condition == "wind" else 0.0
            scenarios.append({"id": f"{split}-{condition}-{index:03d}", "condition": condition,
                              "seed": seed, "options": {"initial_state": state, "wind_force": wind}})
    return {"schema_version": 1, "split": split, "scenarios": scenarios}


if __name__ == "__main__":
    for split, seed, count in (("validation", 1000, 8), ("test", 9000, 25)):
        path = Path(f"configs/pid_{split}_scenarios.json")
        if path.exists():
            raise SystemExit(f"No se sobrescribe un manifiesto existente: {path}")
        path.write_text(json.dumps(generate(split, seed, count), indent=2) + "\n")
