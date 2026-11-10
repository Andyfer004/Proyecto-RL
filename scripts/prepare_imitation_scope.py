"""Congela un complemento nuevo y el protocolo de la ablacion imitacion sola frente a asistido.

El conjunto final `scope_final_v1_complement.json` ya se consumio: no se reutiliza. Este
complemento usa los mismos cinco grupos y el mismo emparejamiento de viento, con una base
de semillas nueva, y se verifica disjunto de todos los manifiestos existentes. Las
candidatas asistidas entregadas no se reentrenan ni se reseleccionan: se evaluan los pesos
congelados tal cual. Se escribe antes de evaluar y se rechaza sobrescribir.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from prepare_scope_evaluation import HIGH, LOW, exposure
from thruster_landing.experiments import assert_disjoint
from thruster_landing.scope_evaluation import sha256

GROUPS = ["represented_l1", "represented_high", "new_displacement", "new_dynamics", "wind_control"]
ASSISTED = [("dqn_assisted42", "dqn", 42,
             "artifacts/scope_alignment_v1/frozen_models/dqn_assisted42.policy.pt"),
            ("double_assisted42", "double_dqn", 42,
             "artifacts/scope_alignment_v1/frozen_models/double_assisted42.policy.pt"),
            ("dqn_assisted43", "dqn", 43,
             "artifacts/scope_alignment_v1/frozen_models/dqn_assisted43.policy.pt")]
IMITATION = [("imitation_only_dqn42", "dqn", 42, "imitation_only_dqn42"),
             ("imitation_only_double42", "double_dqn", 42, "imitation_only_double42"),
             ("imitation_only_dqn43", "dqn", 43, "imitation_only_dqn43")]


def complementary_cases(seed_base, n=32):
    """Misma construccion que el complemento final; solo cambia la base de semillas."""
    cases = []
    for group_index, group in enumerate(GROUPS):
        for index in range(n):
            seed = seed_base + group_index * 1000 + index
            rng = np.random.default_rng(seed)
            ranges = LOW if group == "represented_l1" else HIGH
            state = {k: float(rng.uniform(*v)) for k, v in ranges.items()}
            if group == "new_displacement":
                state["x"] = float(rng.choice([-1, 1]) * rng.uniform(.6, 1.5))
            if group == "new_dynamics":
                state.update(theta=float(rng.choice([-1, 1]) * rng.uniform(.06, .12)),
                             vx=float(rng.choice([-1, 1]) * rng.uniform(.2, .5)),
                             vy=float(rng.uniform(-1.8, -.8)),
                             omega=float(rng.uniform(-.05, .05)))
            winds = [0., -20., 20., -40., 40.] if group == "wind_control" else [0.]
            for wind_index, wind in enumerate(winds):
                condition = group if wind == 0 else (
                    "wind_" + ("m" if wind < 0 else "p") + str(int(abs(wind))))
                cases.append(dict(id="ablation-%s-%03d" % (condition, index), condition=condition,
                                  seed=seed + 100000 * wind_index, suite="ablation_complement",
                                  exposure=exposure(state, wind),
                                  paired_wind_id=("wind-base-%03d" % index
                                                  if group == "wind_control" else ""),
                                  options=dict(initial_state=state.copy(), wind_force=wind)))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imitation-root", type=Path, required=True,
                   help="Salida de run_imitation_only.py con los checkpoints de clasificacion")
    p.add_argument("--root", type=Path, default=Path("artifacts/imitation_ablation_v1"))
    p.add_argument("--seed-base", type=int, default=1440000)
    args = p.parse_args()

    protocol_path = Path("configs/imitation_ablation_v1_protocol.json")
    complement_path = Path("configs/imitation_ablation_v1_complement.json")
    if protocol_path.exists() or complement_path.exists():
        raise ValueError("Protocolo ya congelado; no sobrescribir")
    args.root.mkdir(parents=True, exist_ok=True)
    models = args.root / "frozen_models"
    models.mkdir(exist_ok=False)

    methods = [dict(id="pid", kind="pid", label="PID congelado",
                    selection="configs/pid.json congelado antes del desarrollo RL")]
    for mid, algorithm, seed, source in ASSISTED:
        payload = torch.load(source, weights_only=True, map_location="cpu")
        cfg = payload["config"]
        assert cfg["algorithm"] == algorithm and cfg["seed"] == seed
        assert payload["updates"] > cfg["pretrain_classification_updates"], "El asistido debe tener TD"
        copy = models / (mid + ".policy.pt")
        shutil.copyfile(source, copy)
        methods.append(dict(id=mid, kind="assisted", algorithm=algorithm, seed=seed,
                            checkpoint=str(copy), source=source, family="imitation_plus_td",
                            label=("DQN" if algorithm == "dqn" else "Double DQN")
                                  + " asistido (CE + TD + margen), semilla %d" % seed,
                            classification_updates=cfg["pretrain_classification_updates"],
                            total_updates=payload["updates"], online_steps=payload["env_steps"],
                            selection="Entregado sin cambios; sin reseleccion en este complemento"))
    for mid, algorithm, seed, directory in IMITATION:
        source = args.imitation_root / directory / "imitation_only.policy.pt"
        payload = torch.load(source, weights_only=True, map_location="cpu")
        cfg = payload["config"]
        assert cfg["algorithm"] == algorithm and cfg["seed"] == seed
        assert payload["updates"] == cfg["pretrain_classification_updates"], "Debe ser solo clasificacion"
        assert payload["env_steps"] == 0, "La imitacion sola no interactua online"
        copy = models / (mid + ".policy.pt")
        shutil.copyfile(source, copy)
        methods.append(dict(id=mid, kind="assisted", algorithm=algorithm, seed=seed,
                            checkpoint=str(copy), source=str(source), family="imitation_only",
                            label=("DQN" if algorithm == "dqn" else "Double DQN")
                                  + " imitacion sola (solo CE), semilla %d" % seed,
                            classification_updates=cfg["pretrain_classification_updates"],
                            total_updates=payload["updates"], online_steps=0,
                            selection="Unico checkpoint posible: fin de la fase de clasificacion",
                            training_metadata=str(args.imitation_root / directory / "metadata.json")))

    complement = dict(schema_version=1, split="test", role="imitation_ablation_complement",
                      origin=("Semillas por caso NumPy default_rng desde %d; prescrito sin evaluar "
                              "ninguna politica; sin rechazo por resultado" % args.seed_base),
                      scenarios=complementary_cases(args.seed_base))
    for existing in Path("configs").glob("*scenarios.json"):
        assert_disjoint(complement, json.loads(existing.read_text()))
    assert_disjoint(complement,
                    json.loads(Path("configs/scope_final_v1_complement.json").read_text()))
    for case in complement["scenarios"]:
        s = case["options"]["initial_state"]
        theta = s["theta"]
        clearance = min(s["y"] - .6 * abs(np.sin(theta)) - np.cos(theta),
                        s["y"] - .7 * abs(np.sin(theta)) - .8 * np.cos(theta))
        assert clearance > .1 and abs(s["x"]) + 1 < 12 and s["y"] + 1 < 25 and "gas" not in s
    # Bytes estables entre sistemas: finales LF explicitos, como el resto de configs.
    complement_path.write_bytes((json.dumps(complement, indent=2) + "\n").encode())

    fixed = [Path("configs/env.json"), Path("configs/pid.json"), complement_path]
    fixed += [Path(m[k]) for m in methods for k in ["checkpoint", "source"] if k in m]
    fixed += list(Path("src/thruster_landing").rglob("*.py"))
    fixed += [Path("scripts") / s for s in ["prepare_imitation_scope.py",
                                            "evaluate_imitation_scope.py",
                                            "prepare_scope_evaluation.py",
                                            "train_imitation_only.py"]]
    protocol = dict(version="imitation_ablation_v1",
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(), methods=methods,
                    complement_manifest=str(complement_path), reserved_manifest=None,
                    env_config="configs/env.json", pid_config="configs/pid.json",
                    question=("Aportan algo las actualizaciones TD y de margen sobre la "
                              "imitacion sola?"),
                    design=("Mismos pesos iniciales, mismas demostraciones y mismo orden de "
                            "minibatches en la fase de clasificacion; la familia asistida "
                            "continua con TD a uno y diez pasos mas margen, la familia de "
                            "imitacion se detiene ahi"),
                    no_training=True, no_adjustment_after_results=True,
                    reuses_consumed_final_test=False,
                    limits=("Una semilla principal por algoritmo y una de sensibilidad con "
                            "demostraciones compartidas; viento constante hasta 40 N; tasas "
                            "descriptivas sin afirmacion de superioridad general"),
                    artifacts_root=args.root.as_posix(),
                    # Claves en forma posix: `str(Path)` en Windows daria `\` y el
                    # protocolo solo verificaria en ese sistema.
                    sha256={p.as_posix(): sha256(p) for p in fixed})
    protocol_path.write_bytes((json.dumps(protocol, indent=2) + "\n").encode())
    (args.root / "protocol_sha256.json").write_bytes(
        (json.dumps({protocol_path.as_posix(): sha256(protocol_path)}, indent=2) + "\n").encode())
    print("Congelados", len(complement["scenarios"]), "casos y", len(methods), "metodos")


if __name__ == "__main__":
    main()
