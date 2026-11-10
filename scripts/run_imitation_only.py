"""Ablación de imitación sola, en un directorio NUEVO con presupuesto propio.

Regenera las demostraciones PID de la versión 2 y entrena solo la fase de clasificación
para las tres combinaciones (algoritmo, semilla) de las candidatas entregadas. No ejecuta
TD, margen ni interacción online, y no toca `artifacts/recovery_v1` ni la prueba final.
El recuento de episodios/transiciones se compara con el registro original: si difiere,
las demostraciones no son las mismas y la ablación no sería comparable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CANDIDATES = [("imitation_only_dqn42", "dqn", 42),
              ("imitation_only_double42", "double_dqn", 42),
              ("imitation_only_dqn43", "dqn", 43)]
REFERENCE_DEMOS = Path("artifacts/recovery_v1/demonstrations_v2/metadata.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/recovery_assisted_v2.json"))
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--max-steps", type=int, default=1000000)
    args = p.parse_args()
    if not 0 < args.max_seconds <= 7200 or not 0 < args.max_steps <= 5000000:
        p.error("Máximo 7200 segundos y 5000000 pasos")
    args.output.mkdir(parents=True, exist_ok=False)

    now = time.time()
    (args.output / "budget.json").write_text(json.dumps(dict(
        start_epoch=now, deadline_epoch=now + args.max_seconds, wall_limit_seconds=args.max_seconds,
        step_limit=args.max_steps, actual_steps=0, stages=[],
        interventions=[dict(id="imitation_only", method="classification_only_v2",
                            scope="Ablación de TD; sin entrenamiento online ni prueba final")]),
        indent=2) + "\n")
    paths = [str(args.config)] + [f"configs/recovery_v2_{s}_scenarios.json"
                                 for s in ["train", "development", "check"]]
    (args.output / "frozen_protocol.json").write_text(json.dumps(dict(
        sha256={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in paths}), indent=2) + "\n")

    env = os.environ.copy()
    env["THRUSTER_RECOVERY_ROOT"] = str(args.output)
    env["MPLCONFIGDIR"] = str(args.output / "matplotlib_cache")
    env["PYTHONPATH"] = str(Path("scripts").resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    def run(name, *arguments):
        if time.time() >= now + args.max_seconds:
            raise RuntimeError("Presupuesto de tiempo agotado")
        print(name, flush=True)
        with (args.output / (name + ".log")).open("w") as log:
            subprocess.run([sys.executable, *arguments], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    run("demonstrations", "scripts/collect_recovery_demos.py", "--config", paths[0])
    demos = json.loads((args.output / "demonstrations_v2" / "metadata.json").read_text())
    reference = json.loads(REFERENCE_DEMOS.read_text())
    for key in ("episodes", "successful_episodes", "transitions", "observation_version"):
        if demos[key] != reference[key]:
            raise RuntimeError(f"Demostraciones distintas del registro original en {key}: "
                               f"{demos[key]} frente a {reference[key]}")
    print("demostraciones idénticas al registro original:",
          demos["successful_episodes"], "éxitos,", demos["transitions"], "transiciones", flush=True)

    for directory, algorithm, seed in CANDIDATES:
        run(directory, "scripts/train_imitation_only.py", "--config", paths[0],
            "--algorithm", algorithm, "--seed", str(seed), "--output", str(args.output / directory))
        meta = json.loads((args.output / directory / "metadata.json").read_text())
        print(directory, "desarrollo", meta["development_successes"], "/",
              meta["development_episodes"], meta["development_group_rates"], flush=True)

    record = json.loads((args.output / "budget.json").read_text())
    record.update(status="complete", finished_epoch=time.time())
    (args.output / "budget.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Terminado:", args.output, flush=True)


if __name__ == "__main__":
    main()
