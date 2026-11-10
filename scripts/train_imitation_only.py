"""Solo imitación: entropía cruzada sobre acciones PID. Cero TD, cero margen, cero pasos online.

Reproduce exactamente la fase A de `train_recovery.py` y se detiene ahí. Con la misma
semilla, las mismas demostraciones y el mismo orden de minibatches, los pesos resultantes
son los del candidato asistido al terminar su clasificación: la única diferencia frente
a la candidata entregada son las actualizaciones TD y de margen que vienen después.
Sirve como ablación del aporte de TD frente a la imitación por sí sola.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from thruster_landing.evaluation import write_csv
from thruster_landing.recovery_agent import AssistedDQN
from thruster_landing.recovery_budget import Budget, ROOT
from thruster_landing.recovery_evaluation import evaluate_candidate, export_candidate


def group_rates(rows):
    names = sorted({row["condition"] for row in rows})
    return {name: sum(r["success"] for r in rows if r["condition"] == name)
                  / sum(r["condition"] == name for r in rows) for name in names}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=["dqn", "double_dqn"], default="dqn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/recovery_assisted_v2.json"))
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    cfg.update(algorithm=args.algorithm, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget(f"imitation_only_{cfg['version']}_{args.algorithm}_seed{args.seed}")
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()

    demos = torch.load(ROOT / cfg["demo_directory"] / "demonstrations.pt", weights_only=True)
    agent = AssistedDQN(cfg, demos)
    heldout = ~demos["train_mask"]
    metrics = []
    try:
        for update in range(cfg["pretrain_classification_updates"]):
            if update % 100 == 0:
                budget.guard()
            loss = agent.classification_update()
            if (update + 1) % 1000 == 0:
                with torch.no_grad():
                    predicted = agent.online(demos["observations"][heldout]).argmax(1)
                    accuracy = float((predicted == demos["actions"][heldout]).float().mean())
                metrics.append(dict(phase="classification", updates=agent.updates, loss=loss,
                                    heldout_action_accuracy=accuracy))
                print("CE", update + 1, "loss", loss, "heldout", accuracy, flush=True)
        agent.sync()

        development = json.loads(Path(cfg["development_manifest"]).read_text())
        rows, traces = evaluate_candidate(agent, development, budget=budget, trace=True)
        rates = group_rates(rows)
        successes = sum(r["success"] for r in rows)
        agent.save(args.output / "imitation_only.policy.pt",
                   dict(phase="classification_only", online_steps=0, td_updates=0, margin_updates=0,
                        development_success_rate=successes / len(rows), development_group_rates=rates,
                        imitation_only=True, assisted=False, teacher_actions_at_eval=False))
        export_candidate(args.output / "development.csv", rows, traces)
        write_csv(args.output / "metrics.csv", metrics)
        (args.output / "metadata.json").write_text(json.dumps(dict(
            config=cfg, imitation_only=True, classification_updates=agent.updates,
            td_updates=0, margin_updates=0, online_steps=0, learning_environment_steps=0,
            development_successes=successes, development_episodes=len(rows),
            development_group_rates=rates,
            heldout_action_accuracy=metrics[-1]["heldout_action_accuracy"],
            demonstrations_source=str(ROOT / cfg["demo_directory"] / "demonstrations.pt"),
            reward_changed=False, observation_version=cfg["observation_version"],
            wall_seconds=time.perf_counter() - start), indent=2) + "\n")
        print(args.algorithm, args.seed, "imitación sola: desarrollo",
              successes, "/", len(rows), rates, flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    main()
