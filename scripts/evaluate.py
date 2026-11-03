"""Evalúa/reproduce una política en validación. No abre la prueba final en Etapa 3."""
import argparse
import json
from pathlib import Path
from thruster_landing.checkpoints import load_inference
from thruster_landing.experiments import ROOT, assert_disjoint, digest, load_manifest, verify_reference
from thruster_landing.play import versions
from thruster_landing.rl_evaluation import config_from_dict, evaluate, export_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, default=Path("configs/rl_validation_scenarios.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    verify_reference()
    manifest = load_manifest(args.scenarios, "validation")
    agent, experiment = load_inference(args.checkpoint, args.device)
    assert_disjoint(experiment["train_manifest"], manifest,
                    load_manifest(ROOT / "configs/pid_test_scenarios.json", "test"))
    if args.episodes is not None:
        if not 1 <= args.episodes <= len(manifest["scenarios"]):
            parser.error("episodes debe ser un prefijo válido del manifiesto")
        manifest = {**manifest, "scenarios": manifest["scenarios"][:args.episodes]}
    config = config_from_dict(experiment["env_config"])
    rows, traces = evaluate(agent, config, manifest, args.render, args.trace)
    export_evaluation(args.output, rows, traces if args.trace else None)
    metadata = {"versions": versions(), "checkpoint_sha256": digest(args.checkpoint),
                "manifest": manifest, "config": config.to_dict(), "training_steps": agent.env_steps,
                "exploration": False, "updates_during_evaluation": 0,
                "completed_episodes": len(rows), "algorithm": agent.config.algorithm}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{agent.config.algorithm}: éxitos={sum(row['success'] for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
