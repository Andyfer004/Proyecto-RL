"""Reproducción manual, aleatoria, PID o de checkpoints RL (sin entrenamiento)."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys

from .envs import EnvConfig, ThrusterLandingEnv
from .envs.physics import State
from .agents.pid import PIDConfig, PIDController
from .evaluation import load_scenarios, summarize, write_csv


def versions():
    result = {"python": sys.version, "platform": platform.platform()}
    for name in ("thruster-landing", "numpy", "gymnasium", "pygame-ce", "pytest", "torch", "pandas", "matplotlib"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("random", "manual", "pid", "dqn", "double_dqn"), default="random")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Por defecto: todos los escenarios del manifiesto, o 3 episodios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pid-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint .policy.pt para reproducción RL greedy")
    parser.add_argument("--scenarios", type=Path, default=None)
    parser.add_argument("--trace", action="store_true", help="Guardar estados y consignas por paso")
    parser.add_argument("--output", type=Path, default=Path("artifacts/episodes.csv"))
    args = parser.parse_args(argv)
    if (args.episodes is not None and args.episodes < 1) or args.seed < 0:
        parser.error("episodes debe ser positivo y seed no negativo")
    config = EnvConfig.from_json(args.config) if args.config else EnvConfig()
    policy = None
    if args.controller in ("dqn", "double_dqn"):
        if args.checkpoint is None:
            parser.error("El controlador RL requiere --checkpoint (inferencia)")
        from .checkpoints import load_inference
        from .experiments import verify_reference
        from .rl_evaluation import config_from_dict
        verify_reference()
        policy, experiment = load_inference(args.checkpoint)
        if policy.config.algorithm != args.controller:
            parser.error("El algoritmo no coincide con el checkpoint")
        if args.config and json.loads(json.dumps(config.to_dict())) != experiment["env_config"]:
            parser.error("La configuración no coincide con la usada al entrenar")
        config = config_from_dict(experiment["env_config"])
    pid_config = PIDConfig.from_json(args.pid_config) if args.pid_config else PIDConfig()
    pid = PIDController(config, pid_config) if args.controller == "pid" else None
    manifest = load_scenarios(args.scenarios) if args.scenarios else None
    scenarios = manifest["scenarios"] if manifest else [
        {"id": f"seed-{seed}", "condition": "default", "seed": seed, "options": {}}
        for seed in range(args.seed, args.seed + (args.episodes or 3))]
    if args.episodes is not None:
        if args.episodes > len(scenarios):
            parser.error("episodes excede el número de escenarios del manifiesto")
        scenarios = scenarios[:args.episodes]
    if policy:
        from .experiments import ROOT, assert_disjoint, load_manifest
        if manifest and manifest["split"] == "test":
            parser.error("La prueba final está reservada; usa escenarios de validación")
        assert_disjoint(experiment["train_manifest"], {"scenarios": scenarios},
                        load_manifest(ROOT / "configs/pid_test_scenarios.json", "test"))
    env = ThrusterLandingEnv(config, render_mode="human" if args.render or args.controller == "manual" else None)
    rows, traces = [], []
    try:
        for scenario in scenarios:
            seed = scenario["seed"]
            observation, info = env.reset(seed=seed, options=scenario["options"])
            env.action_space.seed(seed)
            if pid:
                pid.reset()
            initial_state = info["state"]
            total, steps = 0.0, 0
            terminated = truncated = False
            if args.trace:
                traces.append({"scenario_id": scenario["id"], "condition": scenario["condition"],
                               "time": 0.0, **info["state"], "action": "", "outcome": "flying"})
            while not (terminated or truncated):
                if env._renderer is not None and env._renderer.close_requested:
                    break
                if policy:
                    action = policy.act(observation, explore=False)
                elif pid:
                    action = pid.act(State(**info["state"]), info["wind_force"])
                else:
                    action = env._renderer.manual_action() if args.controller == "manual" else env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                if args.trace:
                    traces.append({"scenario_id": scenario["id"], "condition": scenario["condition"],
                                   "time": info["time"], **info["state"], "action": int(action),
                                   "outcome": info["outcome"], **(pid.diagnostics if pid else {})})
                total += reward
                steps += 1
            if not (terminated or truncated):
                break  # Una ventana cerrada no cuenta como episodio de evaluación.
            contact = info["contact_metrics"] or {}
            row = {"controller": args.controller, "seed": seed, "steps": steps,
                   "scenario_id": scenario["id"], "condition": scenario["condition"],
                   "split": manifest["split"] if manifest else "unspecified",
                   "outcome": info["outcome"], "success": info["is_success"],
                   "terminated": terminated, "truncated": truncated,
                   "duration_s": info["time"], "return": total,
                   "gas_consumed_Ns": info["gas_consumed"], "wind_force_N": info["wind_force"],
                   "initial_state": json.dumps(initial_state, sort_keys=True),
                   "final_state": json.dumps(info["state"], sort_keys=True),
                   "final_error_x_m": abs(info["state"]["x"]),
                   "final_center_height_m": info["state"]["y"],
                   "contact_error_x_m": contact.get("horizontal_error"),
                   "contact_vx_m_s": contact.get("vx"), "contact_vy_m_s": contact.get("vy"),
                   "contact_theta_rad": contact.get("theta"), "contact_omega_rad_s": contact.get("omega")}
            rows.append(row)
            print(f"seed={seed} outcome={info['outcome']} t={info['time']:.3f}s retorno={total:.3f}")
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    write_csv(args.output.with_suffix(".summary.csv"), summarize(rows))
    if args.trace:
        write_csv(args.output.with_suffix(".traces.csv"), traces)
    metadata = {"versions": versions(), "config": config.to_dict(), "arguments": vars(args),
                "completed_episodes": len(rows), "environment": "ThrusterLanding-v0",
                "pid_config": pid_config.to_dict() if pid else None,
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if policy else None,
                "checkpoint_training_steps": policy.env_steps if policy else None,
                "scenario_manifest": manifest,
                "manifest_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest() if args.scenarios else None,
                "pid_config_sha256": hashlib.sha256(json.dumps(pid_config.to_dict(), sort_keys=True).encode()).hexdigest() if pid else None}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(f"Episodios completos: {len(rows)}; salida: {args.output}")


if __name__ == "__main__":
    main()
