"""Entrenador CPU reproducible con validación independiente y reanudación completa."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .agents.dqn import DQNAgent, DQNConfig
from .checkpoints import (plain, read_checkpoint, restore_environment, restore_global_rng,
                          save_inference, save_resume)
from .envs import EnvConfig, ThrusterLandingEnv
from .evaluation import write_csv
from .experiments import prepare_experiment
from .play import versions
from .rl_evaluation import evaluate, episode_metrics, export_evaluation


def validation_score(rows):
    """Más éxitos; empate: menor error |x| medio entre éxitos, nunca impactos.

    Sin éxitos no hay error de aterrizaje: rango secundario -inf. Un empate
    completo conserva el checkpoint anterior (el primero evaluado).
    """
    successes = [row for row in rows if row["success"]]
    error = float(np.mean([row["contact_error_x_m"] for row in successes])) if successes else None
    return {"success_rate": len(successes) / len(rows), "landing_error_m": error}


def validation_rank(score):
    return (score["success_rate"], -score["landing_error_m"] if score["landing_error_m"] is not None else -float("inf"))


def train(config, output, env_config=None, train_path="configs/rl_train_scenarios.json",
          validation_path="configs/rl_validation_scenarios.json", resume=None, curriculum=None, stop_steps=None, exploration=None, run_protocol=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Usa un directorio de salida nuevo para conservar ejecuciones")
    experiment = prepare_experiment(env_config or EnvConfig(), train_path, validation_path)
    if curriculum is not None:
        from .curriculum import choose_scenario, annotate_validation, checkpoint_scores
        experiment["curriculum"] = curriculum
    if exploration is not None:
        from .agents.exploration import action_probabilities, WeightedExplorationAgent
        action_probabilities(exploration)
        experiment["exploration_protocol"] = exploration
    if run_protocol is not None:
        experiment["run_protocol"] = run_protocol
    limit = config.total_steps if stop_steps is None else min(stop_steps, config.total_steps)
    if limit <= 0:
        raise ValueError("Presupuesto de segmento inválido")
    torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    agent = (DQNAgent(config) if exploration is None else WeightedExplorationAgent(config, exploration))
    env = ThrusterLandingEnv(env_config or EnvConfig())
    scenario_rng = np.random.default_rng(config.seed + 100)
    runner = {"episodes": [], "metrics": [], "validation": [], "pending_metrics": [],
              "episode_steps": 0, "episode_return": 0.0, "episode_components": {},
              "observation": None, "scenario": None, "initial": None, "scenario_rng": None,
              "best_validation": None}
    if resume:
        payload = read_checkpoint(resume, "resume")
        if payload["experiment"] != experiment:
            raise ValueError("Entorno o conjuntos distintos al checkpoint")
        agent.load_state_dict(payload["agent"])
        if limit <= agent.env_steps:
            raise ValueError("total_steps debe superar el paso guardado")
        restore_environment(env, payload["environment"])
        runner = payload["runner"]
        runner.setdefault("best_validation", None)
        scenario_rng.bit_generator.state = runner["scenario_rng"]
        restore_global_rng(payload["global_rng"])
    output.mkdir(parents=True, exist_ok=False) if not output.exists() else None
    telemetry = None
    if run_protocol is not None and run_protocol.get("decision_logging"):
        from .training_telemetry import TrainingTelemetry
        telemetry = TrainingTelemetry(runner, output)
    exposure = None
    if run_protocol is not None and run_protocol.get("replay_capacity_audit"):
        from .replay_exposure import ReplayExposure
        exposure = ReplayExposure(agent, runner, output)
        agent.learning_observer = exposure
    start_steps = agent.env_steps
    clock_start = time.perf_counter()
    timings = {"environment_seconds": 0.0, "learning_seconds": 0.0,
               "validation_seconds": 0.0, "checkpoint_seconds": 0.0}

    def save(path):
        runner["scenario_rng"] = scenario_rng.bit_generator.state
        if telemetry is not None: telemetry.stream.flush()
        if exposure is not None: exposure.checkpoint()
        save_resume(path, agent, env, runner, experiment)

    try:
        if config.detailed_validation and not resume:
            start = time.perf_counter()
            rows, traces = evaluate(agent, env.config, experiment["validation_manifest"], trace=True)
            if curriculum is not None:
                annotate_validation(rows, experiment["validation_manifest"])
            for row in rows:
                row["env_steps"] = 0
            runner["validation"].extend(rows)
            export_evaluation(output / "initial_validation.csv", rows, traces)
            save_inference(output / "initial.policy.pt", agent, experiment)
            timings["validation_seconds"] += time.perf_counter() - start
        while agent.env_steps < limit:
            if runner["observation"] is None:
                cases = experiment["train_manifest"]["scenarios"]
                scenario = (choose_scenario(cases, scenario_rng, agent.env_steps, curriculum) if curriculum is not None
                            else cases[int(scenario_rng.integers(len(cases)))])
                observation, info = env.reset(seed=scenario["seed"], options=scenario["options"])
                runner.update(observation=observation.tolist(), scenario=scenario, initial=info["state"],
                              episode_steps=0, episode_return=0.0, episode_components={}, episode_discounted_return=0.0)
                if telemetry is not None: telemetry.begin_episode(info)
            observation = np.array(runner["observation"], dtype=np.float32)
            epsilon_used = agent.epsilon
            action = agent.act(observation, explore=True)
            selection_mode = getattr(agent, "last_selection_mode", None)
            start = time.perf_counter()
            next_observation, reward, terminated, truncated, info = env.step(action)
            timings["environment_seconds"] += time.perf_counter() - start
            agent.replay.add(observation, action, reward, next_observation, terminated, truncated,
                             source_level=runner["scenario"].get("source_level", 0))
            agent.env_steps += 1
            runner["episode_steps"] += 1
            runner["episode_return"] += reward
            if curriculum is not None or exploration is not None or run_protocol is not None:
                runner["episode_discounted_return"] = runner.get("episode_discounted_return", 0.0) + config.gamma ** (runner["episode_steps"] - 1) * reward
            for key, value in info["reward_components"].items():
                runner["episode_components"][key] = runner["episode_components"].get(key, 0.0) + float(value)
            runner["observation"] = next_observation.tolist()
            if (agent.env_steps >= config.learning_starts and len(agent.replay) >= config.batch_size
                    and agent.env_steps % config.train_frequency == 0):
                start = time.perf_counter()
                runner["pending_metrics"].append(agent.learn())
                timings["learning_seconds"] += time.perf_counter() - start
            if agent.env_steps >= config.learning_starts and agent.env_steps % config.target_update_frequency == 0:
                agent.sync_target()
            if telemetry is not None:
                telemetry.record(agent,runner,action,selection_mode,epsilon_used,reward,info,env)
            if terminated or truncated:
                row = episode_metrics(info, runner["initial"], runner["episode_return"], runner["episode_steps"],
                                      runner["scenario"], config.algorithm, "train")
                row.update(env_steps=agent.env_steps, episode=len(runner["episodes"]) + 1,
                           terminated=terminated, truncated=truncated)
                row.update({"reward_" + key: value for key, value in runner["episode_components"].items()})
                if telemetry is not None:
                    from .training_telemetry import outcome_details
                    row.update(**telemetry.data["episode"],**{k:v for k,v in outcome_details(info,env).items() if k!="gas_exhausted"},terminal_selection_mode=selection_mode)
                if curriculum is not None or exploration is not None or run_protocol is not None:
                    row.update(source_level=runner["scenario"].get("source_level", 0), discounted_return=runner["episode_discounted_return"])
                runner["episodes"].append(row)
                runner["observation"] = None
            if agent.env_steps % config.log_frequency == 0 or agent.env_steps == limit:
                pending = runner["pending_metrics"]
                row = {"env_steps": agent.env_steps, "updates": agent.updates, "epsilon": agent.epsilon,
                       "target_syncs": agent.target_syncs, "replay_size": len(agent.replay),
                       "updates_in_interval": len(pending)}
                if pending:
                    row.update({key: float(np.mean([m[key] for m in pending])) for key in pending[0]})
                if telemetry is not None: row.update(telemetry.metrics(agent))
                runner["metrics"].append(row)
                runner["pending_metrics"] = []
            if agent.env_steps % config.validation_frequency == 0 or (config.detailed_validation and agent.env_steps == limit):
                start = time.perf_counter()
                rows, traces = evaluate(agent, env.config, experiment["validation_manifest"], trace=config.detailed_validation)
                if curriculum is not None:
                    annotate_validation(rows, experiment["validation_manifest"])
                for row in rows:
                    row["env_steps"] = agent.env_steps
                runner["validation"].extend(rows)
                timings["validation_seconds"] += time.perf_counter() - start
                start = time.perf_counter()
                scored_rows = ([r for r in rows if r["source_level"] == run_protocol["validation_selection_level"]]
                               if run_protocol is not None and "validation_selection_level" in run_protocol else
                               [r for r in rows if r["source_level"] == 0] if curriculum is not None else rows)
                score = validation_score(scored_rows)
                best = runner["best_validation"]
                if best is None or validation_rank(score) > validation_rank(best):
                    runner["best_validation"] = {**score, "env_steps": agent.env_steps}
                    save(output / "best.resume.pt")
                    save_inference(output / "best.policy.pt", agent, experiment)
                    export_evaluation(output / "best_validation.csv", rows, traces if config.detailed_validation else None)
                if curriculum is not None:
                    for level, level_score in checkpoint_scores(rows).items():
                        previous = runner.setdefault("best_by_level", {}).get(str(level))
                        if previous is None or validation_rank(level_score) > validation_rank(previous):
                            runner["best_by_level"][str(level)] = {**level_score, "env_steps": agent.env_steps}
                            save_inference(output / f"best_level{level}.policy.pt", agent, experiment)
                    export_evaluation(output / f"validation_{agent.env_steps:08d}.csv", rows, traces)
                if run_protocol is not None and run_protocol.get("export_all_validations"):
                    export_evaluation(output / f"validation_{agent.env_steps:08d}.csv", rows, traces)
                if config.detailed_validation and agent.env_steps == limit:
                    export_evaluation(output / "final_validation.csv", rows, traces)
                timings["checkpoint_seconds"] += time.perf_counter() - start
                successes = sum(row["success"] for row in rows)
                print(f"{config.algorithm} pasos={agent.env_steps} updates={agent.updates} "
                      f"epsilon={agent.epsilon:.3f} validación={successes}/{len(rows)}", flush=True)
            if agent.env_steps % config.checkpoint_frequency == 0:
                start = time.perf_counter()
                save(output / f"step_{agent.env_steps:08d}.resume.pt")
                save_inference(output / f"step_{agent.env_steps:08d}.policy.pt", agent, experiment)
                timings["checkpoint_seconds"] += time.perf_counter() - start
                write_csv(output / "episodes.csv", runner["episodes"])
                write_csv(output / "updates.csv", runner["metrics"])
                write_csv(output / "validation.csv", runner["validation"])
        start = time.perf_counter()
        save(output / "last.resume.pt")
        save_inference(output / "last.policy.pt", agent, experiment)
        timings["checkpoint_seconds"] += time.perf_counter() - start
    finally:
        if exposure is not None: exposure.close(output)
        if telemetry is not None: telemetry.close()
        env.close()
    write_csv(output / "episodes.csv", runner["episodes"])
    write_csv(output / "updates.csv", runner["metrics"])
    write_csv(output / "validation.csv", runner["validation"])
    elapsed = time.perf_counter() - clock_start
    metadata = {"config": config.to_dict(), "experiment": experiment, "versions": versions(),
                "resumed_from": str(resume) if resume else None, "start_steps": start_steps, "stop_steps": limit,
                "env_steps": agent.env_steps, "updates": agent.updates, "wall_seconds": elapsed,
                "segment_steps_per_second": (agent.env_steps - start_steps) / elapsed, **timings,
                "completed_training_episodes": len(runner["episodes"]),
                "best_validation": runner["best_validation"],
                "best_checkpoint_rule": "max_success_rate_then_min_mean_abs_x_successes_then_earliest; initial_baseline_excluded",
                "decision_telemetry": runner.get("decision_telemetry"),
                "partial_episode_steps": runner["episode_steps"] if runner["observation"] is not None else 0}
    (output / "metadata.json").write_text(json.dumps(plain(metadata), indent=2) + "\n")
    print(f"{config.algorithm}: {agent.env_steps} pasos; {elapsed:.2f} s; "
          f"{metadata['segment_steps_per_second']:.1f} pasos/s (incluye validación y guardado)", flush=True)
    return agent, runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-protocol", type=Path, help="Registro, validación y selección predefinidos del experimento")
    parser.add_argument("--exploration", type=Path, help="Protocolo opcional de muestreo de acciones durante epsilon; recompensa intacta")
    parser.add_argument("--curriculum", type=Path, help="Protocolo fijo de distribuciones de reinicio")
    parser.add_argument("--stop-steps", type=int, help="Pausa fiel en este paso global, incluso a mitad de episodio")
    parser.add_argument("--total-steps", type=int, help="Nuevo presupuesto TOTAL, no pasos adicionales")
    parser.add_argument("--train-scenarios", type=Path, default=Path("configs/rl_train_scenarios.json"))
    parser.add_argument("--validation-scenarios", type=Path, default=Path("configs/rl_validation_scenarios.json"))
    args = parser.parse_args()
    config = DQNConfig.from_json(args.config)
    if args.total_steps is not None:
        config = replace(config, total_steps=args.total_steps)
    train(config, args.output, train_path=args.train_scenarios, validation_path=args.validation_scenarios, resume=args.resume,
          curriculum=json.loads(args.curriculum.read_text()) if args.curriculum else None, stop_steps=args.stop_steps,
          exploration=json.loads(args.exploration.read_text()) if args.exploration else None,
          run_protocol=json.loads(args.run_protocol.read_text()) if args.run_protocol else None)


if __name__ == "__main__":
    main()
