"""Ejecuta una sola vez el protocolo congelado de la ablacion, sin entrenar ni reseleccionar.

Usa el mismo evaluador comun que la prueba final (`scope_evaluation.evaluate_method`), que
bloquea consultas al PID durante la inferencia neuronal y verifica que pesos, optimizador,
replay, contadores y RNG no cambien. Cada familia ve exactamente los mismos casos, el mismo
gas inicial, el mismo periodo de decision y los mismos criterios de contacto.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from thruster_landing.agents.pid import PIDConfig
from thruster_landing.envs import EnvConfig
from thruster_landing.evaluation import write_csv
from thruster_landing.scope_evaluation import (evaluate_method, group_statistics, load_method,
                                               sha256, verify_protocol)


def paired_table(rows_by_method, methods):
    """Comparacion emparejada por caso entre cada asistido y su imitacion sola equivalente."""
    by_key = {(m["algorithm"], m["seed"]): m["id"]
              for m in methods if m.get("family") == "imitation_only"}
    pairs = []
    for method in methods:
        if method.get("family") != "imitation_plus_td":
            continue
        partner = by_key.get((method["algorithm"], method["seed"]))
        if partner is None:
            continue
        assisted = {row["scenario_id"]: row for row in rows_by_method[method["id"]]}
        imitation = {row["scenario_id"]: row for row in rows_by_method[partner]}
        shared = sorted(set(assisted) & set(imitation))
        counts = defaultdict(int)
        for case in shared:
            counts[(bool(assisted[case]["success"]), bool(imitation[case]["success"]))] += 1
        both = counts[(True, True)]
        only_assisted = counts[(True, False)]
        only_imitation = counts[(False, True)]
        neither = counts[(False, False)]
        pairs.append(dict(assisted=method["id"], imitation_only=partner, algorithm=method["algorithm"],
                          paired_cases=len(shared), both_land=both,
                          only_assisted_lands=only_assisted, only_imitation_lands=only_imitation,
                          neither_lands=neither,
                          assisted_successes=both + only_assisted,
                          imitation_successes=both + only_imitation,
                          difference=only_assisted - only_imitation,
                          discordant=only_assisted + only_imitation))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path,
                   default=Path("configs/imitation_ablation_v1_protocol.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/imitation_ablation_v1/final"))
    args = p.parse_args()

    protocol = json.loads(args.protocol.read_text())
    verify_protocol(protocol)
    root = Path(protocol["artifacts_root"])
    record = json.loads((root / "protocol_sha256.json").read_text())
    recorded = record.get(args.protocol.as_posix()) or record.get(str(args.protocol))
    assert recorded == sha256(args.protocol), "El protocolo cambio tras congelarse"
    args.output.mkdir(parents=True, exist_ok=False)

    config = EnvConfig.from_json(protocol["env_config"])
    pid_config = PIDConfig.from_json(protocol["pid_config"])
    complement = json.loads(Path(protocol["complement_manifest"]).read_text())
    manifest = dict(split="test", scenarios=complement["scenarios"])

    audit, all_rows, rows_by_method = [], [], {}
    start = time.perf_counter()
    for method in protocol["methods"]:
        verify_protocol(protocol)
        agent = load_method(method, config, pid_config)
        rows, traces, checks = evaluate_method(agent, method, config, manifest)
        write_csv(args.output / (method["id"] + ".episodes.csv"), rows)
        write_csv(args.output / (method["id"] + ".traces.csv"), traces)
        audit.append(checks)
        rows_by_method[method["id"]] = rows
        all_rows.extend(rows)
        print(method["id"], "exitos", sum(r["success"] for r in rows), "/", len(rows), flush=True)
    verify_protocol(protocol)

    write_csv(args.output / "episodes.csv", all_rows)
    write_csv(args.output / "summary.csv", group_statistics(all_rows))
    pairs = paired_table(rows_by_method, protocol["methods"])
    write_csv(args.output / "paired_families.csv", pairs)
    totals = [dict(method=method["id"], family=method.get("family", "pid"),
                   label=method["label"], episodes=len(rows_by_method[method["id"]]),
                   successes=sum(r["success"] for r in rows_by_method[method["id"]]))
              for method in protocol["methods"]]
    write_csv(args.output / "totals.csv", totals)
    (args.output / "execution.json").write_text(json.dumps(dict(
        protocol_sha256=sha256(args.protocol),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter() - start, episodes=len(all_rows),
        environment_steps=sum(r["steps"] for r in all_rows), training_updates=0,
        checkpoints_changed=False, methods=audit), indent=2) + "\n")

    print()
    for row in totals:
        print("%-26s %-20s %3d/%d" % (row["method"], row["family"], row["successes"], row["episodes"]))
    print()
    for pair in pairs:
        print("%s vs %s: ambos %d, solo asistido %d, solo imitacion %d, ninguno %d"
              % (pair["assisted"], pair["imitation_only"], pair["both_land"],
                 pair["only_assisted_lands"], pair["only_imitation_lands"], pair["neither_lands"]))


if __name__ == "__main__":
    main()
