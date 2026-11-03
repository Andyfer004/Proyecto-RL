"""Manifiestos y métricas: contacto ausente se conserva como dato ausente."""
import csv
import json
from pathlib import Path
from statistics import mean


def load_scenarios(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or data.get("split") not in ("validation", "test"):
        raise ValueError("Manifiesto: schema_version=1 y split validation/test requeridos")
    cases = data.get("scenarios", [])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("El manifiesto requiere escenarios con identificadores únicos")
    for case in cases:
        if not isinstance(case["seed"], int) or case["seed"] < 0:
            raise ValueError("Semilla de escenario inválida")
        if not isinstance(case["options"], dict) or not isinstance(case["condition"], str):
            raise ValueError("Escenario inválido")
    return data


def write_csv(path, rows):
    """Unión ordenada de columnas; escribe vacío si no hubo episodios completos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def summarize(rows):
    result = []
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        success = [row for row in group if row["success"]]
        contacts = [row for row in group if row["contact_vy_m_s"] is not None]
        summary = {"condition": condition, "n": len(group), "n_success": len(success),
                   "n_contact": len(contacts), "success_rate": len(success) / len(group)}
        for outcome in ("success", "body_contact", "missed_platform", "invalid_contact", "out_of_bounds", "time_limit"):
            summary["n_" + outcome] = sum(row["outcome"] == outcome for row in group)
        for label, subset in (("all", group), ("success", success)):
            for column in ("gas_consumed_Ns", "duration_s", "final_error_x_m"):
                summary[f"mean_{column}_{label}"] = mean(row[column] for row in subset) if subset else None
        for column in ("contact_error_x_m", "contact_vx_m_s", "contact_vy_m_s", "contact_theta_rad", "contact_omega_rad_s"):
            summary["mean_" + column + "_contacts"] = mean(row[column] for row in contacts) if contacts else None
            summary["mean_" + column + "_success"] = mean(row[column] for row in success) if success else None
        result.append(summary)
    return result
