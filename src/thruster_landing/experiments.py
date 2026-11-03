"""Integridad de la referencia y separación explícita de escenarios."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference():
    reference = json.loads((ROOT / "configs/reference_stage2.json").read_text())
    missing = []
    for path, expected in reference["sha256"].items():
        target = ROOT / path
        if not target.exists() and path.startswith("artifacts/"):
            missing.append(path)  # Los artefactos originales no se versionan.
        elif not target.exists() or digest(target) != expected:
            raise ValueError(f"Referencia experimental modificada: {path}")
    return {**reference, "unavailable_artifacts": missing}


def load_manifest(path, split):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or data.get("split") != split:
        raise ValueError(f"Se requiere manifiesto de {split}; prueba final excluida")
    cases = data.get("scenarios", [])
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Escenarios vacíos o duplicados")
    for case in cases:
        if type(case["seed"]) is not int or case["seed"] < 0 or not isinstance(case["options"], dict):
            raise ValueError("Escenario inválido")
    return data


def _fingerprints(manifest):
    return ({c["seed"] for c in manifest["scenarios"]},
            {json.dumps(c["options"], sort_keys=True) for c in manifest["scenarios"]})


def assert_disjoint(*manifests):
    for i, first in enumerate(manifests):
        seeds, states = _fingerprints(first)
        for second in manifests[i + 1:]:
            other_seeds, other_states = _fingerprints(second)
            if seeds & other_seeds or states & other_states:
                raise ValueError("Solapamiento entre conjuntos experimentales")


def prepare_experiment(env_config, train_path, validation_path):
    reference = verify_reference()
    train = load_manifest(train_path, "train")
    validation = load_manifest(validation_path, "validation")
    # Leer el manifiesto de prueba solo para excluir identidades; nunca ejecutar casos.
    test = load_manifest(ROOT / "configs/pid_test_scenarios.json", "test")
    pid_validation = load_manifest(ROOT / "configs/pid_validation_scenarios.json", "validation")
    assert_disjoint(train, validation, test, pid_validation)
    frozen_config = json.loads((ROOT / "configs/env.json").read_text())
    if json.loads(json.dumps(env_config.to_dict())) != frozen_config:
        raise ValueError("Etapa 3 requiere configuración física/recompensa congelada")
    return {"environment": "ThrusterLanding-v0", "package_version": "0.1.0", "env_config": frozen_config,
            "reference": reference, "train_manifest": train, "validation_manifest": validation,
            "train_sha256": digest(train_path), "validation_sha256": digest(validation_path)}
