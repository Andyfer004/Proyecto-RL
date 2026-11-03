`env.json` es la configuración física completa de referencia. `env_wind.json`
cambia el viento horizontal constante por episodio: U(-80, 80) N.
Los archivos JSON permiten omitir parámetros (se aplican los defaults).
Los pesos están bajo `reward`. Los rangos iniciales son [mínimo, máximo].
`pid.json` configura la cascada, saturaciones y pesos de asignación, sin cambiar
física ni criterios de éxito. Los manifiestos `pid_validation_scenarios.json` y
`pid_test_scenarios.json` fijan escenarios separados. El segundo se reservó hasta
congelar los parámetros; no usarlo para retocar el PID y afirmar otra evaluación
independiente. Versiones de esta etapa en `requirements-stage2.txt`.

Etapa 3: `dqn.json` / `double_dqn.json` comparten hiperparámetros salvo algoritmo.
`rl_train_scenarios.json` y `rl_validation_scenarios.json` son independientes de
los manifiestos PID y entre sí. `reference_stage2.json` congela la referencia
física/PID mediante hashes. Versiones CPU verificadas en `requirements-stage3.txt`.
La reanudación admite cambiar solo `total_steps`, que cuenta todos los pasos
incluidos los ya ejecutados. Valores del smoke test, no hiperparámetros óptimos.

Piloto nuevo `pilot_100k_seed42_v1`: `dqn_pilot.json` / `double_dqn_pilot.json`.
Epsilon desciende en 80000 pasos tras warmup; presupuesto total 100000, validación
y guardado cada 10000. `detailed_validation` registra baseline/trazas. El protocolo
previo está en `pilot_100k_protocol.json`; los hashes posteriores en
`reference_pilot_100k.json`. Los smoke configs/resultados se conservan.

Ablación `lr_0p0001_seed42_100k_v1`: `dqn_lr1e4.json` / `double_dqn_lr1e4.json`
cambian únicamente learning_rate a 0.0001 frente al piloto. El protocolo previo
está en `lr1e4_protocol.json`. `reference_before_lr_ablation.json` conserva hashes
de todos los archivos/checkpoints anteriores. La sonda común del replay es solo
para diagnóstico offline: no se entrena con ella. No se cambió clipping ni loss.

Ablación `smoothl1_beta10_lr1e4_seed42_100k_v1`: `dqn_beta10.json` y
`double_dqn_beta10.json` cambian solo `smooth_l1_beta` a 10 respecto al baseline
LR=.0001. Default=1 mantiene compatibilidad con configuraciones anteriores.
Implementación: nn.SmoothL1Loss, sin factor compensatorio ni HuberLoss(delta=10).
Protocolo previo en `beta10_protocol.json`, baseline preservado en
`reference_before_beta10.json`. Sondas A/B/D fijas y replay nuevo C se distinguen.

## Alcance de los manifiestos de auditoría local

`reference*.json` y `curriculum_execution_manifest.json` son **registros locales**
de integridad histórica. Sus rutas bajo `artifacts/` identifican archivos excluidos,
no archivos incluidos en GitHub. Se conservan los bytes de estos manifiestos y
protocolos para no invalidar hashes entre etapas. Algunas referencias congelan
versiones anteriores del código: no deben reinterpretarse como hashes del estado
actual. `reference_stage2.json` sigue siendo la comprobación activa de física/PID;
la ausencia de sus artefactos locales está prevista por `verify_reference`.

Currículo: `dqn_curriculum.json` y `double_dqn_curriculum.json` amplían el máximo
a 300 000 pasos, pero la ejecución real se detuvo a 50 000 por agente según
`curriculum_protocol.json`. Los manifiestos tienen 128 casos de entrenamiento y
32 de validación por nivel, separados de los originales. No se entrenaron los
niveles 2/3 ni se ejecutó prueba final RL. Evidencia pública compacta en
[docs/resultados](../docs/resultados/README.md); los diagnósticos históricos
necesitan checkpoints/replays locales que no se distribuyen.
