# Matriz de trazabilidad del alcance

La propuesta original en `propuesta.md` se conserva como contrato de objetivos. Esta revisión contrasta archivos y resultados reales. Fecha de revisión: 2026-11-04. Los resultados históricos se conservan; no equivalen por sí solos a una comparación común.

| Compromiso | Código/configuración comprobados | Evidencia existente verificada | Cierre verificado y pendiente remanente |
|---|---|---|---|
| Simulador físico propio 2D | `envs/physics.py`, `envs/config.py`, `configs/env.json` | ODE RK4 independiente y caída libre analítica en `artifacts/recovery_v1/audit/report.json`; pruebas físicas existentes | Cumplido técnicamente: suite118/checker válidos; hashes físicos preservados. Sin 3D |
| Empuje principal y propulsores laterales | `physics.thrusters`, `wrench`, parámetros 1800/600 N y brazos de 0,6 m | Ocho acciones contrastadas independientemente; signos/momentos documentados en `diseno.md` | Cumplido: parámetros y catálogo congelados e idénticos durante2328 episodios |
| Observaciones posición, velocidades, orientación, omega y gas | Observación de nueve valores, incluye además viento; `recovery_agent.landing_features` | Pruebas de escalas y transformación; candidatas con la misma información física | Resuelto: `scope_evaluation.decode_observation`, prueba específica y evaluador común; PID solo recibe el Box decodificado |
| Combinaciones discretas | `physics.ACTIONS`, espacio Discrete(8) | Pruebas de acciones y asignación PID; inferencia neuronal argmax | Resuelto: evaluador común verifica acción válida, mismo `EnvConfig`,0,05 s,18000 Ns |
| DQN y Double DQN comparados con PID | `agents/dqn.py`, `agents/pid.py`; `recovery_agent.py` para variante asistida | Baselines 100k desde cero; checkpoints y trazas asistidas funcionales, pérdidas TD/margen registradas | Comparación común ejecutada; baselines0 éxitos, políticas asistidas funcionales. Ampliación adicional documentada; beneficio TD no demostrado |
| Condiciones iniciales variadas y viento | `pid_test_scenarios.json`: 100 casos, cuatro grupos de 25 | Altura, desplazamiento, inclinación, velocidades y viento existentes; PID ya evaluado históricamente | Resuelto:100 reservados intactos+288 complementarios; grupos y resultados completos en `evaluacion_alcance.md` |
| Éxito, velocidad al contacto, precisión y consumo | `rl_evaluation.episode_metrics`, contacto y terminaciones del entorno | CSV históricos distinguen métricas de contacto y estados finales | Resuelto: `summary.csv`, `failures.csv`, `paired_vs_pid.csv` y `paired_wind.csv`;105 episodios DQN sin contacto con métricas ausentes |
| Animación | `envs/rendering.py`, render `human`/`rgb_array` | Pruebas RGB bajo SDL dummy; no evidencia de teclado físico verificado | GIF verificados de PID/DQN/Double asistidos, replay guardado para cualquier método; teclado/ventana real sin comprobar |
| Reporte experimental | `docs/resultados*`, `docs/recuperacion_agente.md` y artefactos | Referencias preservadas; informe asistido con límites explícitos | Resuelto: `evaluacion_alcance.md`, protocolo SHA-256,2328 resultados, figuras y CSV por grupo/resultado |
| Inspiración Tesla Roadster y límites académicos | `docs/diseno.md`, propuesta de cuerpo rígido plano | Parámetros académicos propios, sin modelo comercial ni aerodinámica detallada | Resuelto en `diseno.md`, seguimiento de propuesta e informe; parámetros académicos propios |

## Metodología inspeccionada antes de la prueba

Las políticas funcionales combinan imitación y TD, por lo que se denominan **DQN y Double DQN asistidos por demostraciones PID**. La ampliación respecto a la metodología inicial está implementada como componente adicional y se documenta de forma diferenciada. La comparación DQN/Double DQN desde cero permanece como baseline de la propuesta original.

La inspección de `classification_update`, `learn` y `scripts/train_recovery.py` muestra: primero 10000 actualizaciones supervisadas de entropía cruzada, luego 10000 actualizaciones con TD de uno y diez pasos más margen supervisado. Online se mezcla replay propio con demostraciones permanentes y se siguen aplicando TD y margen. Los checkpoints seleccionados contienen contadores de 24969 (DQN42) o 27469 (Double42/DQN43) actualizaciones: respectivamente 14969 y 17469 posteriores al calentamiento de imitación. Los CSV de métricas registran pérdidas TD y multietapa no nulas durante esas fases; el evaluador común exportará la auditoría detallada.

El margen supervisado y la entropía cruzada imitan acciones PID. TD usa recompensas reales y bootstrap con red objetivo; Double DQN separa selección y evaluación. Que haya actualizaciones TD no demuestra su beneficio causal. Los checkpoints solo de clasificación también aterrizaban en desarrollo; no hay una comparación controlada que permita afirmar que TD mejora sobre imitación sola.

Durante inferencia `act(..., explore=False)` aplica una transformación fija a la observación y el argmax de la MLP; no consulta PID, demostraciones ni información futura. El evaluador verificará invariancia de pesos, optimizador, replay y RNG, y prohibirá consultas docentes durante episodios neuronales.

## Cierre de pendientes

El [protocolo final congelado](../configs/scope_final_v1_protocol.json) se ejecutó sin entrenamiento:2328 episodios,242465 pasos físicos, seis métodos sobre los mismos388 casos. Resultados y límites en [evaluacion_alcance.md](evaluacion_alcance.md). `artifacts/scope_alignment_v1/final/execution.json` confirma cero updates/exploración y red sin consultas PID; `independent_checks.json` confirma clasificación geométrica independiente de todos los casos.

La comprobación manual de teclado/ventana permanece pendiente de verificación local. El aprendizaje funcional de DQN/Double DQN desde cero y el aporte causal de TD continúan sin demostrarse; no se marcan como cumplidos. Los demás pendientes técnicos reversibles de esta matriz se resolvieron sin ampliar el alcance.
