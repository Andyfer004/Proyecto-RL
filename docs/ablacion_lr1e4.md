# Ablación LR=0.0001 tras auditoría de DQN / Double DQN

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

Experimento **`lr_0p0001_seed42_100k_v1`**, comparado con
`pilot_100k_seed42_v1`. Se ejecutaron únicamente 100000 pasos por algoritmo desde
cero, semilla 42. No hubo campaña de semillas, evaluación final, imitación ni
replay priorizado. Se preservaron resultados, configuraciones y **todos** los
checkpoints anteriores; sus hashes están en `reference_before_lr_ablation.json`.
Física, recompensa, PID, ocho acciones, umbrales y manifiestos no cambiaron.

## 1. Auditoría previa: no se encontró un defecto de aprendizaje

Se inspeccionaron `agents/dqn.py`, `replay.py`, `train.py`, `rl_evaluation.py` y el
entorno. Las cuatro pruebas nuevas en `tests/test_transition_audit.py` cubren
huecos de integración, en lugar de duplicar los tests existentes de fórmulas:

- **Acción y Q actualizado:** se ejecutó el método real `learn` con una red lineal
  de valores conocidos, acciones almacenadas [7,0,3], Q seleccionados [7,0,3],
  recompensas [10,1,-2], terminated [True,False,True] y gamma=0.5. Los targets fueron
  [10,4.5,-2]. Se inspeccionaron gradientes de los sesgos: acciones 7 y 0 recibieron
  -1/3, acción 3 recibió +1/3 y las otras cero. Loss Huber=11/3. Esto comprueba
  `gather` y que se actualiza la acción almacenada, no el argmax actual.
- **Formas:** las observaciones son [B,9], las salidas [B,8], acciones [B] int64,
  recompensas/dones [B], Q(s,a)/targets [B]. La prueba anterior intercepta la loss
  real y exige formas [3] para todos sus vectores. Las pruebas de integración con
  B=1 también pasan. No se observó broadcasting [B,B].
- **Transición antes de reset:** fixtures sintéticas fuerzan seis contactos y,
  en otra prueba, seis truncamientos. Cada `obs`, acción y `next_obs` del replay
  coincide exactamente con las salidas de reset/step del episodio correcto;
  `next_obs` es distinto del siguiente reset. Son fixtures de prueba, no cambios
  de escenarios de entrenamiento del experimento.
- **Red objetivo:** igualdad exacta con la principal al crearla; gradientes None y
  requires_grad=False; copias duras verificadas en los pasos 2,4,6 de una fixture
  con frecuencia 2. En producción el código conserva frecuencia 500 pasos de
  entorno desde warmup=1000, con 199 sincronizaciones en cada piloto de 100000.
- **Normalización:** la observación float32 de dimensión 9 de un reset de
  entrenamiento coincide exactamente con la recibida por el evaluador greedy
  para el mismo caso 016. Ningún agente aplica otra normalización ni estadísticas
  móviles. Los tests previos verifican las escalas SI, seno/coseno y gas.

Los tests existentes verifican targets DQN/Double DQN con números diferentes,
bootstrap solo para truncamientos artificiales, observación final real, red
objetivo sin gradientes, actualización y reanudación. **81 pruebas pasaron**;
checker Gymnasium válido con sus dos advertencias conocidas del Box infinito.
El resumen numérico de auditoría está en `docs/lr1e4_audit.json`.

No hubo corrección de baseline que obligara a detener la ablación. El único
cambio de código de aprendizaje fue **telemetría**: leer la norma de gradientes
tras el clipping existente, sin modificar gradientes, loss, parámetros ni Adam.
La fuente previa del agente se conservó en `artifacts/lr_audit_reference/`.

## 2. Protocolo de la ablación

`configs/lr1e4_protocol.json` se fijó antes de entrenar, junto con la muestra de
replay. `dqn_lr1e4.json` y `double_dqn_lr1e4.json` difieren de sus respectivos
pilotos previos **solo** en `learning_rate`: 0.0003 → **0.0001**.

Todo lo demás: gamma=0.99, buffer=50000, batch=64, warmup=1000, una actualización
cada 4 pasos, copia objetivo cada 500, MLP 128/128 ReLU, Huber beta=1, clipping de
norma=10, epsilon 1→0.1 en 80000 pasos tras warmup, presupuesto=100000, CPU y un
hilo Torch. Desde cero, mismas redes iniciales: sus tensores coinciden exactamente
con las redes iniciales del piloto anterior. Cada agente realizó 24751
actualizaciones. No se continuó ningún checkpoint.

Mismos 128 casos de entrenamiento y mismos 32 de validación. Baseline inicial y
validación cada 10000 pasos, sin explorar ni modificar pesos/replay. Cada ejecución
produjo 352 filas (32 casos repetidos en 11 puntos), que no son muestras
independientes de un experimento de múltiples semillas. La prueba final solo se
leyó para comprobar exclusión de identidades; no se ejecutó.

Directorios nuevos: `artifacts/dqn_lr1e4_100k/` y
`artifacts/double_dqn_lr1e4_100k/`. Se conserva la selección de checkpoint del
piloto: éxito, después menor error horizontal entre éxitos y después el primero.
Como no hubo éxitos, `best` vuelve a ser el paso 10000 por empate, sin afirmar que
sea un controlador de aterrizaje útil. `last` es el paso 100000.

## 3. Diagnóstico de Q: definiciones y muestra fija

El contraste con Monte Carlo se llama **brecha frente al retorno de la política
greedy**: Q de su primera acción menos el retorno descontado observado siguiendo
esa política. **No es error exacto respecto a Q óptimo**: la política actual no
necesariamente realiza las acciones futuras óptimas. Tampoco se usa retorno sin
descuento para ese contraste. Se reconstruyeron las recompensas de cada transición
de las trazas y se verificó que sus sumas igualan los CSV (tolerancia 1e-6).
Se conservaron los informes previos sin reescribir su historia; esta interpretación
es la utilizada en las tablas, figuras y CSV comparativos nuevos.

Se congeló una muestra común **antes** de entrenar: 64 transiciones terminales y
64 no terminales del replay DQN previo del paso 10000. Selección uniforme sin
reemplazo, semilla de sonda 20261102, índices e identificadores SHA256 fijos.
`artifacts/lr_audit_reference/fixed_probe.pt` conserva obs, acción, recompensa,
next_obs, terminated y truncated; su JSON conserva procedencia y hashes.

Es una sonda offline, no se añade a ningún replay nuevo ni se usa para actualizar
pesos. Se aplicó la **misma muestra de 128 transiciones a los cuatro modelos y
sus 11 checkpoints**, incluidos los iniciales: 5632 registros individuales.
Es off-policy y procede de un piloto: no representa toda la distribución de
estados ni está diseñada para estimar rendimiento de aterrizaje.

Por transición se registra Q(s,a), recompensa, target, bootstrap y
**TD = target − Q(s,a)**. En terminales se añade el **error exacto frente a la
recompensa = Q(s,a) − r**, con signo contrario al TD. Bootstrap terminal=0 y
target terminal=r exactamente en todos los casos. En no terminales el target
incluye bootstrap estimado: un TD pequeño no demuestra que el valor sea correcto.

`fixed_transitions.csv` conserva todos los valores individuales; el resumen
separa grupos, modelo y checkpoint, con n, media, mínimo, p05/p25/p50/p75/p95 y
máximo **con signos**. No se cambia de transiciones entre checkpoints. No se
registran supuestos retornos óptimos desconocidos.

Las redes objetivo provienen de checkpoints completos. Los checkpoints periódicos
son posteriores a las copias objetivo (10000 es múltiplo de 500): muestran el
estado sincronizado, no toda la evolución dentro de cada intervalo de 500 pasos.
Por eso este análisis no identifica por sí solo el efecto del desfase objetivo.

## 4. Resultados de control y retornos

| Métrica final | DQN LR .0003 | DQN LR .0001 | Double DQN LR .0003 | Double DQN LR .0001 |
|---|---:|---:|---:|---:|
| Éxitos entrenamiento / episodios completos | 0/2855 | 0/2685 | 0/2642 | 0/2709 |
| Éxitos en cada validación | 0/32 | 0/32 | 0/32 | 0/32 |
| Contactos en validación final | 0 | 0 | 0 | 0 |
| Salidas finales por techo | 32 | 32 | 32 | 32 |
| Duración media final, todos (s) | 1.16475 | 1.21585 | 1.29519 | 1.17470 |
| Impulso medio final, todos (N s) | 3494.25 | 3317.28 | 3195.46 | 3494.11 |
| Retorno sin descuento, final | -101.63790 | -102.19349 | -102.05825 | -101.92482 |
| Retorno descontado observado, final | -80.98922 | -80.60806 | -79.21983 | -81.07681 |
| Q greedy firmado, estados iniciales | +119.78369 | +83.86561 | +221.98585 | +428.65888 |
| Brecha frente al retorno greedy | +200.77291 | +164.47366 | +301.20568 | +509.73569 |

Ninguna métrica final se presenta como aterrizaje: **no hubo contacto**. Los
campos `contact_*` quedaron ausentes. Los estados finales fueron ascendentes,
a altura media del centro 24.073 m y 23.991 m para los agentes nuevos, con vy
media +19.958 y +22.275 m/s. La tabla de contactos por todos los puntos intermedios
está en `validation_progress.csv` y los criterios por caso en `contact_criteria.csv`.

Fallos de entrenamiento con LR .0001:

| Causa | DQN | Double DQN |
|---|---:|---:|
| out_of_bounds | 1753 | 1905 |
| body_contact | 612 | 538 |
| missed_platform | 270 | 224 |
| invalid_contact | 50 | 42 |
| time_limit / success | 0 / 0 | 0 / 0 |

Al parar quedaron episodios parciales de 11 y 19 pasos, conservados para reanudar
sin contarlos como completos. Componentes de recompensa y consumo por episodio
están en los registros y resúmenes de análisis.

## 5. Q y TD sobre las mismas transiciones

Medias firmadas al paso 100000; recompensa media terminal de la sonda=-100.07249.
Target terminal es esa recompensa para todos los modelos, no un bootstrap.

| Modelo | Q terminal | Error terminal Q−r | p05 error | p95 error |
|---|---:|---:|---:|---:|
| DQN LR .0003 | +23.14568 | +123.21816 | +42.73166 | +207.79727 |
| DQN LR .0001 | +7.12949 | +107.20198 | +19.24032 | +208.69510 |
| Double DQN LR .0003 | +159.76442 | +259.83690 | +123.11482 | +353.96552 |
| Double DQN LR .0001 | +291.50249 | +391.57497 | +188.19784 | +647.05032 |

El TD terminal tiene signo negativo y media igual al negativo de la columna de
error. La recompensa terminal es conocida, así que aquí sí puede identificarse
un error real. En DQN nuevo el error medio bajó frente al piloto, pero aumentó
de +82.803 al paso 50000 a +107.202 al 100000; el p95 final sigue sobre +208.
En Double DQN nuevo pasó de +109.297 a +391.575 en ese intervalo. No se observa
estabilización común por reducir LR.

| Modelo | Q no terminal | Bootstrap medio | TD medio target−Q | p05 TD | p95 TD |
|---|---:|---:|---:|---:|---:|
| DQN LR .0003 | +79.88613 | +79.38207 | -0.57902 | -4.61845 | +2.70421 |
| DQN LR .0001 | +50.90312 | +51.27373 | +0.29566 | -4.05550 | +5.55438 |
| Double DQN LR .0003 | +196.45000 | +196.36530 | -0.15966 | -3.90711 | +4.73398 |
| Double DQN LR .0001 | +341.69577 | +349.58962 | +7.81890 | -2.62528 | +31.90280 |

En la sonda, recompensa no terminal media≈-0.07495; target=recompensa+bootstrap.
La media firmada de TD puede ocultar cancelaciones. Q y bootstrap pueden crecer
juntos mientras el TD no terminal permanece relativamente pequeño. El sesgo
terminal no puede explicarse por haber añadido bootstrap a terminales: se comprobó
que es exactamente cero.

## 6. Gradientes antes y después del clipping

La norma posterior no estaba registrada durante el piloto original. **No se
reconstruye una media histórica posterior usando la media anterior**: sería
incorrecto por la no linealidad del recorte. Esa celda queda ausente en el CSV
comparativo. En las nuevas ejecuciones se midieron ambas normas en cada actualización
y se promediaron por intervalo.

| Modelo | Norma anterior, media último intervalo real | Norma posterior, misma media |
|---|---:|---:|
| DQN LR .0003 | 14.76114 | no registrada |
| DQN LR .0001 | 8.40464 | 7.86442 |
| Double DQN LR .0003 | 18.37502 | no registrada |
| Double DQN LR .0001 | 31.66432 | 10.00000 |

Para comparar ambos experimentos se midió también un backward **offline sobre
el mismo batch fijo de 128** por checkpoint, sin optimizer.step ni cambio de pesos.
Este batch balanceado no es el minibatch aleatorio de 64 del entrenamiento, y sus
normas no se presentan como reconstrucciones de actualizaciones pasadas:

| Modelo, checkpoint final | Norma antes, sonda fija | Norma después, sonda fija |
|---|---:|---:|
| DQN LR .0003 | 34.754 | 10.000 |
| DQN LR .0001 | 19.577 | 10.000 |
| Double DQN LR .0003 | 44.441 | 10.000 |
| Double DQN LR .0001 | 25.640 | 10.000 |

El clipping funcionó; no garantiza buen control ni valores correctos. La norma
con batch fijo incluso puede bajar mientras el error terminal crece, porque las
derivadas de Huber se saturan y dependen también del Jacobiano de la red.

## 7. ¿Llegan a acercarse a la plataforma?

Se predefinió para las trayectorias finales una zona diagnóstica: cuerpo por encima
del suelo, centro horizontal dentro de ±1.5 m, ambos apoyos sobre la plataforma y
apoyo inferior a no más de 1 m de altura. No es un criterio de éxito nuevo.
**0/32 casos finales** de cada agente nuevo entró en esa zona: ascienden desde
10–12 m y salen por el techo. La distancia mínima media del centro a (0,1) fue
~9.91 m; no se aproximan al aterrizaje en su política final.

Sí hubo acercamientos y contactos en checkpoints intermedios, que no se ocultan:

- DQN nuevo: 224 contactos en los siete puntos 10000–70000; **los 224 incumplieron
  velocidad vertical**. También fallaron orientación en 212, velocidad angular
  en 210 y velocidad horizontal en 187. Los criterios no son excluyentes.
- Double DQN nuevo: 160 contactos en 10000 y 30000–60000; todos tuvieron apoyos
  dentro de plataforma y centro dentro del umbral horizontal, pero **los 160
  incumplieron velocidad vertical**. El mínimo |vy| observado fue 5.158 m/s,
  frente al límite 1 m/s. Hubo además fallos de vx en 138, θ en 135 y ω en 147.

Al paso 10000 Double DQN nuevo tuvo 32 contactos inválidos, |vy| medio 7.339 m/s,
|θ| medio 0.097 rad y error horizontal medio 0.606 m. Dos casos cumplieron todos
los criterios salvo vy:

| Caso de validación | x absoluto (m) | vx (m/s) | vy (m/s) | θ (rad) | ω (rad/s) |
|---|---:|---:|---:|---:|---:|
| 003 | 0.64911 | -0.67372 | -8.11736 | -0.00226 | -0.20769 |
| 014 | 0.64781 | -0.67374 | -8.13970 | -0.00187 | -0.18924 |

Se localizaron por diagnóstico de criterios, **no** se presentan como ejemplos
representativos elegidos por éxito. Las figuras comparativas representativas usan
los casos fijos 000 y 016 (primero vertical y primero desplazado), iguales en todos
los modelos; también se dibujan las 32 trayectorias finales y las trazas
inicial/mejor/final del experimento nuevo. No hay ejemplos de éxitos que mostrar.

## 8. Interpretación y siguiente decisión

Se distinguieron las posibilidades:

- **a) Menor inestabilidad y progreso de control:** no respaldada como resultado
  global; no hay aterrizajes ni aproximación de las políticas finales.
- **b) Menor inestabilidad sin progreso de aterrizaje:** reducir Q en DQN respecto
  al piloto es una mejora parcial de magnitud, pero no demuestra estabilización:
  el error terminal vuelve a crecer y sigue siendo grande.
- **c) Inestabilidad persistente:** es la conclusión del experimento. DQN conserva
  deriva; Double DQN presenta mayores brechas y errores terminales con LR menor.
  Una sola semilla no permite generalizar superioridad ni efecto monotónico de LR.

El patrón es compatible con competencia entre muchos targets no terminales
bootstrap y una señal terminal grande cuya derivada Huber se satura; no prueba
que ese sea el mecanismo único. Los errores terminales permanecen grandes aunque
su target es correcto. El batch de diagnóstico está balanceado, pero **el replay
real de entrenamiento no lo está**. El replay fijo no se utilizó como una técnica
de entrenamiento.

**Siguiente experimento propuesto, no implementado ni ejecutado:** primero
cuantificar offline los gradientes separados de terminales y no terminales
(normas y coseno), aprovechando esta misma sonda, para distinguir saturación e
interferencia. Después, como una única ablación de entrenamiento identificada
aparte, probar **SmoothL1/Huber beta=10 frente a beta=1**, manteniendo el resto del
experimento actual, incluidos LR=.0001, clipping=10, recompensa y distribución.
Aumentar beta reduce la derivada de residuos pequeños no terminales respecto a
los residuos terminales grandes; permite comprobar esa hipótesis sin añadir
imitación, replay priorizado ni cambiar el problema físico. Puede empeorar el
aprendizaje y no se presenta como solución garantizada.

Criterio de decisión de esa futura prueba: reducción sostenida del error terminal
y de la deriva de bootstrap, sin perder los contactos orientados y con menor
velocidad de descenso; no solo un Q menor. No conviene ampliar ahora a millones
de pasos. No se encadenó ninguna otra ablación.

## 9. Archivos y comandos

Configuraciones: `configs/dqn_lr1e4.json`, `double_dqn_lr1e4.json`; protocolo previo
`lr1e4_protocol.json`; auditoría `docs/lr1e4_audit.json`.

Resultados reales:

- `artifacts/dqn_lr1e4_100k/`, `artifacts/double_dqn_lr1e4_100k/`: logs, initial/best/last
  checkpoints, validación y trazas; best=10000, last=100000.
- `artifacts/lr_ablation_curves.png`: los cuatro entrenamientos sin sobrescribirlos.
- `artifacts/lr_ablation_comparison/fixed_transitions.csv`: 5632 filas; la misma
  transición conserva su identificador entre modelos y checkpoints.
- `fixed_transition_summary.csv`: estadísticas firmadas por grupo/paso.
- `fixed_td_signed.png`, `fixed_gradients.png`: señales firmadas y clipping offline.
- `greedy_policy_return_gap.csv/png`: brecha frente al retorno de la política greedy.
- `comparison.csv`, `validation_progress.csv`, `contact_criteria.csv`, `approach.csv`.
- `fixed_examples.png`, `all_final_trajectories.png`: casos fijos y todos los casos.
- `artifacts/lr1e4_analysis/examples.png`: comparación inicial/mejor/final nueva.
- Copias pequeñas versionadas: `docs/lr1e4_comparison.csv`,
  `lr1e4_fixed_transition_summary.csv`, `lr1e4_fixed_batch_gradients.csv`.

Repetir el entrenamiento en directorios nuevos (solo esta ablación):

```bash
source .venv/bin/activate
python -m pytest
python scripts/check_env.py
python scripts/train.py --config configs/dqn_lr1e4.json --output artifacts/dqn_lr1e4_repeat
python scripts/train.py --config configs/double_dqn_lr1e4.json --output artifacts/double_dqn_lr1e4_repeat
python scripts/analyze_pilot.py artifacts/dqn_lr1e4_repeat artifacts/double_dqn_lr1e4_repeat --output-dir artifacts/lr_repeat_analysis
python scripts/fixed_replay_diagnostics.py --probe artifacts/lr_audit_reference/fixed_probe.pt --runs artifacts/dqn_pilot_100k artifacts/double_dqn_pilot_100k artifacts/dqn_lr1e4_repeat artifacts/double_dqn_lr1e4_repeat --output artifacts/lr_repeat_comparison
python scripts/compare_lr_ablation.py --new-runs artifacts/dqn_lr1e4_repeat artifacts/double_dqn_lr1e4_repeat --new-analysis artifacts/lr_repeat_analysis --output artifacts/lr_repeat_comparison
```

La muestra existente no se sobrescribe. Para reconstruirla si faltara, usando
los artefactos previos preservados:

```bash
python scripts/fixed_replay_diagnostics.py --probe artifacts/lr_audit_reference/fixed_probe.pt --create-from artifacts/dqn_pilot_100k/step_00010000.resume.pt
```

Ese comando debe ejecutarse solo si el archivo no existe. La procedencia, semilla,
índices e identificadores permiten verificar que se seleccionan las mismas
transiciones. Versiones CPU de Etapa 3 conservadas. Solo las nuevas normas reales
tras clipping se añadieron al entrenamiento; las sondas se computan offline y no
modifican pesos, replay ni archivos anteriores.

Tiempo real de ciclo, incluyendo validación y guardado: DQN **56.574 s / 1767.59
pasos/s**; Double DQN **56.984 s / 1754.89 pasos/s**. Ejecuciones secuenciales.
No incluye análisis/gráficas posteriores ni importación/preparación inicial.
No se lanzó la campaña completa. Los artefactos grandes permanecen fuera de Git.
