# Piloto controlado de 100000 pasos por algoritmo

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

Experimento `pilot_100k_seed42_v1`. Se ejecutó **solo este piloto**, desde cero,
con una semilla por algoritmo. No se ejecutó la campaña de tres semillas ni la
prueba final. Física, recompensa, catálogo de ocho acciones, umbrales y PID se
conservaron: `verify_reference()` comprobó los hashes antes y después. Los
resultados PID/smoke previos permanecen en sus directorios originales.

## Protocolo fijado antes de comenzar

`configs/pilot_100k_protocol.json` registra las decisiones y hashes previos.
Configuraciones nuevas: `configs/dqn_pilot.json` y `double_dqn_pilot.json`.
Son iguales salvo `algorithm`. No se cargó ningún checkpoint ni demostración.

| Parámetro | Valor en ambos |
|---|---|
| Semilla / dispositivo / hilos Torch | 42 / CPU / 1 |
| Arquitectura | 9 → 128 ReLU → 128 ReLU → 8 |
| Learning rate / gamma | 0.0003 / 0.99 |
| Replay / batch / warmup | 50000 / 64 / 1000 pasos |
| Actualización | un minibatch cada 4 pasos |
| Red objetivo | copia dura cada 500 pasos, desde warmup |
| Huber / límite de gradiente | beta=1 / norma global 10 |
| Epsilon | 1.0 → 0.1 lineal en 80000 pasos **después** del warmup |
| Presupuesto / validación / guardado | 100000 / cada 10000 / cada 10000 pasos |

El smoke usaba descenso epsilon de 8000 pasos: habría alcanzado 0.1 en el paso
9000, dejando el 91% del piloto con exploración mínima. El piloto llega a 0.1
al paso 81000. Valores: 0.89875 al paso 10000, 0.56125 al 40000 y 0.11125 al
80000. Es una decisión de diagnóstico, no un ajuste óptimo. Se cambiaron también
las frecuencias de validación/guardado de 5000 a 10000 y se habilitó registro
detallado. Ambos comenzaron de nuevo por cambiar ese calendario.

Misma distribución: los 128 estados verticales sin viento de
`rl_train_scenarios.json`, muestreados con reemplazo, con RNG separado de la
exploración. Misma validación: los 32 estados de `rl_validation_scenarios.json`,
16 verticales y 16 con desplazamiento/inclinación pequeños. Son los mismos casos
en cada punto, **no** 320 escenarios independientes. La prueba final se leyó
únicamente para excluir coincidencias, nunca para ejecutar episodios.

Se registró baseline greedy de la red aleatoria en paso 0 y validación greedy
cada 10000 pasos. Evaluar no actualiza pesos ni replay ni consulta RNG de
exploración. Sus interacciones no cuentan dentro de los 100000 pasos de aprendizaje.

### Criterio de mejor checkpoint

Se predefinió: mayor porcentaje de éxito; en empate, menor media de |x| al contacto
**entre éxitos**; si persiste empate, conservar el primero. Cuando no hay éxitos,
el error de aterrizaje es ausente, no se sustituye por error de impactos o de
salidas del área. El baseline inicial se excluye de la selección. En ambos agentes
todos los puntos tuvieron cero éxitos: `best.*` corresponde al paso **10000** por
empate. Ese nombre no demuestra que el checkpoint aterrice.

## Resultados reales

| Métrica | DQN | Double DQN |
|---|---:|---:|
| Pasos de entrenamiento | 100000 | 100000 |
| Actualizaciones / copias objetivo | 24751 / 199 | 24751 / 199 |
| Episodios completos de entrenamiento | 2855 | 2642 |
| Éxitos de entrenamiento | 0 | 0 |
| Éxitos en cada punto de validación | 0/32 | 0/32 |
| Contactos en validación final | 0 | 0 |
| Salidas por el techo en validación final | 32 | 32 |
| Retorno medio final, todos | -101.63790 | -102.05825 |
| Duración media final, todos (s) | 1.16475 | 1.29519 |
| Impulso medio final, todos (N s) | 3494.25 | 3195.46 |
| Loss media del último intervalo | 6.10662 | 9.59483 |
| Error TD absoluto medio, último intervalo | 6.36519 | 9.87369 |
| Magnitud Q media muestreada, último intervalo | 108.83282 | 215.52640 |

Fallos durante entrenamiento:

| Causa | DQN | Double DQN |
|---|---:|---:|
| out_of_bounds | 2491 | 2223 |
| body_contact | 204 | 264 |
| missed_platform | 152 | 144 |
| invalid_contact | 8 | 11 |
| time_limit / success | 0 / 0 | 0 / 0 |

Los episodios parciales del final se conservan en los checkpoints y no se suman
como episodios completos. Los CSV incluyen semilla/estado inicial, resultado,
retorno y componentes, gas, duración y datos de contacto. En la validación final
las columnas `contact_*` están **vacías en los 32 casos**: salir por el techo no es
aterrizar ni impactar. Sus estados finales se reportan con nombres distintos.

El CSV `docs/pilot_100k_validation_progress.csv` conserva todos los puntos, los
n_contact, causas y medias de componentes de recompensa; las medias de contacto
en puntos sin contacto son ausentes. No se interpreta una reducción de consumo
causada por terminar pronto como eficiencia de aterrizaje.

## Diagnóstico antes de ampliar presupuesto

### Control parcial frente a aterrizaje

Ambas redes iniciales eran idénticas. En baseline, los 32 casos hicieron contacto
fallido: |x| medio 1.868 m, |θ| 1.361 rad, |ω| 18.047 rad/s, |vy| 14.679 m/s.
Al final, los siguientes valores son **estados de salida**, no métricas al contacto:

| Estado final medio | DQN | Double DQN |
|---|---:|---:|
| Altura del centro (m) | 24.197 | 23.942 |
| Error horizontal absoluto (m) | 0.086 | 2.671 |
| Inclinación absoluta (rad) | 0.00451 | 0.63989 |
| Velocidad angular absoluta (rad/s) | 0.000 | 1.215 |
| vy, con signo (m/s) | +23.157 | +17.899 |

DQN aprendió una respuesta con buen centrado y orientación pero aceleró hacia
arriba hasta abandonar el área. Double DQN redujo rotación respecto a su baseline,
pero su desplazamiento horizontal empeoró y también ascendió. No hay mejora del
aterrizaje ni control adecuado de velocidad vertical. Estos datos de una semilla
no establecen superioridad de un algoritmo.

### Q: finito no significa estable o calibrado

Se reconstruyeron recompensas **por transición** desde las trazas usando la función
de recompensa congelada. Se verificó que su suma coincide con cada retorno CSV
(tolerancia 1e-6) y se calculó el retorno descontado con gamma=0.99. Así se compara
Q con la magnitud correcta, no directamente con retornos sin descuento.

| Media sobre los mismos 32 estados iniciales, checkpoint final | DQN | Double DQN |
|---|---:|---:|
| Q de la acción greedy | +119.784 | +221.986 |
| Retorno descontado observado de esa política | -80.989 | -79.220 |
| Diferencia predicción − observado | +200.773 | +301.206 |

El Q greedy medio pasó de +0.038 en baseline a esos valores, mientras no apareció
ningún aterrizaje. Las pérdidas y errores TD también crecieron. En los estados
visitados en validación final, max |Q| llegó a 146.84 / 252.33. No hubo NaN/Inf,
pero sí deriva optimista y mala calibración. El resultado contradice la idea de
que bastaría comprobar finitud y ampliar ciegamente el entrenamiento.

En los últimos 50000 elementos de replay hay 1700 / 1589 transiciones terminales
(3.40% / 3.18%) y **ninguna** terminación positiva. Las recompensas del replay
final estuvieron en [-100.2094,-0.00251] para DQN y [-100.2405,+0.01250] para
Double DQN. La retroalimentación de bootstrap con aproximación y señal positiva
ausente es una hipótesis respaldada por estos registros; no prueba un mecanismo
único. Double DQN no elimina toda inestabilidad de aproximación funcional.

### Recompensa, duración y ahorro de gas

Todos los fallos llevan componente terminal -100. El resto de los componentes
medios de validación final fueron:

| Componente | DQN | Double DQN |
|---|---:|---:|
| Progreso | -0.32289 | -0.74766 |
| Velocidad | -0.11094 | -0.11055 |
| Orientación | -0.00050 | -0.05337 |
| Gas consumido | -0.97062 | -0.88763 |
| Tiempo | -0.23295 | -0.25904 |

Entre políticas que fracasan, el retorno sin descuento puede mejorar si se acaba
antes y se consume menos, aunque el aterrizaje no mejore. Por ejemplo, DQN en
paso 10000 duró 11.175 s y obtuvo -105.950 sin descuento; al final duró 1.165 s y
obtuvo -101.638. El criterio de mejor checkpoint **no** utiliza ese retorno.

El descuento introduce otra preferencia: para esas mismas políticas DQN, los
retornos descontados medios fueron -14.784 y -80.989. Retrasar una penalización
terminal grande puede resultar ventajoso bajo gamma=0.99, cuyo horizonte
característico es unos 100 pasos, equivalentes a 5 s. No hubo truncamientos ni
vuelo indefinido observado; no se afirma que se haya aprendido a flotar. Tampoco
hay evidencia de una política eficiente de ahorro: ambas salen con mucho gas.

### Exploración y verificaciones de implementación

Epsilon seguía en 0.674 al paso 30000, cuando ya aparecía la deriva de Q, y no
alcanzó 0.1 hasta el paso 81000. La exploración no desapareció temprano. Su nivel
alto sí puede dificultar secuencias precisas, pero estos datos no aíslan su efecto.

Se revisaron los replays finales: observaciones finitas de dimensión 9; seno y
coseno con error máximo de norma 1.19e-7; gas en [0,1]; viento cero. Velocidades y
ω normalizados pueden exceder ±1 según el diseño, sin recorte: no es un error.
Los tests verifican escalas físicas. No se detectó intercambio de componentes.

Se recalcularon targets de un bloque de replay con las redes guardadas: fueron
finitos y los terminales coincidieron exactamente con la recompensa (error máximo
0). La suite verifica valores conocidos distintos para DQN/Double DQN, ausencia
de gradientes objetivo y bootstrap de truncamientos con observación final real.
En este piloto hubo cero truncamientos: ese caso está verificado por pruebas,
no por episodios del piloto. No se halló evidencia de errores en esas operaciones;
esto no descarta cualquier posible defecto de implementación.

### Cambios propuestos como experimentos nuevos, no ejecutados

**No recomiendo empezar aún la campaña de múltiples semillas con esta configuración.**
Antes de aumentar presupuesto, propongo:

1. Ablación de estabilidad con igual entorno/distribución/semilla: LR **0.0001**
   frente a 0.0003, manteniendo lo demás fijo; seguir Q firmado y calibración con
   retornos descontados cada 10000 pasos. No asumir que bajar LR resolverá todo.
2. En otra ablación separada, límite de gradiente **1** frente a 10. Las pérdidas
   aumentan y la norma previa al recorte ya supera el límite actual en intervalos
   tardíos. No cambiar simultáneamente varios parámetros sin poder atribuir efecto.
3. Si la deriva persiste, auditar TD de terminales y no terminales por separado y
   comparar replay estratificado/priorizado con corrección de muestreo: los
   terminales son solo ~3% del buffer. Sería una variante nueva del algoritmo,
   no un resultado de DQN básico ni una demostración PID.
4. Una vez estable Q, estudiar por separado gamma **0.999** frente a 0.99 para
   reducir la ventaja de retrasar fallos. Puede dificultar estabilidad y crédito;
   requiere experimento propio. Cualquier cambio posterior de recompensa o
   dificultad debe versionarse aparte y validarse sin abrir prueba final.

No se aplicó ninguno de estos cambios. No se modificó recompensa, dificultad,
física ni criterio de éxito, ni se usó imitación. No se atribuye todo el fallo a
un único hiperparámetro.

## Archivos, checkpoints y reproducción

Directorios reales: `artifacts/dqn_pilot_100k/` y `artifacts/double_dqn_pilot_100k/`.
Cada uno incluye `episodes.csv`, `updates.csv`, `validation.csv`, `metadata.json`,
`initial_validation.*`, `best_validation.*`, `final_validation.*`, políticas
inicial/mejor/última, checkpoints periódicos y `best.resume.pt` / `last.resume.pt`.
Los mejores son del paso 10000; los últimos, del 100000. Se cargaron ambos tipos
y se verificaron sus contadores. Artefactos grandes siguen excluidos de Git.
`configs/reference_pilot_100k.json` registra hashes de resultados y checkpoints.

Figuras inspeccionadas:

- `artifacts/pilot_100k_curves.png`: retorno, loss, TD, Q, epsilon y validación.
- `artifacts/pilot_100k_analysis/initial_final_trajectories.png`: **los 32 casos**
  iniciales/finales por algoritmo, coloreados por resultado.
- `examples.png`: primer caso de cada condición y primer éxito/fallo si existe,
  comparando inicial/mejor/final; no se eligen por retorno. No hay ejemplos de éxito.
- `contact_progress.png`: métricas condicionadas a contacto; ausentes donde n=0.
- `q_calibration.png`: evolución de Q y contraste con retornos descontados.

CSV de análisis: `validation_progress.csv`, `timing.csv`, `examples.csv`,
`q_progress.csv`, `q_calibration.csv`, más `diagnostics.json`. Se conservan copias
pequeñas en `docs/pilot_100k_validation_progress.csv`, `pilot_100k_timing.csv` y
`pilot_100k_q_progress.csv`.

Desde la raíz, reproducir desde cero en **directorios nuevos**:

```bash
source .venv/bin/activate
python -m pytest
python scripts/check_env.py
python scripts/train.py --config configs/dqn_pilot.json --output artifacts/dqn_pilot_repeat
python scripts/train.py --config configs/double_dqn_pilot.json --output artifacts/double_dqn_pilot_repeat
python scripts/plot_training.py artifacts/dqn_pilot_repeat artifacts/double_dqn_pilot_repeat --output artifacts/pilot_repeat_curves.png
python scripts/analyze_pilot.py artifacts/dqn_pilot_repeat artifacts/double_dqn_pilot_repeat --output-dir artifacts/pilot_repeat_analysis
```

La semilla y datos son fijos; los tiempos no son reproducibles bit a bit. Versiones
CPU en `configs/requirements-stage3.txt`. Los tests pasaron **77 pruebas**, con las
dos advertencias conocidas del Box infinito; el checker Gymnasium también pasó.
Las nuevas pruebas verifican selección del mejor checkpoint, errores de aterrizaje
ausentes cuando no hay éxitos y exportación inicial/mejor/final. Las verificaciones
de artefactos confirmaron contadores, mismos 32 identificadores en todos los
puntos, suma de componentes y ausencia de métricas de contacto en salidas.

Evaluar o visualizar las políticas existentes (solo validación):

```bash
python scripts/evaluate.py --checkpoint artifacts/dqn_pilot_100k/best.policy.pt --trace --output artifacts/dqn_pilot_best_eval.csv
python scripts/evaluate.py --checkpoint artifacts/double_dqn_pilot_100k/last.policy.pt --trace --output artifacts/double_pilot_last_eval.csv
python scripts/play.py --controller dqn --checkpoint artifacts/dqn_pilot_100k/last.policy.pt --scenarios configs/rl_validation_scenarios.json --episodes 3 --render --output artifacts/dqn_pilot_visual.csv
```

No se afirma verificación de teclado/ventana reales durante este piloto. Las
figuras se generaron con Agg desde trazas; las verificaciones SDL dummy previas
siguen siendo comprobaciones automáticas, no interacción humana.

## Tiempo medido y campaña prospectiva

Ejecuciones **secuenciales**, CPU y un hilo Torch. Tiempo medido del ciclo,
incluyendo baseline inicial, las diez validaciones, trazas, checkpoints y CSV;
excluye importación/preparación inicial del proceso y análisis/gráficas posteriores.

| Algoritmo | Tiempo (s) | Pasos/s | Física (s) | Aprendizaje (s) | Validación (s) | Guardado/exports de checkpoints (s) |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 54.760 | 1826.15 | 29.172 | 14.305 | 5.712 | 0.818 |
| Double DQN | 55.440 | 1803.74 | 29.526 | 14.876 | 5.384 | 0.848 |

El resto del tiempo incluye selección de acciones, replay, acumulación y escrituras
CSV periódicas/finales. Total de ciclos: **110.201 s** para las 200000 interacciones.

Para la campaña propuesta anteriormente de 1000000 pasos × 3 semillas × 2
algoritmos, la extrapolación con esos rendimientos es
`3*1000000/1826.1466 + 3*1000000/1803.7373 = 3306.0 s ≈ 55.1 min`.
Incluye proporcionalmente la validación y guardado actuales. Reservaría **60–90
minutos**, posiblemente más si las trayectorias aprendidas son más largas o crece
el coste de guardar historias. No es una medición de esa campaña ni garantía de
que aprenda. **No se ejecutó ni se recomienda lanzarla antes de resolver el
problema de calibración observado.**
