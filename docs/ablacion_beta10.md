# Ablación única: SmoothL1 beta=10 frente a beta=1

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

**Experimento:** `smoothl1_beta10_lr1e4_seed42_100k_v1`.
**Referencia preservada:** `lr_0p0001_seed42_100k_v1`.

**Conclusión:** se redujo la deriva optimista observada y el error terminal en
sondas fijas, pero **no aparecieron aterrizajes** en entrenamiento ni validación.
Hay vuelos más largos y algunos contactos con buena orientación, a costa de
mucho más gas y sin controlar el descenso. No se amplió presupuesto, ni se cambió
recompensa, dificultad o muestreo, ni se ejecutó prueba final.

## 1. Control experimental y prueba de la pérdida

Se entrenaron DQN y Double DQN desde cero, CPU, semilla 42, 100000 pasos y 24751
actualizaciones cada uno. Configuraciones: `dqn_beta10.json` / `double_dqn_beta10.json`.
Al expandir los defaults, difieren de sus baselines **solo** en `smooth_l1_beta`:
1→10. Redes iniciales idénticas a los agentes anteriores, comprobadas tensor a
tensor. Se conservan LR=.0001, Adam, MLP 128/128 ReLU, gamma=.99, buffer=50000,
batch=64, clipping=10, warmup=1000, una actualización cada 4 pasos, copia objetivo
cada 500, epsilon 1→.1 en 80000 pasos tras warmup y los mismos manifiestos.

El agente usa exactamente:

```python
self.loss_function = torch.nn.SmoothL1Loss(beta=config.smooth_l1_beta, reduction="mean")
loss = self.loss_function(q, targets)
```

No hay multiplicador compensatorio ni `HuberLoss(delta=10)`. Antes de entrenar se
probaron valores y pendientes, incluyendo la actualización real de `learn`:
residuos [2,20,2] producen loss media (.2+15+.2)/3 y gradientes seleccionados
[.4/3,1/3], sin factor 10 adicional. Para residuo 20 la pendiente por muestra es
1; `HuberLoss(delta=10)` tendría pendiente 10 y pérdida diez veces mayor.
Se comprobó además equivalencia exacta de beta=1 con la función anterior.

Para e=Q−target, la pendiente es `clip(e/beta,−1,1)`. Beta=10 reduce diez veces
la pendiente cuando |e|<1, reduce la saturación previa en 1≤|e|<10 a e/10 y
mantiene ±1 para |e|≥10. Su efecto depende de los residuos, no del nombre del grupo.

Validación greedy inicial y cada 10000 pasos, mismos 32 casos, sin explorar ni
actualizar pesos. Se mantuvo el criterio de checkpoint: éxito, luego menor error
horizontal entre éxitos, luego primero. Todos los puntos tuvieron cero éxitos:
`best`=10000 por empate y `last`=100000; no se eligió por loss o por Q.

## 2. Tabla de resultados de control

| Métrica final | DQN β1 | DQN β10 | Double DQN β1 | Double DQN β10 |
|---|---:|---:|---:|---:|
| Éxitos entrenamiento / episodios completos | 0/2685 | 0/958 | 0/2709 | 0/1242 |
| Éxitos en cada punto de validación | 0/32 | 0/32 | 0/32 | 0/32 |
| Contactos en validación final | 0 | 0 | 0 | 32 |
| Salidas por techo finales | 32 | 32 | 32 | 0 |
| Contactos fuera de plataforma finales | 0 | 0 | 0 | 27 |
| Contactos inválidos finales | 0 | 0 | 0 | 5 |
| Retorno SIN descuento, todos | -102.193 | -108.989 | -101.925 | -109.707 |
| Duración media, todos (s) | 1.216 | 16.373 | 1.175 | 20.074 |
| Impulso medio, todos (N s) | 3317.28 | 16923.78 | 3494.11 | 18000.00 |

No hubo éxitos ni límites temporales en los episodios completos de entrenamiento.
Fallos de entrenamiento con β10: DQN 779 salidas, 96 fuera de plataforma,
65 contactos del cuerpo y 18 contactos inválidos; Double DQN 1106 salidas,
44 fuera de plataforma, 88 del cuerpo y 4 inválidos. Los episodios parciales
finales (239/381 pasos) no se cuentan como completos y quedan en el checkpoint.

Consumo y duración finales **separados por resultado**:

| Agente / resultado | n | Duración media (s) | Impulso medio (N s) |
|---|---:|---:|---:|
| DQN β1 / salida | 32 | 1.216 | 3317.28 |
| DQN β10 / salida | 32 | 16.373 | 16923.78 |
| Double β1 / salida | 32 | 1.175 | 3494.11 |
| Double β10 / fuera de plataforma | 27 | 20.075 | 18000.00 |
| Double β10 / contacto inválido | 5 | 20.070 | 18000.00 |

Double β10 terminó sin gas en los 32 casos. En sus contactos: error horizontal
absoluto medio **3.013 m**, |vx|=.187 m/s, |vy|=**17.633 m/s**, |θ|=.0203 rad y
|ω|≈0. Ninguno cumplió el límite vertical de 1 m/s; también falló posición o
geometría de plataforma. Los cinco contactos inválidos tuvieron error horizontal
medio 1.808 m; los otros 27, 3.236 m. Estos valores no se atribuyen a los otros
agentes: sus columnas de contacto finales están vacías porque salieron del área.

Hubo control parcial en puntos intermedios: DQN β10 al paso 30000 tuvo 32 contactos
inválidos con |vy| medio 8.092 m/s y |θ|=.074 rad; al 40000, 32 contactos fuera de
plataforma con |vy|=5.839 m/s y |θ|=.023 rad. No equivalen a aterrizajes ni muestran
progreso monótono. Todos los puntos y causas, sin selección favorable, están en
`validation_progress.csv`, `validation_by_outcome.csv` y `training_by_outcome.csv`.

## 3. Fuentes diagnósticas: no se mezclan distribuciones

Ninguna sonda se insertó en replay ni produjo pasos de optimizador:

- **A_fixed128:** las mismas 64 terminales y 64 no terminales antiguas de
  `fixed_probe.pt`, aplicadas a los cuatro modelos en los 11 checkpoints. Es una
  sonda balanceada, **no** una estimación de proporciones del replay.
- **B_fixed_baseline:** los mismos 128 minibatches uniformes de 64 del diagnóstico
  anterior, con sus índices y replay original LR=.0001. Cada par β1/β10 ve las
  mismas transiciones; es el contraste de gradientes con datos controlados.
- **C_new_replay:** 128 minibatches uniformes por replay β10 final. Se aplican a
  ambos modelos del par y se reportan por separado de B. Los estados cambiaron
  por las políticas aprendidas, no por un cambio manual de distribución o muestreo.
- **D_fixed_holdout:** las mismas 256 terminales reservadas del diagnóstico previo
  por algoritmo, en todos los checkpoints. Son una sonda de datos de entrenamiento,
  no un conjunto externo de prueba final.

Los identificadores de transición, hashes, índices y procedencias están guardados.
Se mide también cada red con beta diagnóstico 1 y 10 **sin actualizarla**, para
separar el efecto inmediato de la pérdida del efecto de haber aprendido otros pesos.
El campo `model_beta` distingue la pérdida de entrenamiento de `evaluated_beta`.

## 4. Q, targets y TD firmados

Medias finales en la **misma sonda A**. TD=target−Q; target terminal es recompensa
exactamente, no bootstrap. En no terminales el target es estimado por la red:

| Modelo / grupo | Q firmado | Target | TD firmado |
|---|---:|---:|---:|
| DQN β1 / T | +7.129 | -100.072 | -107.202 |
| DQN β10 / T | -82.081 | -100.072 | -17.992 |
| DQN β1 / N | +50.903 | +51.199 | +0.296 |
| DQN β10 / N | -21.512 | -20.914 | +0.598 |
| Double β1 / T | +291.502 | -100.072 | -391.575 |
| Double β10 / T | -55.784 | -100.072 | -44.288 |
| Double β1 / N | +341.696 | +349.515 | +7.819 |
| Double β10 / N | -13.283 | -12.794 | +0.489 |

La media firmada puede cancelar sobreestimaciones e infraestimaciones. Por eso:

| Error terminal absoluto medio (MAE) | DQN β1→β10 | Double β1→β10 |
|---|---:|---:|
| Mismas 64 terminales A | 107.202 → 51.934 | 391.575 → 45.088 |
| Mismas 256 terminales D | 182.231 → 89.777 | 599.885 → 44.160 |

En A, el error terminal DQN β10 todavía tiene p05=−65.77 y p95=+99.06; Double β10,
p05=−2.47 y p95=+96.66. No se resolvió todo el error ni se probó convergencia.
El TD N medio de DQN en A incluso aumentó ligeramente; un Q menor no garantiza
que toda medida mejore. Los CSV conservan muestras y percentiles firmados de
todas las fuentes, incluyendo C, en lugar de comparar solo magnitudes.

## 5. Gradientes ponderados y distribuciones de residuos

Dos capas ocultas compartidas; gT/gN son gradientes de las pérdidas medias de
cada grupo. Se verifica `g_batch = pT*gT + pN*gN`. La norma de pTgT no se suma
como si fuera un porcentaje de actualización Adam. No hay coseno ficticio en
batches sin alguno de los grupos.

Sobre **B, con datos fijos** y objetivo propio de cada modelo:

| Modelo | Extracciones T/N | Media norma pTgT | Media norma pNgN | Coseno medio |
|---|---:|---:|---:|---:|
| DQN β1 | 267/7925 | 1.911 | 19.470 | -0.579 |
| DQN β10 | 267/7925 | 1.602 | 3.543 | +0.060 |
| Double β1 | 266/7926 | 5.690 | 70.464 | -0.700 |
| Double β10 | 266/7926 | 1.459 | 3.126 | -0.119 |

Sobre **C, datos nuevos**, manteniendo emparejados los mismos datos entre modelos:

| Modelo evaluado | Replay fuente | Extracciones T/N | Norma pTgT / pNgN |
|---|---|---:|---:|
| DQN β1 | nuevo DQN β10 | 35/8157 | .170 / 20.490 |
| DQN β10 | nuevo DQN β10 | 35/8157 | .157 / .241 |
| Double β1 | nuevo Double β10 | 69/8123 | .808 / 84.868 |
| Double β10 | nuevo Double β10 | 69/8123 | .281 / .217 |

En los replays nuevos completos hay 217/50000 y 370/50000 terminales: episodios
más largos producen menos terminales por interacción. Sus frecuencias de muestreo
fueron .427%/.842%, frente a ~3.25% en B. No se interpreta esa composición distinta
como una mejora del modelo; A/B/D permiten contrastar con datos idénticos.

Tampoco se presupone que beta afecte solo a N. En el modelo Double β10 sobre B,
14.29% de T y 97.69% de N tienen |TD|<10. Manteniendo **ese mismo modelo y datos**,
el cambio diagnóstico beta 1→10 reduce la derivada absoluta media T de .997 a
.927, y N de .685 a .169; las normas ponderadas pasan de 1.544/10.497 a
1.459/3.126. En DQN β10 sobre B, las T observadas permanecen saturadas, pero N
reduce su derivada media de .568 a .165. Son distribuciones medidas, no una regla
universal terminal/no terminal.

En C los T observados tienen |TD|≥10, mientras 97.30%/99.74% de N tienen |TD|<1.
Ese resultado explica un efecto relativo más marcado sobre N en esas muestras,
pero con solo 35/69 extracciones T la evidencia es limitada. Las tablas completas
incluyen objetivos comunes beta=1 y beta=10; no se confunde loss propia con loss
comparada a una escala común.

## 6. Brecha Q inicial frente al retorno greedy

Retornos descontados reconstruidos por transición de las mismas trazas; su suma
sin descuento coincide con el CSV. Esto es una **brecha frente al retorno de la
política greedy**, no error exacto respecto a Q óptimo:

| Valor medio final, 32 estados iniciales | DQN β1→β10 | Double β1→β10 |
|---|---:|---:|
| Q de la primera acción greedy | +83.866 → +14.672 | +428.659 → +15.168 |
| Retorno descontado observado | -80.608 → -6.690 | -81.077 → -4.688 |
| Brecha Q−retorno | +164.474 → +21.362 | +509.736 → +19.856 |

Los retornos descontados mejoran en gran parte por retrasar fallos unos 16–20 s;
los retornos **sin** descuento empeoran, se consume mucho gas y éxito sigue cero.
No se atribuye la reducción de brecha únicamente a mejores predicciones de un
mismo comportamiento: también cambió la trayectoria greedy. El MAE terminal fijo
aporta evidencia adicional independiente de esa trayectoria.

**Interpretación:** mejora numérica observada, sin aprendizaje de aterrizaje.
Hay control parcial de orientación y prolongación del vuelo, pero no descenso
seguro y preciso con gas limitado. La menor loss no es por sí sola evidencia:
beta cambia su escala. Persisten errores terminales grandes y brecha positiva.
Una semilla y 100000 pasos no permiten asegurar estabilidad asintótica ni
superioridad general entre algoritmos. No se amplió el presupuesto.

## 7. Checkpoints, verificaciones y reproducción

Directorios reales: `artifacts/dqn_beta10_100k/` y
`artifacts/double_dqn_beta10_100k/`. Cada uno contiene logs, metadatos, validación,
trazas y checkpoints periódicos; `best.policy.pt`/`best.resume.pt` corresponden al
paso 10000 por empate, `last.policy.pt`/`last.resume.pt` al 100000. La configuración
beta se guarda en ambos tipos y los antiguos checkpoints siguen cargando beta=1
por defecto. El baseline y agentes originales se preservaron por hashes;
la fuente previa del agente y del test adaptado están en `beta10_reference/`.

**87 pruebas pasaron**, checker Gymnasium y `pip check` válidos. Se mantienen las
dos advertencias conocidas del Box infinito. Las pruebas nuevas verifican clase,
valores y pendientes, ausencia de multiplicación compensatoria, compatibilidad
beta=1 y descomposición proporcional. Se verificaron solo beta como diferencia,
pesos iniciales idénticos, 100000 pasos/24751 updates/199 copias, mismos 32 casos,
fuentes fijas y campos de contacto ausentes cuando no hay contacto.

Figuras inspeccionadas en `artifacts/beta10_comparison/`: `control_curves.png`,
`fixed_q_and_gap.png`, `gradient_sources.png` y `all_final_trajectories.png`.
CSV principales: `comparison.csv`, `validation_by_outcome.csv`,
`training_by_outcome.csv`, `signed_q_target_td.csv`, `terminal_error_sources.csv`,
`gradient_distribution_sources.csv`, `fixed_holdout_summary.csv` y
`greedy_policy_return_gap.csv`. Datos individuales e índices en
`artifacts/beta10_diagnostics/`; se distinguen A/B/C/D y beta entrenado/evaluado.
Copias pequeñas versionadas con prefijo `docs/beta10_`.

Repetir **solo esta ablación** en directorios nuevos:

```bash
source .venv/bin/activate
python -m pytest
python scripts/check_env.py
python scripts/train.py --config configs/dqn_beta10.json --output artifacts/dqn_beta10_repeat
python scripts/train.py --config configs/double_dqn_beta10.json --output artifacts/double_dqn_beta10_repeat
python scripts/analyze_pilot.py artifacts/dqn_beta10_repeat artifacts/double_dqn_beta10_repeat --output-dir artifacts/beta10_repeat_analysis
python scripts/diagnose_beta10.py --new-runs artifacts/dqn_beta10_repeat artifacts/double_dqn_beta10_repeat --output artifacts/beta10_repeat_diagnostics
python scripts/report_beta10.py --new-runs artifacts/dqn_beta10_repeat artifacts/double_dqn_beta10_repeat --new-analysis artifacts/beta10_repeat_analysis --diagnostics artifacts/beta10_repeat_diagnostics --output artifacts/beta10_repeat_comparison
```

Visualización opcional, solo validación (no se afirma teclado real comprobado):

```bash
python scripts/play.py --controller double_dqn --checkpoint artifacts/double_dqn_beta10_100k/last.policy.pt --scenarios configs/rl_validation_scenarios.json --episodes 3 --render --output artifacts/beta10_visual.csv
```

Tiempos de ciclo incluyendo validación/guardado: DQN 72.179 s, 1385.45 pasos/s;
Double DQN 62.166 s, 1608.59 pasos/s. Se ejecutaron secuencialmente. El mayor tiempo
incluye evaluaciones con vuelos más largos. No incluye análisis posterior ni
importación/preparación inicial; no es una comparación de rendimiento estadística.
No se ejecutó otra ablación, campaña completa ni prueba final.
