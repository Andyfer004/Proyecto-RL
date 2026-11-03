# Diagnóstico acotado: contribución, ajuste e interferencia

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

**Resultado:** las redes pueden ajustar targets terminales aislados. En los
checkpoints de LR=0.0001, actualizaciones no terminales y mixtas aumentan el error
terminal; intervienen tanto gradientes compartidos como los momentos de Adam.
El error se concentra especialmente en salidas por el techo. No es evidencia de
incapacidad general de la arquitectura ni una explicación única de toda la historia.

Diagnóstico `offline_interference_capacity_v1`, solo sobre los cuatro checkpoints
finales guardados. **Cero interacciones de entorno y cero evaluaciones**, sin
acceso evaluativo a prueba final. Todos los experimentos, física, recompensa,
umbrales y PID permanecieron intactos. No se ejecutó beta=10.

## 1. Minibatches representativos y gradientes

Por replay: 128 minibatches uniformes de 64, sin reemplazo dentro de cada batch,
con reemplazo entre batches; 8192 extracciones por modelo. No se balancearon grupos.
Se guardaron índices y semillas. T=terminal, N=no terminal; TD=target−Q.

En las dos capas ocultas compartidas se calculó
`g_T = grad(mean(loss_T))`, `g_N = grad(mean(loss_N))`, y se verificó
`grad(loss_batch) = p_T*g_T + p_N*g_N`. Las normas de contribución son anteriores
al clipping/Adam; no son porcentajes de cambio de parámetros. No se suma la norma
de ambos vectores como si fuera la norma del total. La capa final por acción se
excluye de esta descomposición; las actualizaciones de prueba sí usan toda la red.

| Modelo (LR) | T / 8192 | % T replay → muestra | Normas medias gT / gN¹ | Normas medias pTgT / pNgN | Coseno medio² |
|---|---:|---:|---:|---:|---:|
| DQN .0003 | 263 | 3.400 → 3.210 | 90.40 / 32.79 | 2.825 / 31.741 | +0.848 |
| Double .0003 | 268 | 3.178 → 3.271 | 144.15 / 50.40 | 4.592 / 48.766 | +0.795 |
| DQN .0001 | 267 | 3.246 → 3.259 | 63.15 / 20.13 | 1.911 / 19.470 | −0.579 |
| Double .0001 | 266 | 3.380 → 3.247 | 180.71 / 72.80 | 5.690 / 70.464 | −0.700 |

¹ Normas de medias dentro del grupo, condicionadas a que exista ese grupo.
Las contribuciones ponderadas incluyen cero cuando el grupo está ausente.
² Se dejó ausente cuando alguna norma era cero. Hubo 15/15/11/15 batches sin T.
En LR=.0001, coseno negativo en 107/117 y 111/113 batches definidos. En los dos
checkpoints .0003 no hubo cosenos negativos: la interferencia no es universal.

Todos los residuos T muestreados estuvieron en la región lineal de SmoothL1
(beta=1). Para N: 29.5%, 30.2%, 49.0% y 86.7%, respectivamente. El gradiente T
por grupo puede ser grande, pero su proporción reduce mucho su contribución.
En LR=.0001 su proyección media sobre el gradiente completo fue negativa
(−5.31% y −6.55%); son proyecciones geométricas, no cuotas de aprendizaje Adam.

Se conservaron Q, target, TD firmado y loss **por muestra** en `sample_values.csv`.
Ejemplo de medias por grupo en los checkpoints .0001:

| Modelo / grupo | Q | Target | TD | Loss por muestra |
|---|---:|---:|---:|---:|
| DQN / T | +77.970 | −100.051 | −178.021 | 177.521 |
| DQN / N | +66.361 | +66.853 | +0.492 | 1.349 |
| Double / T | +481.990 | −100.089 | −582.079 | 581.579 |
| Double / N | +414.789 | +419.188 | +4.399 | 4.683 |

Un gran TD terminal no implica proporcionalmente gran influencia: la derivada
se satura, hay pocos T y los Jacobianos varían por estado.

## 2. Ajuste supervisado congelado

Copias independientes de cada red; mismo MLP y normalización. Por replay se
seleccionaron 1024 grupos distintos de `(s,a)`: 768 para ajuste y 256 reservados,
sin duplicados exactos entre partes. Target fijo=recompensa terminal, sin bootstrap.
500 actualizaciones, batch=64, Adam **nuevo** con LR=.001, SmoothL1 beta=1,
clipping=10; sin adaptar el presupuesto al resultado. El LR de regresión es un
parámetro de esta prueba de capacidad, no otra ablación RL.

| Copia de red | MAE reservado inicial → final | MAE ajuste final |
|---|---:|---:|
| DQN .0003 | 202.165 → 1.360 | 1.003 |
| Double .0003 | 298.394 → 3.025 | 1.691 |
| DQN .0001 | 182.231 → 1.066 | 1.095 |
| Double .0001 | 599.885 → 2.964 | 2.852 |

Hay capacidad de ajuste y transferencia local a `(s,a)` excluidos de la regresión.
No prueba separación perfecta entre estados terminales y vecinos no terminales:
las etiquetas son casi constantes, próximas a −100. La reserva procede del mismo
replay de entrenamiento del agente original; **no** es un test de generalización
externa ni aprendizaje de aterrizaje. Las copias se guardan con nombre
`supervised_diagnostic_only`, nunca como políticas de aterrizaje.

## 3. Interferencia de una actualización

Por checkpoint: 32 minibatches que contienen ambos grupos, excluyendo de ellos
los `(s,a)` terminales reservados. Se conservan sus proporciones: loss de cada
rama=`suma de pérdidas activas / B completo`, no media reponderada del grupo.
Se rechazaron batches vacíos de algún grupo solo en esta prueba, no al estimar
frecuencias de la sección 1. Los targets se calculan antes de bifurcar.

Cada rama comienza con pesos idénticos y copia independiente del mismo estado
Adam. Se controla también Adam nuevo, con el mismo LR original por modelo.
Se aplica clipping original y **una** actualización de toda la red. Se mide el
mismo MAE reservado antes/después. Positivo=empeora. Media ± SD de 32 minibatches:

| Red .0001 / Adam | Solo T | Solo N | Mixto |
|---|---:|---:|---:|
| DQN / nuevo | −0.184 ± .037 | +0.161 ± .051 | +0.147 ± .053 |
| DQN / guardado | +0.0248 ± .0034 | +0.0576 ± .0044 | +0.0560 ± .0050 |
| Double / nuevo | −0.957 ± .213 | +1.102 ± .014 | +1.097 ± .016 |
| Double / guardado | +0.127 ± .032 | +0.320 ± .005 | +0.318 ± .007 |

Con Adam nuevo, T mejora y N/mixto empeoran en los 32/32 casos de ambos modelos.
Con Adam guardado incluso T empeora: los momentos previos importan. Reiniciar
Adam no es una solución deducible: el mixto con Adam nuevo empeora más aquí.
Los modelos .0003 son un contraste: el mixto reduce MAE con ambos estados Adam
(DQN: −.147/−1.160; Double: −.034/−1.515, guardado/nuevo). No se extrapola una
interferencia local a todos los checkpoints anteriores.

Estas ramas parten de los checkpoints, no de la copia ya ajustada de la sección 2.
Demuestran aumento del error terminal actual, no una curva de olvido posterior a
500 pasos de ajuste. La variabilidad es entre minibatches, no entre semillas.

## 4. Dónde se concentra el problema y qué estado recibe r

Se analizaron **todas** las terminales de los cuatro replays: 6602. El tipo de
evento inferido desde `s′` coincide con los logs originales, y la última observación
almacenada coincide exactamente con la codificación del estado final original
(error máximo 0). Se comparó también cada transición con su predecesora del mismo
episodio: `prev.next_obs == s` en los 6602 casos.

| Red | Salidas por techo / terminales | MAE techo / impacto del cuerpo | % del error absoluto total en techo |
|---|---:|---:|---:|
| DQN .0003 | 1379/1700 | 220.19 / 90.22 | 87.9% |
| Double .0003 | 1384/1589 | 337.44 / 36.63¹ | 95.6% |
| DQN .0001 | 1177/1623 | 218.15 / 48.94 | 91.8% |
| Double .0001 | 743/1690 | 761.16 / 271.55 | 57.2% |

¹ El error firmado medio de ese grupo es +24.62; MAE se informa por separado.
Los CSV conservan ambas magnitudes para evitar cancelaciones.

La concentración no se limita al techo: Double .0001 tiene 621 salidas laterales
con error firmado medio +539.67. La acción 7 concentra mucho error en los modelos
.0001 (MAE 211.56/677.88), pero está asociada a las trayectorias de ascenso; no se
le atribuye causalidad ni se concluye que sea una acción físicamente incorrecta.
Las velocidades del cuartil superior se asocian a mayor MAE; **la mayor norma de
observación no**: su cuartil superior tiene menor MAE en los cuatro modelos.
No hay evidencia de que todo se reduzca a entradas de magnitud extrema.

La penalización corresponde a **Q(s,a) antes del evento**, no a Q de la última
observación `s′`. Por ejemplo, primera salida de techo con misma acción que la
anterior en Double .0001, reevaluada con el checkpoint final:
`Q(s_previo,a)=604.58`, recompensa previa=−.192;
`Q(s,a)=630.32`, recompensa terminal=−100.135; altura `s`=23.572 m y altura final
`s′`=23.997 m. Son estados cercanos pero distintos; el límite lo cruzan las
esquinas del vehículo. Se conserva el salto de etiquetas, no se reasigna al reset.

## 5. Interpretación y única intervención propuesta

La incapacidad de ajustar terminales aisladas no está respaldada. Sí hay
competencia de gradientes e incrementos de error por updates N/mixtos en los
checkpoints .0001, además de efecto de momentos Adam. El problema tiene estructura
espacial, especialmente junto al techo, pero no se reduce a una única región.
Son mediciones de cuatro snapshots finales, no una prueba causal completa de la
inestabilidad a lo largo del entrenamiento.

**Única propuesta posterior: variar solo beta de `F.smooth_l1_loss`, de 1 a 10**,
en un experimento nuevo desde cero que conserve los demás parámetros del baseline
LR=.0001. No se ha implementado ni ejecutado.

Para residuo `e=Q−target`, la derivada de SmoothL1 es
`clip(e/beta,−1,1)`. Beta=10 **no aumenta** la derivada de residuos ≥10: siguen en
±1. Reduce diez veces la de residuos <1 y reduce las de 1–10 de ±1 a `e/10`.
En estas muestras, 98.5–100% de T tenían residuo ≥10, mientras solo 0.05–9.74%
de N alcanzaban 10. Hay fundamento para probar un cambio relativo de dirección
frente al dominio N medido, pero no garantía: también cambia la escala total y
la interacción con Adam/clipping. No es simplemente “dar más peso a errores grandes”.
No debe confundirse con cambiar a `HuberLoss(delta=10)`, cuya escala es distinta.
El criterio futuro sería reducción sostenida del error terminal sin deteriorar
control, no solo menor Q. No se propone reiniciar Adam ni cambiar simultáneamente
replay, recompensa, clipping o distribución.

## Reproducción y verificaciones

Protocolo previo: `configs/offline_interference_protocol.json`. Datos, índices,
splits, curvas y copias están en `artifacts/offline_interference_v1/`. CSV breves:
`docs/offline_contributions.csv`, `offline_interference.csv` y
`offline_supervised_fit.csv`. Los CSV completos incluyen medias, SD, percentiles,
valores por muestra, eventos y pares de estados. Figuras: `supervised_capacity.png`,
`one_step_interference.png`, `terminal_regions.png`.

```bash
source .venv/bin/activate
python -m pytest tests/test_offline_diagnostics.py
python scripts/diagnose_offline_interference.py --output artifacts/offline_interference_repeat
python scripts/report_offline_interference.py artifacts/offline_interference_repeat
```

Dos tests nuevos pasaron: identidad de gradiente ponderado/coseno indefinido y
copias de pesos/Adam independientes con denominador B correcto. Se verificaron
splits sin duplicados `(s,a)`, targets terminales exactos, gradientes objetivo
nulos y hashes originales antes/después. Presupuesto real: **2000 actualizaciones
de regresión** y **768 actualizaciones unitarias de copias**, 0 pasos de entorno.
El cálculo principal tardó 6.22 s en CPU, aparte de figuras y verificaciones.
No se ejecutó Huber beta=10, ninguna campaña ni prueba final.
