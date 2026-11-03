# Piloto con dificultad progresiva — semilla 42

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

**Resultado: parada al terminar el nivel 1.** DQN y Double DQN ejecutaron
50 000 pasos cada uno, con cero aterrizajes en entrenamiento y validación.
Se aplicó la condición predefinida; no se entrenaron los niveles 2 y 3.
No se ejecutó la prueba final ni una campaña de varias semillas.

Experimento: `curriculum_beta10_lr1e4_seed42_v1`. Los artefactos están en
`artifacts/curriculum_beta10_lr1e4_seed42_v1/` (archivo local excluido).
Los experimentos anteriores, física, recompensa, catálogo, umbrales, gas y PID
permanecen intactos. La integridad se verificó contra
`configs/reference_before_curriculum.json`; las dos fuentes de infraestructura
modificadas tienen copias anteriores en `artifacts/curriculum_reference/`.
`curriculum_execution_manifest.json` conserva los hashes del protocolo,
escenarios, configuraciones y código efectivamente ejecutados.

## Protocolo fijado antes del entrenamiento

Rangos uniformes independientes, unidades SI; altura del **centro**, no del apoyo.
Viento cero y gas completo de 18 000 N·s en todos los reinicios.

| Nivel | x (m) | y (m) | vx (m/s) | vy (m/s) | θ (rad) | ω (rad/s) |
|---|---|---|---|---|---|---|
| 1 | ±0,10 | 1,6–2,4 | ±0,03 | −0,30 a −0,05 | ±0,01 | ±0,005 |
| 2 | ±0,40 | 4–7 | ±0,10 | −0,70 a 0 | ±0,04 | ±0,01 |
| 3 | 0 | 10–12 | 0 | −1 a 0 | 0 | 0 |

El nivel 3 recupera la distribución **real del manifiesto de entrenamiento RL**,
que era vertical y centrada, aunque los valores predeterminados del entorno
permitan más perturbaciones. Por ello el nivel 3 no es un superconjunto del 2:
restaura exactamente la referencia original, con mayor altura y sin perturbaciones laterales. Cada nivel tiene 128 escenarios nuevos de
entrenamiento y 32 de validación: semillas 61000/62000/63000 y
71000/72000/73000, respectivamente, más el índice del caso. Se verificaron
identidades y estados disjuntos; los 32 casos originales se conservan solo
para validación. El manifiesto de prueba final se lee únicamente para excluir
identidades, sin ejecutar esos episodios.

Las menores separaciones geométricas del suelo en los manifiestos son
0,599, 2,997 y 9,015 m. No hay contacto ni éxito inicial. Con masa 100 kg,
la aceleración máxima vertical es aproximadamente 20,19 m/s². Considerando
una decisión de demora de 0,05 s, desde vy=−0,3 m/s se necesitan unos
0,043 m para demorar y frenar; desde −1 m/s, unos 0,117 m. Son márgenes
inferiores a las separaciones disponibles. Esto demuestra viabilidad local,
no estabilidad de una política cualquiera. El PID consumió en promedio
2209, 3806 y 6418 N·s, por debajo del presupuesto, en esta comprobación.

| Control, 32 casos nuevos por nivel | Nivel 1 | Nivel 2 | Nivel 3 |
|---|---:|---:|---:|
| PID congelado | 32 éxitos | 31 éxitos, 1 contacto inválido | 32 éxitos |
| Aleatorio uniforme | 0 | 0 | 0 |
| Apagado | 0 | 0 | 0 |

Apagado impactó a vy media −4,38/−9,60/−14,09 m/s. El nivel 1 no permite
aterrizar automáticamente en estos casos. Cero éxitos aleatorios en 32 casos
no demuestra probabilidad nula: la comprobación es pequeña.
Datos: `artifacts/curriculum_preflight/episodes.csv` (archivo local excluido).
Estos rollouts no aportaron transiciones al replay ni demostraciones a los agentes.

Calendario máximo idéntico: 50k de nivel 1; 100k con reinicios 80% nivel 2 y
20% nivel 1; 150k con 80% nivel 3, 10% nivel 2 y 10% nivel 1. Las mezclas se
aplican al siguiente reinicio, sin terminar artificialmente un episodio al cruzar
una frontera. Se conservan red, Adam, replay y RNG; `source_levels` registra el
origen de cada transición, incluido en muestreo y checkpoints. Cada 10k se guarda
el replay completo: permite reconstruir el origen de todas las transiciones,
además del registro por episodio. La pausa a 50k conserva también el episodio vivo.

LR=0,0001; `SmoothL1Loss(beta=10)` sin compensación; Adam, gamma=0,99,
MLP 128/128 ReLU, replay uniforme 50k, batch 64, warmup 1000, actualización
cada 4 pasos, copia objetivo cada 500 desde warmup y clipping de norma 10.
Epsilon permanece 1 hasta 1000 y baja a 0,1 en el paso 81000: a 50k es 0,44875.
Solo cambian distribución inicial y presupuesto máximo frente a beta=10 anterior.

Evaluación greedy a 0/10k/20k/30k/40k/50k: los mismos 128 casos, sin actualizar
pesos ni explorar. El mejor checkpoint usa éxito en los **32 originales**, luego
menor |x| entre éxitos y, si persiste el empate, el primero. También se conservan
mejores políticas separadas por nivel. Todos los mejores quedaron en 10k por empate;
el estado final es `last.*`, a 50k. La regla conjunta de parada está en
`artifacts/curriculum_beta10_lr1e4_seed42_v1/level1_gate.json` (archivo local excluido).

## Resultados reales

| Agente | Episodios completos de entrenamiento | Éxitos | Fuera de área / cuerpo / fuera de plataforma / contacto inválido | Duración media (s) | Consumo medio (N·s) |
|---|---:|---:|---|---:|---:|
| DQN | 757 | 0 | 640 / 94 / 16 / 7 | 3,274 | 4327 |
| Double DQN | 731 | 0 | 600 / 103 / 20 / 8 | 3,384 | 4433 |

Retorno medio no descontado/descontado de entrenamiento: DQN −103,73/−57,16;
Double DQN −103,83/−56,87. Los componentes de recompensa se conservan en los CSV.
Cada agente realizó 12 251 actualizaciones y 99 copias objetivo. Quedaron episodios
parciales de 59 y 161 pasos; no se cuentan como episodios terminados ni aterrizajes.
Todas las 1536 evaluaciones de episodios (incluidos los iniciales y repeticiones de
casos fijos) fallaron. No son 1536 condiciones iniciales independientes.

Validación al paso 50k; las magnitudes de contacto excluyen episodios sin contacto:

| Agente | Conjunto | Éxitos / 32 | Contactos | vy media (m/s) | |x| medio (m) | |θ| media (rad) | Duración media (s) | Consumo medio (N·s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | Nivel 1 | 0 | 32 | −15,143 | 0,501 | 0,016 | 19,862 | 18000 |
| DQN | Nivel 2 | 0 | 32 | −16,097 | 0,489 | 0,042 | 19,939 | 18000 |
| DQN | Nivel 3 | 0 | 14 | −18,017 | 0,129 | 0,007 | 13,518 | 12599 |
| DQN | Original | 0 | 14 | −18,015 | 0,220 | 0,008 | 13,610 | 12673 |
| Double DQN | Nivel 1 | 0 | 32 | −13,324 | 0,537 | 0,174 | 19,689 | 18000 |
| Double DQN | Nivel 2 | 0 | 32 | −13,384 | 0,533 | 0,241 | 19,674 | 18000 |
| Double DQN | Nivel 3 | 0 | 32 | −13,595 | 0,576 | 0,106 | 19,686 | 18000 |
| Double DQN | Original | 0 | 32 | −13,508 | 0,559 | 0,191 | 19,688 | 18000 |

Todos esos contactos fueron inválidos; las otras 18 salidas de DQN en cada conjunto
alto fueron por el techo. No hubo límites temporales en la validación final.
En el nivel 1, todos los contactos están dentro de |x|≤1,5 y todos incumplen vy≥−1.
DQN cumple los demás límites cinemáticos; Double DQN incumple además θ en 11/32
casos y ω en 8/32. No se confunde la posición final de las salidas con precisión de
aterrizaje. Los CSV desglosan también vx, ω, componentes, duración y consumo por resultado.

Ambos agentes ascienden desde el nivel 1 hasta alturas máximas medias de 13,00 y
13,68 m, agotan gas cerca de t=18,34/18,40 s a alturas medias de 12,68/10,07 m y caen.
Han mejorado la orientación respecto a la política inicial, pero no descubrieron
el frenado final. Retrasar una penalización terminal puede mejorar su componente
descontado: a 400 decisiones, −100·0,99^399≈−1,81, frente a aproximadamente −93
tras ocho decisiones. Es una explicación compatible con las trayectorias, no una
prueba de que esa sea la única causa. No se cambió gamma ni recompensa.

## Diagnóstico numérico y referencia anterior

Muestra común A: exactamente las 64 transiciones terminales y 64 no terminales
anteriores en cada checkpoint. Se conserva también el holdout terminal D anterior.
Fuente C: 128 minibatches uniformes de 64 del replay nuevo; fuente B: replay beta=10
anterior. Ambos modelos se miden en ambas fuentes; índices e identidades se guardan.
Los CSV contienen Q, recompensa, target, TD=target−Q, bootstrap y error terminal Q−r,
con medias y percentiles firmados. Las fuentes no se mezclan.

Fuente C, modelos al paso 50k (estadísticos por muestra, no media de medias de grupos):

| Agente / grupo | Muestras / 8192 | Q medio | Target medio | TD medio | Loss media | Residuos en región lineal |
|---|---:|---:|---:|---:|---:|---:|
| DQN terminal | 129 | −56,450 | −100,059 | −43,609 | 41,498 | 78,3% |
| DQN no terminal | 8063 | −10,570 | −10,374 | +0,197 | 0,032 | 0% |
| Double DQN terminal | 104 | −62,890 | −100,062 | −37,172 | 36,948 | 76,0% |
| Double DQN no terminal | 8088 | −12,890 | −12,590 | +0,300 | 0,046 | 0% |

Frecuencia terminal 1,57%/1,27%. Normas medias de la contribución ponderada al
gradiente compartido, incluyendo minibatches sin terminales: T/N=0,373/0,421
para DQN y 0,337/0,645 para Double DQN. Coseno medio donde ambas normas son
no nulas: −0,306 (DE 0,261) y −0,364 (DE 0,287). Se verificó g=pT·gT+pN·gN.
El error terminal por sí solo no mide su contribución al aprendizaje. Los errores
terminales locales siguen siendo grandes y hay oposición entre los gradientes.
No hay evidencia de explosión numérica: métricas finitas desde el inicio del
aprendizaje; normas medias por intervalo antes/después de clipping coinciden
(máximo de esas medias 1,380/1,486). Los valores vacíos del warmup son esperados.

En la muestra A, MAE terminal a 50k=32,05/32,44 y TD firmado terminal
−26,93/−26,01; no terminal +0,294/+0,222. Ese replay anterior tiene estados fuera de
la distribución actual: no debe interpretarse como error poblacional del nuevo replay.
En el holdout D, MAE DQN pasa de 62,34 (referencia 50k) a 69,65 y Double DQN de
26,57 a 35,26: no hay mejora uniforme de los errores terminales en todas las sondas.

En los originales, Q inicial greedy / retorno descontado / **brecha frente al retorno
de la política greedy**: DQN 3,379/−13,583/+16,962; Double DQN
4,996/−4,352/+9,348. No es un error exacto respecto a Q óptimo.

| Agente / ejecución | Pasos | Éxitos originales | Contactos | Salidas por techo | |vy| al contacto | |x| al contacto |
|---|---:|---:|---:|---:|---:|---:|
| DQN beta10 anterior | 50k | 0/32 | 13 | 19 | 19,832 | 1,660 |
| DQN currículo | 50k | 0/32 | 14 | 18 | 18,015 | 0,220 |
| DQN beta10 anterior | 100k | 0/32 | 0 | 32 | — | — |
| Double DQN beta10 anterior | 50k | 0/32 | 0 | 32 | — | — |
| Double DQN currículo | 50k | 0/32 | 32 | 0 | 13,508 | 0,559 |
| Double DQN beta10 anterior | 100k | 0/32 | 32 | 0 | 17,633 | 3,013 |

Hay aproximación a la plataforma y cambios de comportamiento, sin aterrizajes.
No se demostró aprender cerca de la plataforma; por tanto, tampoco retener esa
habilidad ni transferirla. No hubo exposición de entrenamiento a niveles posteriores.
La comparación con 100k cambia distribución **y** presupuesto; la comparación a 50k
igual presupuesto sigue teniendo una única semilla. Ninguna establece superioridad
general de algoritmos ni atribuye todo cambio únicamente al currículo.
Antes de otro entrenamiento conviene medir, como diagnóstico separado, la probabilidad
de descubrir un contacto válido al azar en subregiones del nivel 1 y revisar su ventana
de frenado: aquí nunca entró una transición de éxito en el replay. No se ejecutó ese
experimento adicional ni se propone ampliar automáticamente este piloto.

## Artefactos, tiempo y reproducción

`artifacts/curriculum_beta10_lr1e4_seed42_v1/report/learning_curves.png` (archivo local excluido),
`artifacts/curriculum_beta10_lr1e4_seed42_v1/report/fixed_trajectories.png` (archivo local excluido),
`artifacts/curriculum_beta10_lr1e4_seed42_v1/report/fixed_level1_states.png` (archivo local excluido),
`artifacts/curriculum_beta10_lr1e4_seed42_v1/report/fixed_td_comparison.png` (archivo local excluido).
El criterio de selección es el primer ID de cada conjunto, fijado antes del análisis;
se exportan además todas las trayectorias finales. No existen ejemplos exitosos que mostrar.
`report/` contiene tablas por resultado, métricas de contacto, origen de transiciones,
comparación, brecha greedy y tiempos; `diagnostics/` contiene muestras y gradientes.
En cada `*_level1/`: checkpoints cada 10k, `best` y `last`, políticas y reanudaciones
completas (Adam, dos redes, replay, RNG, runner y estado del entorno).

Tiempo de entrenamiento incluyendo validación y guardado: DQN 80,06 s (624,5 pasos/s),
Double DQN 70,44 s (709,8 pasos/s), total 150,50 s. Validación consumió 55,01/44,91 s;
guardado 2,00/1,83 s. Preflight, pruebas y análisis están fuera de esas medidas.
No se extrapola esto como tiempo medido de niveles que no se ejecutaron.

Desde la raíz, usando el entorno instalado:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_env.py
# Reproduce el protocolo completo en un workspace con esa salida aún inexistente:
.venv/bin/python scripts/train_curriculum.py
# Repetición aislada de nivel 1 en un directorio nuevo (Double DQN: cambiar ambos nombres):
.venv/bin/python scripts/train.py --config configs/dqn_curriculum.json --curriculum configs/curriculum_protocol.json --train-scenarios configs/curriculum_train_scenarios.json --validation-scenarios configs/curriculum_validation_scenarios.json --stop-steps 50000 --output artifacts/dqn_curriculum_repeticion
# Reanudar hasta la misma parada desde 40k; no continuar más allá de 50k con esta evidencia:
.venv/bin/python scripts/train.py --config configs/dqn_curriculum.json --curriculum configs/curriculum_protocol.json --train-scenarios configs/curriculum_train_scenarios.json --validation-scenarios configs/curriculum_validation_scenarios.json --resume artifacts/curriculum_beta10_lr1e4_seed42_v1/dqn_level1/step_00040000.resume.pt --stop-steps 50000 --output artifacts/dqn_curriculum_reanudado
# Evaluar sin gráficos; agregar --render para reproducción visual local:
.venv/bin/python scripts/evaluate.py --checkpoint artifacts/curriculum_beta10_lr1e4_seed42_v1/dqn_level1/last.policy.pt --scenarios configs/curriculum_validation_scenarios.json --trace --output artifacts/curriculum_reproduccion.csv
# Repetir comprobación PID/aleatorio/apagado en una salida nueva:
.venv/bin/python scripts/check_curriculum.py --output artifacts/curriculum_preflight_repeticion
# Repetir las sondas y gradientes, sin actualizar agentes:
.venv/bin/python scripts/diagnose_curriculum.py --output artifacts/curriculum_diagnostico_repeticion
# Regenerar tablas/curvas (sin entrenamiento):
MPLCONFIGDIR=/tmp/curriculum-mpl .venv/bin/python scripts/report_curriculum.py
MPLCONFIGDIR=/tmp/curriculum-mpl .venv/bin/python scripts/inspect_curriculum.py
```

Los scripts de preflight y diagnóstico rechazan directorios de salida ya existentes
para conservar sus evidencias. El script de entrenamiento también rechaza salidas
existentes. Las configuraciones JSON son `dqn_curriculum.json`,
`double_dqn_curriculum.json` y `curriculum_protocol.json`.
Pruebas: **91 aprobadas**; checker Gymnasium válido (dos avisos conocidos de Box
con límites infinitos). Las nuevas pruebas cubren calendario/mezclas, parada conjunta,
procedencia circular y compatibilidad de replays antiguos, cambios permitidos,
reinicios sin contacto y reanudación exacta cruzando una frontera sin reset artificial.
No se verificó una ventana real ni un teclado físico en esta ejecución.
