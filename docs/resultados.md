# Resultados reales — Etapa 2

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

Configuración `configs/pid.json`, física y éxito originales, CPU Python 3.12.12.
Se ajustó en validación (31/32 éxitos); luego se evaluaron 100 escenarios reservados
sin cambiar parámetros. PID es determinista: las semillas varían escenarios,
no entrenamientos independientes.

| Condición | Aterrizajes / n | Tasa observada | Impulso medio, todos (N s) | Duración media, todos (s) | Error horizontal medio en éxitos (m) |
|---|---|---|---|---|---|
| Vertical sin viento | 25/25 | 100% | 6510.70 | 6.629 | 0.000 |
| Desplazamiento pequeño | 24/25 | 96% | 6401.82 | 6.511 | 0.085 |
| Inclinación moderada | 24/25 | 96% | 6270.33 | 6.383 | 0.074 |
| Viento moderado | 25/25 | 100% | 6257.44 | 6.370 | 0.132 |

Total: **98/100 aterrizajes**. Los cien episodios tuvieron contacto; no hubo salida
del área, límite temporal, agotamiento del gas, contacto del cuerpo ni fallo por
quedar fuera de plataforma. Consumo y duración incluyen los dos fallos.
Son tasas muestrales para este manifiesto, no garantías ni evidencia de superioridad
sobre otros algoritmos. Los agentes RL se implementaron después en Etapa 3;
su prueba corta y conjuntos separados están documentados en `dqn.md`.

## Fallos conservados

| Escenario | Semilla | Resultado | vy al contacto (m/s) | θ (rad) | ω (rad/s) |
|---|---|---|---|---|---|
| test-offset-000 | 9100 | invalid_contact | -0.30435 | -0.00246 | -0.47788 |
| test-tilted-001 | 9201 | invalid_contact | -0.57026 | -0.00314 | +0.45683 |

En ambos se superó |ω|≤0.25 rad/s; los demás umbrales se cumplieron. El paso de
momento discreto de 0.477876 rad/s por pulso explica esta limitación. No se ajustó
el controlador con estos fallos: cualquier cambio posterior necesita otra prueba
reservada para sostener una nueva afirmación de generalización.

## Archivos

- `artifacts/pid_validation.csv`, `.summary.csv`, `.traces.csv`, `.metadata.json`.
- `artifacts/pid_test.csv`, `.summary.csv`, `.traces.csv`, `.metadata.json`.
- `artifacts/pid_test_plots/trajectories.png`, `states.png`, `resources.png`.
- `docs/pid_test_summary.csv`: copia pequeña del resumen para conservarlo en Git.

El resumen separa medias sobre todos, contactos y éxitos, con sus tamaños de
muestra; estados finales sin contacto no se rellenan como aterrizajes. Los archivos
generados de `artifacts/` no se versionan. Comandos, ajuste, limitaciones y
presupuesto analítico de gas en [pid.md](pid.md).

Se mantienen las aproximaciones de Etapa 1: masa/inercia constantes, gas como
impulso disponible, viento constante, contacto terminal sin asentamiento,
integración en subpasos y detector de cruce discreto.

## Etapa 3: prueba corta RL

Una semilla por algoritmo, 10000 pasos cada uno: DQN 0/32 y Double DQN 0/32 éxitos
en validación al paso 10000; ninguno aterrizó durante entrenamiento. No se ejecutó
la prueba final ni una campaña de semillas. Esta evidencia verifica integración
y estabilidad numérica, no aprendizaje suficiente ni superioridad. Detalles y
limitaciones en `dqn.md`; resumen versionado en `rl_smoke_summary.csv`.

## Piloto de 100000 pasos (experimento nuevo)

Ambos algoritmos se entrenaron desde cero con epsilon más lento y semilla 42:
0/32 éxitos en todos los puntos de validación y 0 éxitos de entrenamiento. Al final,
32/32 salidas por el techo en cada uno, sin contactos. Q presentó deriva optimista
a pesar de ser finito. Resultados completos, trayectorias iniciales/finales,
checkpoints mejor/último, diagnóstico y tiempos en `piloto_100k.md`. No se abrió
prueba final ni se lanzó la campaña de tres semillas.

## Ablación controlada LR=0.0001

Experimento nuevo desde cero, semilla 42 y 100000 pasos por algoritmo. Auditoría
previa sin defectos de aprendizaje detectados. Cero éxitos de entrenamiento y
validación; ambos agentes terminaron con 32/32 salidas por techo. Diagnóstico
sobre la misma muestra fija de replay en todos los checkpoints: DQN redujo parte
del error terminal, Double DQN lo aumentó; persiste la inestabilidad. Q menor no
se interpreta como mejor aprendizaje. Ver `ablacion_lr1e4.md` y sus CSV firmados.
No se abrió prueba final ni se ejecutó una segunda ablación.

## Ablación única SmoothL1 beta=10

Con LR=.0001, semilla 42 y 100000 pasos por algoritmo desde cero, cambiando solo
beta, se observó menor deriva optimista y menor error terminal en sondas fijas.
No hubo éxitos de entrenamiento ni validación: DQN terminó con 32 salidas por
techo y Double DQN con 27 contactos fuera de plataforma y 5 inválidos, tras agotar
gas. Hay mejora numérica y control parcial, sin aterrizaje aprendido. La loss a
beta distinto no se compara como evidencia suficiente. Ver `ablacion_beta10.md`.
No se amplió campaña ni se evaluó prueba final; los baselines se preservaron.

## Estado consolidado y currículo

El piloto posterior de currículo se detuvo tras 50 000 pasos por agente, con
cero aterrizajes de entrenamiento y de validación. La validación final contiene
128 casos por agente (32 originales más 32 por nivel); todos fallaron. No se
entrenaron niveles 2/3 ni se ejecutó prueba final RL. Ver
[piloto_curriculum.md](piloto_curriculum.md) y la
[selección compacta de resultados](resultados/README.md), que resume también los
experimentos fallidos y distingue archivos publicados de registros solo locales.
