# Ablación: imitación sola frente a imitación más TD — 9 de noviembre de 2026

**Las actualizaciones TD y de margen no mostraron beneficio sobre la imitación sola.** En un
complemento nuevo de 288 casos, la política de solo entropía cruzada obtuvo 279/288 con
semilla 42 frente a 276/288 de la candidata asistida equivalente, y 270/288 frente a 276/288
con semilla 43. Ninguna de las tres comparaciones emparejadas es distinguible del azar
(McNemar exacto, p = 0,629 / 0,152 / 0,345). Esto cierra el pendiente declarado en
[evaluacion_alcance.md](evaluacion_alcance.md) y
[trazabilidad_alcance.md](trazabilidad_alcance.md): ahora hay una medición, no una ausencia.
No demuestra que TD sea inútil en general, ni que estas tasas se mantengan con más semillas.

Experimento `imitation_ablation_v1`. No se reutilizó el conjunto final ya consumido, no se
reentrenaron ni reseleccionaron las candidatas entregadas, y no se modificó recompensa,
física, acciones, gas, umbrales ni PID. Cero actualizaciones durante la evaluación.

## Por qué la comparación es una ablación y no dos métodos distintos

El entrenamiento asistido tiene tres fases: 10 000 actualizaciones de entropía cruzada sobre
acciones del PID, 10 000 de TD offline a uno y diez pasos más margen, y después TD y margen
online mezclando demostraciones permanentes con replay propio. La familia de imitación sola
repite **exactamente** la primera fase y se detiene ahí: misma configuración, misma semilla,
mismas demostraciones y mismo orden de minibatches. Los pesos resultantes son los que la
candidata asistida tenía al terminar de imitar. La única diferencia entre las dos familias
son las actualizaciones que vienen después.

Que la reproducción sea fiel se comprueba con tres cifras ya publicadas, reproducidas aquí
desde cero en Windows:

| Magnitud reproducida | Registro original | Esta ejecución |
|---|---:|---:|
| Episodios de demostración PID | 1024 | 1024 |
| Episodios exitosos retenidos | 1005 | 1005 |
| Transiciones docentes | 84 882 | 84 882 |
| Decisiones físicas de la recolección | 87 099 | 87 099 |
| Desarrollo tras clasificación, V2 | 119/128 | 119/128 |

`run_imitation_only.py` aborta si las demostraciones no coinciden con el registro original,
porque entonces la ablación no sería comparable.

**Los pesos de imitación sola no dependen del algoritmo.** La entropía cruzada no usa
bootstrap, red objetivo ni `argmax`, así que `imitation_only_dqn42` e
`imitation_only_double42` son el mismo modelo: el hash de sus pesos online coincide
(`9de8aa1e42b7ae21`). Se conservan como dos entradas para emparejar cada candidata asistida
con su origen, pero son **dos modelos distintos, no tres**: semilla 42 y semilla 43. La fila
de Double DQN comparte el lado de imitación con la de DQN y no aporta una muestra
independiente.

## Conjunto y protocolo congelados antes de evaluar

`configs/imitation_ablation_v1_protocol.json` registra los SHA-256 de checkpoints,
configuraciones, manifiesto, código físico y scripts, y se verifica antes de cada método y
entre métodos. `configs/imitation_ablation_v1_complement.json` tiene 288 casos construidos
igual que el complemento final —los mismos cinco grupos y el mismo emparejamiento de viento
0, ±20 y ±40 N— con base de semillas 1 440 000 y verificación de disjunción contra todos los
manifiestos previos, incluido el complemento ya consumido. Se repitió su comprobación
independiente de holgura geométrica. Ningún caso se descartó por resultado.

La evaluación usa el **mismo evaluador común** que la prueba final
(`scope_evaluation.evaluate_method`), sin modificarlo: bloquea `PIDController.act` durante la
inferencia neuronal y verifica que pesos, optimizador, replay, contadores y RNG no cambien.
Todos los métodos ven los mismos casos, el mismo gas inicial, el mismo período de decisión y
los mismos criterios de contacto. 2016 episodios, 237 259 pasos de entorno, 165 s en esta CPU.

## Resultados

| Método | Familia | Éxitos / 288 | Fallos |
|---|---|---:|---:|
| PID congelado | referencia clásica | **278** | 10 |
| DQN asistido 42 | imitación + TD | 276 | 12 |
| Double DQN asistido 42 | imitación + TD | 271 | 17 |
| DQN asistido 43 | imitación + TD | 276 | 12 |
| Imitación sola, semilla 42 | solo imitación | **279** | 9 |
| Imitación sola, semilla 42 (par de Double) | solo imitación | 279 | 9 |
| Imitación sola, semilla 43 | solo imitación | 270 | 18 |

Por grupo, con 32 casos cada uno:

| Grupo | PID | DQN as.42 | Double as.42 | DQN as.43 | Imit.42 | Imit.43 |
|---|---:|---:|---:|---:|---:|---:|
| represented_l1 | 32 | 32 | 28 | 30 | 32 | 30 |
| represented_high | 31 | 31 | 32 | 31 | 32 | 30 |
| new_displacement | 31 | 32 | 26 | 31 | 31 | 29 |
| new_dynamics | 31 | 31 | 31 | 28 | 30 | 29 |
| wind_control | 30 | 30 | 31 | 32 | 32 | 31 |
| wind_m20 | 30 | 31 | 32 | 31 | 32 | 31 |
| wind_p20 | 31 | 32 | 31 | 31 | 30 | 29 |
| wind_m40 | 31 | 28 | 28 | 31 | 30 | 31 |
| wind_p40 | 31 | 29 | 32 | 31 | 30 | 30 |

**Los 97 fallos de los siete métodos son `invalid_contact`.** No hubo impactos de cuerpo,
salidas de plataforma, salidas del área ni límites temporales para ningún método. Todas las
políticas, clásicas y neuronales, llegan a tocar la plataforma y fallan por no cumplir
alguna ventana cinemática al contacto.

## Comparaciones emparejadas por caso

Las tasas agregadas no bastan: los métodos fallan en casos distintos. La tabla cuenta, sobre
los mismos 288 estados iniciales, en cuántos aterriza uno y no el otro. El valor p es McNemar
exacto de dos colas sobre los pares discordantes.

| Comparación | Solo el primero | Solo el segundo | Discordantes | p |
|---|---:|---:|---:|---:|
| DQN asistido 42 frente a imitación 42 | 7 | 10 | 17 | 0,629 |
| Double asistido 42 frente a imitación 42 | 8 | 16 | 24 | 0,152 |
| DQN asistido 43 frente a imitación 43 | 17 | 11 | 28 | 0,345 |
| PID frente a imitación 42 | 8 | 9 | 17 | 1,000 |
| PID frente a DQN asistido 42 | 10 | 8 | 18 | 0,815 |
| Imitación 42 frente a imitación 43 | 16 | 7 | 23 | 0,093 |

Dos lecturas, ambas con el mismo peso:

1. **TD no se distingue.** En dos de las tres comparaciones la imitación sola gana más casos
   discordantes que el asistido, y en la tercera pierde. Sumando las tres, el asistido gana
   32 casos discordantes y la imitación 37. Ningún par alcanza significancia.
2. **La variación entre semillas es igual o mayor que la variación entre familias.** Las dos
   políticas de imitación sola difieren entre sí en 23 casos discordantes, más que cualquier
   par asistido/imitación. Con una semilla principal por configuración no se puede atribuir
   una diferencia de tres o seis aterrizajes a la presencia de TD.

## Lo que esto sí y no permite afirmar

- Sí: en los conjuntos medidos, el controlador neuronal funcional se explica por la
  imitación del PID. Las actualizaciones TD y de margen no aportaron una mejora medible.
- Sí: las tres candidatas entregadas y las dos de imitación aterrizan en todos los grupos,
  incluido viento no visto en entrenamiento, con fallos residuales del mismo tipo que el PID.
- No: no se concluye que TD perjudique. Las diferencias observadas son compatibles con ruido
  de semilla y el presupuesto online fue de 30 000 decisiones por candidata, no una campaña.
- No: no se evaluó una familia de imitación con más episodios docentes, otra arquitectura u
  otro profesor, ni se repitió la ablación con tres o más semillas por configuración.
- No: no se reabrió el conjunto final `scope_final_v1_complement.json`. Las cifras de este
  documento pertenecen a un complemento distinto y no sustituyen la prueba ya ejecutada;
  las dos no se promedian ni se comparan caso por caso.
- La tasa de imitación 42 superior al PID por un aterrizaje no es evidencia de superioridad:
  su comparación emparejada da p = 1,000.

Consecuencia para el informe: la frase «no se aisló el beneficio de las actualizaciones TD
frente a la imitación por sí sola» puede sustituirse por un resultado medido. El proyecto
pasa de tener un hueco declarado a tener un resultado negativo cuantificado, que es una
contribución legítima y más difícil de objetar.

## Reproducción

Desde la raíz, con las dependencias instaladas en `.venv`. Los directorios deben ser nuevos
y el protocolo no se puede sobrescribir.

```sh
# Demostraciones PID y tres entrenamientos de solo clasificacion, presupuesto propio.
python scripts/run_imitation_only.py --output artifacts/imitation_only_v1

# Congelar complemento y protocolo. Rechaza sobrescribir si ya existen.
PYTHONPATH=scripts python scripts/prepare_imitation_scope.py --imitation-root artifacts/imitation_only_v1

# Evaluacion comun, una sola vez, sin entrenar ni reseleccionar.
PYTHONPATH=scripts python scripts/evaluate_imitation_scope.py --output artifacts/imitation_ablation_v1/final
```

`run_imitation_only.py` crea su propio `budget.json`: el de `artifacts/recovery_v1` está
cerrado con `status: complete` y cualquier intento de consumir pasos ahí falla a propósito.
Presupuesto de esta ablación: **120 154 decisiones de 1 000 000**, de las cuales 87 099 son
la recolección de demostraciones y el resto las evaluaciones de desarrollo. Las tres fases de
clasificación no consumen pasos de entorno: son 10 000 actualizaciones supervisadas cada una,
entre 27 y 48 s en esta CPU.

Salidas: `episodes.csv` con estado inicial y final, causa, componentes de recompensa y
métricas de contacto ausentes cuando no hubo contacto; `summary.csv` por método, grupo y
resultado; `paired_families.csv` con las discordancias; `totals.csv` y `execution.json` con
la auditoría por método (consultas al profesor en cero, cero actualizaciones, estado del
agente sin cambios).

## Entorno de esta ejecución

Windows 11, Python 3.12.11, CPU. PyTorch 2.15.0+cpu, NumPy 2.5.4, Gymnasium 1.4.0,
pandas 3.1.1, Matplotlib 3.11.2. **118 pruebas pasaron** y el checker oficial de Gymnasium
fue válido, con los dos avisos conocidos de cotas infinitas del Box. Se reprodujo además la
prueba final congelada completa —seis métodos, 388 casos, 2328 episodios, 242 465 pasos— y
las doce tasas coincidieron con [evaluacion_alcance.md](evaluacion_alcance.md).

Nota de portabilidad encontrada al reproducir: `scripts/evaluate_scope.py` busca el hash del
protocolo con `record[str(a.protocol)]`, y `protocol_sha256.json` guarda la clave con
separador posix. En Windows `str(Path(...))` usa `\` y el script termina en `KeyError` antes
de evaluar. El archivo está dentro de los SHA-256 del protocolo congelado, así que no se
corrigió: hacerlo invalidaría la verificación. La reproducción se ejecutó con un controlador
externo equivalente que importa y llama a las mismas funciones y repite la misma comprobación
buscando la clave en forma posix. Para una versión futura, guardar y leer las rutas con
`as_posix()`.
