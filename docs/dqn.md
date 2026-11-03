# Etapa 3: DQN y Double DQN

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

## Referencia congelada y conjuntos

Antes de implementar se leyeron los CSV reales de PID: 100 episodios, 98 éxitos,
con resumen idéntico a `docs/pid_test_summary.csv`. No se volvieron a ejecutar sus
escenarios de prueba. `configs/reference_stage2.json` fija hashes SHA256 de PID,
física, recompensa, configuración, manifiestos y resultados disponibles.
`verify_reference()` rechaza cambios en archivos presentes; permite que falten
solo artefactos ignorados por Git, dejándolo registrado. El resumen PID versionado
sigue siendo obligatorio. Entorno: `ThrusterLanding-v0`, paquete `0.1.0`.
No se cambió ningún archivo de `envs/`, parámetro físico, recompensa, acción ni
umbral de éxito.

Conjuntos materializados por `scripts/make_rl_scenarios.py`, que no sobrescribe:

- Entrenamiento: `rl_train_scenarios.json`, 128 estados verticales sin viento,
  semillas 20000–20127; x=vx=θ=ω=0, y U(10,12) m, vy U(-1,0) m/s.
- Validación: `rl_validation_scenarios.json`, semillas 30000–30031. Dieciséis casos
  verticales independientes y dieciséis con x±0.2 m, vx±0.05 m/s y θ±0.02 rad.
- Prueba final PID: permanece reservada. Los manifiestos PID se leen solo para
  excluir coincidencias de semilla o de opciones de estado/viento, nunca para
  recolectar transiciones o ajustar RL.

El entrenador exige los splits train/validation, comprueba separación contra
ambos manifiestos PID y exige la configuración física/recompensa congelada.
Ambos algoritmos muestrean uniformemente con reemplazo los mismos 128 casos,
usando un RNG de escenarios independiente. No se usó currículo, demostraciones,
PID como política de exploración ni imitación. Las redes comienzan con pesos
aleatorios equivalentes, sin checkpoints previos. La validación no entra al replay.
Los scripts RL rechazan manifiestos de prueba final en esta etapa.

## Algoritmos compartidos

`agents/dqn.py`: MLP 9 → 128 ReLU → 128 ReLU → 8. Una salida Q por acción.
`agents/replay.py`: buffer circular, minibatches uniformes sin reemplazo dentro
del batch. Copia las observaciones para evitar aliasing; almacena por separado
terminated y truncated. Red objetivo congelada (`requires_grad=False`) y copia
dura periódica desde la principal. Adam, Huber (`smooth_l1_loss`, beta=1) y
recorte de norma global de gradiente a 10. Se registran la norma **anterior** al
recorte y las magnitudes medias de Q de acciones muestreadas, no de todas las salidas.

Para cada transición:

- DQN: `y = r + gamma * (1-terminated) * max_a Q_target(s_next,a)`.
- Double DQN: `a_star = argmax_a Q_online(s_next,a)` y
  `y = r + gamma * (1-terminated) * Q_target(s_next,a_star)`.

La selección/evaluación del siguiente estado se realiza bajo `torch.no_grad()`.
`compute_targets` también está decorada sin gradientes. Solo la red principal
actualiza parámetros. Un estado terminal no hace bootstrap. Un límite temporal
artificial sí: se usa la observación real final de la transición, antes de reset;
no hay wrappers con autoreset. Esto interpreta los 25 s como límite del recolector,
no como horizonte físico finito. El tiempo no forma parte de la observación actual.
Si se quisiera un objetivo de horizonte finito habría que cambiar la formulación,
lo cual queda fuera de esta entrega.

La diferencia entre algoritmos es exclusivamente la regla de target.
Los archivos de configuración son iguales salvo `algorithm` y las rutas de salida
se pasan por CLI. Se comprueba esa equivalencia en tests, incluida inicialización
idéntica de ambas redes con la misma semilla.

## Configuración inicial, no óptima

`configs/dqn.json` y `configs/double_dqn.json` exponen:

| Parámetro | Valor |
|---|---|
| seed / device / torch_threads | 42 / cpu / 1 |
| hidden_sizes | [128,128] |
| learning_rate / gamma | 0.0003 / 0.99 |
| buffer_capacity / batch_size | 50000 / 64 |
| learning_starts | 1000 transiciones iniciales aleatorias |
| train_frequency | 1 minibatch cada 4 pasos de entorno |
| target_update_frequency | copia dura cada 500 pasos, desde learning_starts |
| epsilon_start / epsilon_end | 1.0 / 0.1 |
| epsilon_decay_steps | 8000 pasos después de learning_starts, lineal |
| total_steps | 10000 |
| validation_frequency / checkpoint_frequency | 5000 / 5000 pasos |
| log_frequency / gradient_norm_limit | 100 pasos / 10 |

Durante warmup se escogen acciones uniformes; después epsilon-greedy. Evaluación:
argmax determinista, sin consultar el RNG de exploración. Todas las decisiones
ocurren a 20 Hz como en PID; no se cambian estados desde los agentes.
La configuración no se ajustó tras observar estos resultados ni se presenta como
óptima. CPU es la plataforma verificada; CUDA/MPS son opciones explícitas cuya
reproducibilidad y rendimiento no se verificaron.

## Registros

Cada ejecución crea un directorio nuevo (se rechaza sobrescribir uno no vacío):

- `episodes.csv`: episodios completos, semilla/estado inicial, retorno, suma de
  cada componente de recompensa, éxito, causa, duración, gas y métricas físicas.
  Datos al contacto quedan ausentes cuando no hay contacto.
- `updates.csv`: medias de loss, |TD|, |Q| y norma de gradiente por intervalo,
  epsilon, pasos acumulados, número de actualizaciones, tamaño de replay y copias
  objetivo. Antes de aprender, los campos de loss quedan vacíos, no cero.
- `validation.csv`: todos los episodios de cada punto de validación, con contador
  de entrenamiento. Los pasos de validación no cuentan como interacciones de
  entrenamiento ni producen actualizaciones.
- `metadata.json`: versiones, configuración completa, referencia congelada,
  manifiestos/hashes, contadores y tiempos medidos por actividad y total de sesión.

El presupuesto se detiene **exactamente** en total_steps, aunque corte un episodio.
Su retorno parcial no se mezcla con episodios completos: queda en el checkpoint
y su cantidad de pasos en metadata. DQN terminó con 25 pasos de episodio parcial;
Double DQN con 13. Un paso parcialmente integrado por contacto cuenta como una
interacción, igual que en el resto del proyecto.

## Checkpoints y reanudación

`*.policy.pt` (`kind=inference`): pesos principales, configuración, dimensiones,
paso de entrenamiento y descripción del experimento. Para inferencia/evaluación,
no para continuar el optimizador. `*.resume.pt` (`kind=resume`): redes principal y
objetivo, Adam, contadores, replay completo con posición circular y RNG; RNG de
exploración, muestreo de escenarios, Python, NumPy y Torch (y aceleradores si
están disponibles); estado físico, viento, tiempo, gas inicial, finalización,
última acción, RNG del entorno/espacio de acciones; observación y acumuladores
del episodio, métricas históricas y consignación de la próxima selección de caso.

Se puede reanudar **a mitad de episodio**, sin descartarlo ni transformarlo en
terminal. La captura/restauración está aislada en `checkpoints.py` y no altera la
implementación del entorno ni se utiliza para controlar trayectorias. Se usa el
modelo de estado específico de v0; si cambia la dinámica hay que versionarlo.
La carga exige mismo experimento/configuración salvo `total_steps`, que debe
superar el contador guardado. Es un presupuesto total, no pasos adicionales.
Guardar usa archivo temporal y reemplazo; cargar utiliza `weights_only=True`.

Las pruebas comparan una ejecución continua contra 32 pasos + reanudación hasta
64 para **ambos** algoritmos, incluyendo episodios completos y validación:
parámetros, Adam, logs, RNG de replay y transiciones coinciden exactamente. La
fidelidad se afirma para misma versión de código/dependencias, CPU/dispositivo e
hilos. No se garantiza identidad entre plataformas. Los checkpoints periódicos
y finales contienen historia completa, lo que aumenta almacenamiento en campañas
largas; se deben espaciar los guardados y conservar solo los necesarios.

## Comandos exactos

Desde la raíz, con Python 3.12:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,render,rl,analysis]'
python -m pytest
python scripts/check_env.py
python -m pip check
```

Repetir la prueba corta desde cero usando **rutas nuevas**:

```bash
python scripts/train.py --config configs/dqn.json --output artifacts/dqn_repeat
python scripts/train.py --config configs/double_dqn.json --output artifacts/double_dqn_repeat
python scripts/plot_training.py artifacts/dqn_repeat artifacts/double_dqn_repeat --output artifacts/repeat_curves.png
```

Continuar (ejemplo para una ejecución posterior; **no se ejecutó** en esta entrega):

```bash
python scripts/train.py --config configs/dqn.json --resume artifacts/dqn_smoke/last.resume.pt --total-steps 20000 --output artifacts/dqn_continued
python scripts/train.py --config configs/double_dqn.json --resume artifacts/double_dqn_smoke/last.resume.pt --total-steps 20000 --output artifacts/double_dqn_continued
```

Evaluar un checkpoint con validación separada y sin explorar:

```bash
python scripts/evaluate.py --checkpoint artifacts/dqn_smoke/last.policy.pt --scenarios configs/rl_validation_scenarios.json --trace --output artifacts/dqn_validation.csv
python scripts/evaluate.py --checkpoint artifacts/double_dqn_smoke/last.policy.pt --scenarios configs/rl_validation_scenarios.json --trace --output artifacts/double_dqn_validation.csv
python scripts/plot_training.py artifacts/dqn_smoke artifacts/double_dqn_smoke
python scripts/plot_pid.py artifacts/dqn_validation.csv
```

Visualizar (comportamiento real del checkpoint, aunque falle):

```bash
python scripts/evaluate.py --checkpoint artifacts/dqn_smoke/last.policy.pt --episodes 3 --render --output artifacts/dqn_visual.csv
python scripts/play.py --controller double_dqn --checkpoint artifacts/double_dqn_smoke/last.policy.pt --scenarios configs/rl_validation_scenarios.json --episodes 3 --render --output artifacts/double_visual.csv
```

`play.py` acepta ambos nombres de algoritmo y verifica que coincidan con el
checkpoint. `evaluate.py` permite `--device cpu|cuda|mps`; por defecto CPU.
Ambos usan los parámetros físicos guardados. Los CSV de evaluación y gráficas
sobrescriben su ruta explícita; el entrenador siempre exige directorio nuevo.

Las versiones instaladas se fijan en `configs/requirements-stage3.txt`, incluyendo
PyTorch 2.15.0; para repetirlas instalar ese archivo y luego el paquete editable
con `python -m pip install --no-deps -e .`. No se garantiza esa combinación en
sistemas distintos de macOS arm64/Python 3.12.

## Resultado real y diagnóstico

Referencia de ejecución: `artifacts/dqn_smoke/` y `artifacts/double_dqn_smoke/`.
Una semilla, 10000 pasos por algoritmo, 2251 actualizaciones y 19 sincronizaciones.
Ambos completaron sin loss, TD, Q ni retornos no finitos. Los pesos cambiaron,
los checkpoints se cargaron y su evaluación reprodujo exactamente las filas de
validación al paso 10000. No hubo aterrizajes en entrenamiento ni validación.

| Algoritmo | Episodios completos de entrenamiento | Validación a 5000 | Validación a 10000 | Retorno medio final de validación |
|---|---:|---:|---:|---:|
| DQN | 221 | 0/32 | 0/32 | -100.36408 |
| Double DQN | 206 | 0/32 | 0/32 | -102.07935 |

DQN final: 32 contactos inválidos; acciones predominantes apagado (844) y todos
(167) en los 32 casos. El primer caso vertical llegó a -7.760 m/s con θ=ω=0:
frenó demasiado tarde. Double DQN final: 18 contactos fuera de plataforma y
14 contactos del cuerpo; en el primer caso ω=-2.867 rad/s y vy=-15.907 m/s.
Las trazas completas respaldan el diagnóstico; no se consideran aterrizajes esas
métricas de contacto. Loss media del último intervalo: 2.180 y 1.983; |Q| medio
0.731 y 1.121. Valores finitos no prueban que las estimaciones sean correctas.

El retorno mejora parcialmente mientras sigue habiendo cero éxitos: un fallo
más rápido y barato puede acercarse a -100. La penalización terminal es grande,
pero la señal positiva de aterrizaje no apareció en replay. Solo se recopilaron
10000 transiciones y ~200 episodios, con momentos discretos gruesos y exploración
alta durante casi toda la prueba. Estas observaciones son compatibles con falta
de experiencia y dificultad de exploración/asignación de crédito; no prueban una
causa única ni que más pasos basten. No se modificó automáticamente recompensa,
física, umbrales o demostraciones para mejorar el resultado.

No se pueden comparar estas tasas con 98/100 PID: PID utilizó otro manifiesto y
RL no ha abierto prueba final. Tampoco se concluye superioridad DQN/Double DQN.
La curva `artifacts/rl_smoke_curves.png` es descriptiva, con una sola semilla.
Resumen pequeño versionado: `docs/rl_smoke_summary.csv`. Traza y figuras opcionales
de evaluación están bajo `artifacts/*_validation*`.

## Verificación y alcance

**75 pruebas pasaron**, preservando las 63 anteriores. Se verificaron replay,
targets con números conocidos, máscara terminal/truncada, ausencia de gradientes,
copia dura, acciones válidas, igualdad de inicialización, checkpoints de ambos
tipos, restauración de RNG/episodio, separación de conjuntos y evaluación greedy
sin modificar pesos/replay/RNG. Checker Gymnasium y `pip check` pasaron; siguen
las dos advertencias conocidas del Box infinito.
Las curvas se generaron con Agg y se inspeccionaron visualmente. Reproducción
DQN con `--render` comprobada en un episodio con **SDL dummy**, sin comprobar
ventana/teclado reales. No se ejecutó la campaña de múltiples semillas ni prueba
final. Los tests usan presupuestos pequeños para verificar lógica, separados de
las ejecuciones de integración de 10000 pasos.

## Presupuesto propuesto, aún no ejecutado

CPU, un hilo Torch por proceso. Las dos pruebas cortas corrieron concurrentemente:
DQN 5.766 s (1734 pasos/s) y Double DQN 7.002 s (1428 pasos/s), incluyendo validación
y guardado; sus ventanas se solapan y **no** deben sumarse para medir el tiempo
real de esa ejecución paralela. Tiempos de aprendizaje puros: 1.469/1.563 s;
dinámica: 3.087/3.074 s. Son mediciones muy cortas, con warmup, cachés y episodios
fallidos breves; el rendimiento cambiará con episodios largos y más validaciones.

Propongo un siguiente piloto de **100000 pasos por algoritmo**, todavía con una
semilla y validación separada, para revisar exploración, pérdidas y estabilidad.
No se ejecutó. Si la integración se mantiene estable, presupuestar hasta
**1000000 pasos por semilla × 3 semillas × 2 algoritmos = 6000000 interacciones**.
Con el intervalo medido de 1428–1734 pasos/s, la suma de tiempos de proceso sería
aproximadamente 58–70 minutos; reservar **1–2 horas** por validaciones/checkpoints
más grandes y variación de carga, sin afirmar cuánto aprenderán. Validar cada
10000 pasos y guardar cada 50000; elegir y congelar un descenso epsilon más largo
antes de la campaña, equivalente en ambos algoritmos (por ejemplo 500000 pasos,
como propuesta no ajustada). El schedule de 8000 pasos solo corresponde al smoke.

No se propone gastar ese presupuesto ciegamente: si el piloto continúa sin éxito,
revisar las trayectorias y la formulación experimental con validación, sin abrir
prueba ni relajar criterios. Toda modificación de recompensa o condiciones tendría
que versionarse como experimento nuevo. El conjunto final ya usado para PID no es
un recurso para ajustar los agentes RL.
