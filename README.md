# Thruster Landing

Proyecto universitario de control de aterrizaje de un cuerpo rígido **2D**, con
propulsores discretos y gas limitado. Incluye física propia, entorno Gymnasium,
renderizado, PID en cascada y agentes DQN/Double DQN. No es una adaptación de
LunarLander ni un modelo de vuelo 3D.

El PID obtuvo **31/32 en validación y 98/100 en la prueba reservada documentada**.
Los pilotos RL **no demostraron aterrizajes**, ni en entrenamiento ni en
validación. No se ejecutó la prueba final RL ni una campaña de varias semillas.
Las tasas PID y RL pertenecen a manifiestos distintos y no permiten afirmar
superioridad mediante una comparación directa.

## Instalación

Desde la raíz, con Python 3.12 (entorno verificado: 3.12.12, CPU):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,render,rl,analysis]'
python -m pytest
python scripts/check_env.py
python -m pip check
```

Para simulación y PID sin gráficos basta `python -m pip install -e '.[dev]'`.
`render` instala pygame-ce; `rl`, PyTorch; `analysis`, Pandas y Matplotlib.
No instales pygame y pygame-ce simultáneamente. El paquete admite Python
3.11–3.14; solo se verificó 3.12. Las versiones exactas del entorno local están en
[configs/requirements-stage3.txt](configs/requirements-stage3.txt):

```bash
python -m pip install -r configs/requirements-stage3.txt
python -m pip install --no-deps -e .
```

## Simulador y PID

Estos comandos funcionan sin checkpoints ni resultados previos:

```bash
python scripts/play.py --controller random --episodes 10 --seed 42 --output artifacts/random_new.csv
python scripts/play.py --controller random --config configs/env_wind.json --episodes 10 --seed 42 --output artifacts/wind_new.csv
python scripts/play.py --controller pid --config configs/env.json --pid-config configs/pid.json --scenarios configs/pid_validation_scenarios.json --trace --output artifacts/pid_validation_new.csv
python scripts/plot_pid.py artifacts/pid_validation_new.csv
python scripts/play.py --controller pid --pid-config configs/pid.json --scenarios configs/pid_validation_scenarios.json --episodes 3 --render --output artifacts/pid_visual_new.csv
python scripts/play.py --controller manual --episodes 3 --output artifacts/manual_new.csv
```

Sin `--render` se ejecuta sin gráficos; el modo manual requiere ventana y teclado.
También está disponible `thruster-play`. Espacio activa el propulsor central;
izquierda/derecha inclinan; combinadas con espacio activan central y lateral;
ambas flechas activan los laterales y arriba todos. Esc o cerrar interrumpe sin
contar el episodio incompleto. No se verificó interacción con teclado físico.

Los manifiestos fijan semillas y estados; sin `--episodes` se ejecutan todos los
casos. `--trace` añade estados y consignas por paso; se generan resumen y metadatos.
Usa rutas nuevas: la ejecución visual puede sobrescribir una salida existente.
La prueba PID documentada usa `configs/pid_test_scenarios.json`, separado del
ajuste; no lo uses para retocar parámetros y afirmar otra prueba independiente.
El [informe PID](docs/pid.md) detalla reproducción, ecuaciones y limitaciones.

## Entrenamiento y evaluación RL

Ejemplos para una ejecución futura desde cero; **no se han ejecutado nuevos
experimentos al preparar esta publicación**. Los directorios deben ser nuevos.
Los comandos de evaluación usan el checkpoint creado por el entrenamiento del
mismo bloque; no dependen de pesos incluidos en GitHub.

```bash
python scripts/train.py --config configs/dqn.json --output artifacts/dqn_new
python scripts/train.py --config configs/double_dqn.json --output artifacts/double_dqn_new
python scripts/evaluate.py --checkpoint artifacts/dqn_new/last.policy.pt --scenarios configs/rl_validation_scenarios.json --trace --output artifacts/dqn_new_validation.csv
python scripts/plot_training.py artifacts/dqn_new artifacts/double_dqn_new --output artifacts/new_curves.png
python scripts/play.py --controller dqn --checkpoint artifacts/dqn_new/last.policy.pt --scenarios configs/rl_validation_scenarios.json --episodes 3 --render --output artifacts/dqn_new_visual.csv
```

Los configs básicos fijan 10 000 pasos por algoritmo, semilla 42, CPU y MLP
128/128. Hay escenarios separados para entrenamiento y validación; los scripts
RL bloquean la prueba final. Se comprueba la referencia física/PID mediante hashes.
`.policy.pt` sirve para inferencia; `.resume.pt` contiene Adam, replay, RNG y el
episodio vivo. Para reanudar con presupuesto **total** de 20 000 pasos:

```bash
python scripts/train.py --config configs/dqn.json --resume artifacts/dqn_new/last.resume.pt --total-steps 20000 --output artifacts/dqn_continued
```

Las configuraciones de pilotos, ablaciones y currículo se conservan en `configs/`.
Los diagnósticos offline que leen replays históricos requieren artefactos locales
excluidos; no son necesarios para ejecutar el simulador, PID o entrenar desde cero.
Consulta [DQN/Double DQN](docs/dqn.md) y [los protocolos](configs/README.md).

## Estado actual y limitaciones

| Evaluación documentada | Pasos por agente | Resultado |
|---|---:|---|
| PID, validación de ajuste | No aplica | 31/32 aterrizajes |
| PID, prueba reservada | No aplica | **98/100**; dos fallos por velocidad angular |
| DQN y Double DQN, integración | 10 000 | Ambos 0/32 en validación final |
| Piloto con exploración más lenta | 100 000 | Ambos 0/32; salidas por techo |
| Ablación LR=0.0001 | 100 000 | Ambos 0/32; persiste deriva de Q |
| Ablación SmoothL1 beta=10 | 100 000 | Ambos 0/32; mejora numérica sin aterrizajes |
| Currículo, parada tras nivel 1 | 50 000 | Ambos 0/128 finales: 32 por nivel más 32 originales |

Todos los experimentos RL usan una sola semilla (42). Los puntos repetidos de
validación no son muestras independientes. El currículo no entrenó niveles 2/3;
su validación sí los inspeccionó. La mejora de loss/Q no demuestra control exitoso.
La selección compacta y el resumen de intentos fallidos están en
[docs/resultados/](docs/resultados/README.md); los informes técnicos conservan los
diagnósticos. Las cifras se contrastaron con CSV y metadatos locales existentes.

El simulador mantiene masa/inercia constantes, gas como impulso disponible,
viento constante observable, integración en subpasos y contacto terminal sin
asentamiento. Vehículo de 100 kg, empujes 1800/600/600 N y gas de 18 000 N·s;
decisiones a 20 Hz y física a 200 Hz. Los resultados no garantizan generalización
fuera de los rangos evaluados. La campaña comparativa completa queda pendiente.

## Estructura del proyecto

- `src/thruster_landing/envs/`: física, recompensa, Gymnasium y renderizado.
- `src/thruster_landing/agents/`: PID, replay, DQN y Double DQN.
- `src/thruster_landing/`: entrenamiento, currículo, evaluación y diagnósticos.
- `configs/`: parámetros, escenarios, protocolos, versiones y referencias históricas.
- `scripts/`: ejecución, entrenamiento, evaluación, gráficas y comprobaciones.
- `tests/`: física, API, controladores, renderizado, aprendizaje y reanudación.
- `docs/`: [diseño](docs/diseno.md), [metodología](docs/metodologia.md),
  [resultados](docs/resultados.md) y [verificaciones](docs/verificacion.md).
- `docs/resultados/`: tablas pequeñas y figuras seleccionadas publicadas.
- `artifacts/`: registros completos, checkpoints y trazas **solo locales**, ignorados.

Los manifiestos `configs/reference*.json` y `curriculum_execution_manifest.json`
son registros de auditoría local, no listas de archivos incluidos: conservan hashes
históricos y pueden mencionar artefactos ausentes del clon. No se publican pesos,
replays, logs completos, videos, conversaciones, credenciales ni datos privados.
Se conserva código y documentación útil independientemente de asistencia de IA.
