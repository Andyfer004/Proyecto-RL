# Thruster Landing — versión entregable 1

[**Abrir el informe completo en PDF (23 páginas)**](output/pdf/informe_final.pdf)

[**Ver el video de demostración en YouTube**](https://youtu.be/X4G2c8bK0bM)

Simulador académico propio **2D** de estabilización y aterrizaje con propulsores discretos, gas limitado y viento horizontal. La inspiración Tesla Roadster es conceptual: física y parámetros son propios, sin equivalencia comercial, hardware ni extensión a 3D.

El [informe final técnico y matemático](docs/informe_final.md) explica dinámica, PID, recompensa, DQN/Double DQN, demostraciones, protocolo y límites. La [matriz de trazabilidad](docs/trazabilidad_alcance.md) contrasta el alcance original con evidencia real. La [demostración](docs/demostracion.md) y su [galería offline](docs/demostracion.html) reproducen estados guardados, sin entrenar ni reevaluar.

## Resultados congelados

| Método principal | Reservados | Complementarios |
|---|---:|---:|
| PID puro | 98/100 | 287/288 |
| DQN asistido por demostraciones PID | 96/100 | 275/288 |
| Double DQN asistido por demostraciones PID | 94/100 | 266/288 |
| DQN desde cero, checkpoint seleccionado | 0/100 | 0/288 |
| Double DQN desde cero, checkpoint seleccionado | 0/100 | 0/288 |

Las políticas funcionales combinan **imitación y TD**; durante inferencia actúa exclusivamente la red. **No se aisló el beneficio de las actualizaciones TD frente a la imitación por sí sola**. La asistencia es una ampliación metodológica adicional al enfoque inicial. PID tuvo mayor éxito entre los métodos principales. No se demostró adaptación automática, superioridad general ni menor coste computacional de las redes.

Se conservaron 100 casos reservados y 288 complementarios independientes con altura, desplazamiento, inclinación, velocidades y viento. El PID ya había sido evaluado en los 100 históricos. La segunda semilla DQN asistida dio 100/100 y 277/288; no se eligió retrospectivamente como candidata principal ni equivale a una campaña equilibrada de semillas. El [informe experimental](docs/evaluacion_alcance.md) incluye todos los grupos, causas de fallo y consumo sobre éxitos/fracasos y pares comunes.

**La prueba final ya fue ejecutada y los modelos están congelados.** Este cierre no añade entrenamiento ni reajustes. Cualquier intervención posterior exige otro experimento y evaluación independiente.

## Instalación

Desde la raíz del proyecto, con Python3.12 (entorno verificado3.12.12, macOS arm 64, CPU):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,render,rl,analysis]'
python -m pip check
```

Las versiones de la evidencia están fijadas en [requirements-stage3.txt](configs/requirements-stage3.txt): PyTorch2.15.0, NumPy2.5.4, Gymnasium1.4.0, pandas3.1.0, Matplotlib3.11.2, pygame-ce2.5.8 y Pillow13.0.0. Para reproducir ese entorno compatible:

```sh
python -m pip install -r configs/requirements-stage3.txt
python -m pip install --no-deps -e .
```

El paquete declara Python3.11–3.14; solo se verificó3.12. El modo básico sin redes/gráficos necesita únicamente dependencias base; la entrega completa usa los extras anteriores. No instalar pygame y pygame-ce simultáneamente.

## Demostración sin entrenamiento

1. Abrir `docs/demostracion.html` directamente en un navegador: usa GIF locales y figuras ya guardadas.
2. Comprobar que el clon incluye la evidencia pública, sus hashes y las cifras, sin simular ni optimizar:

```sh
.venv/bin/python scripts/verify_repository.py
```

3. Opcionalmente, reconstruir tablas/figuras o GIF desde CSV, sin nuevos pasos físicos:

```sh
.venv/bin/python scripts/report_scope.py
.venv/bin/python scripts/animate_scope.py
```

Las animaciones RGB se exportaron bajo SDL dummy. El script congelado asigna ese driver por defecto incluso con `--human`; por ello ese comando solo no certifica ventana real. La incidencia y una opción de driver explícito están en [demostracion.md](docs/demostracion.md). Teclado/ventana reales siguen sin verificarse.

## Entrega y reproducción

La guía [entrega/README.md](entrega/README.md) describe contenido y pendientes. Para construir el ZIP local con fuentes, configuraciones, informe, CSV, GIF y modelos seleccionados:

```sh
.venv/bin/python scripts/build_delivery.py
```

Salida: `artifacts/delivery_v1/proyecto_rl_entrega_v1.zip`, con manifiesto SHA-256 interno y checksum del ZIP. El repositorio incluye una selección explícita de modelos de inferencia, CSV, trazas, GIF y metadatos congelados, inventariada en [publicacion_manifest.json](entrega/publicacion_manifest.json). Por tanto, un clon permite abrir la demostración y verificar la evaluación sin entrenar ni descargar pesos aparte. `artifacts/` permanece ignorado por defecto para resultados nuevos; solo se versiona esa selección. Los replays, checkpoints de reanudación, datos docentes binarios regenerables y resto del historial bruto quedan locales. El ZIP también incluye el PDF actual y su LaTeX.

Los comandos históricos de entrenamiento permanecen en [dqn.md](docs/dqn.md) y [recuperacion_agente.md](docs/recuperacion_agente.md), pero no forman parte de esta demostración ni autorizan reajustes sobre prueba final. Los checkpoints asistidos son de inferencia, sin estado completo para reanudación fiel.

## Verificación y límites

La última suite completa registrada aprobó 118 pruebas y el checker Gymnasium fue válido, con dos advertencias conocidas por Box de límites infinitos. El cierre actual verifica hashes,2328 episodios finales, contadores CE/TD, cifras emparejadas, ejemplos matemáticos y rutas, sin repetir entrenamiento/evaluación. `pip check` no detectó incompatibilidades. No se confunden estas verificaciones con garantías de estabilidad o validación real.

La física usa masa/inercia constantes, presupuesto de impulso en N·s, observación ideal, viento constante y primer contacto terminal sin rebote ni asentamiento. Se evaluaron ocho acciones a 20 Hz,100kg, empujes 1800/600/600N y 18000N·s. Las limitaciones y el número desigual de semillas están descritos en el informe, sin tratarlos como problemas resueltos.

## Organización

- `src/thruster_landing/envs/`: física, geometría, recompensa, entorno y renderer propios.
- `src/thruster_landing/agents/`: PID, replay y DQN/Double DQN desde cero.
- `src/thruster_landing/recovery_agent.py`: variante asistida explícitamente identificada.
- `configs/`: parámetros, manifiestos, protocolos y referencias congeladas.
- `scripts/`: ejecución, evaluación, diagnóstico y cierre de entrega.
- `tests/`: pruebas de física, contrato, controladores, aprendizaje y persistencia.
- `docs/`: informe final, demostración, trazabilidad e informes históricos.
- `docs/resultados/`: tablas/figuras seleccionadas; los archivos `alcance_*` pertenecen al protocolo final.
- `artifacts/scope_alignment_v1/`: prueba final, modelos y trazas congelados.
- `artifacts/recovery_v1/`: desarrollo asistido y evidencias previas.
- `entrega/`: instrucciones de distribución y pendientes explícitos.

El [historial de resultados](docs/resultados/README.md) conserva también los intentos fallidos: smoke, pilotos, LR, beta, currículo, exploración ponderada y replay 150k. Esos registros no se reescriben como éxitos asistidos.

## Qué se publicó

[Entrega versionada](entrega/REPOSITORIO.md) detalla los archivos y los comandos para una copia limpia. Los resultados de nuevas ejecuciones y borradores siguen excluidos mediante `.gitignore`. La publicación conserva los hashes del protocolo final; no añade entrenamiento ni evaluaciones finales.
