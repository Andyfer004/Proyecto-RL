> Estado entregable actual: [informe_final.md](informe_final.md), [evaluacion_alcance.md](evaluacion_alcance.md) y [demostracion.md](demostracion.md). Lo que sigue conserva el registro histórico de cada etapa; sus afirmaciones de pendientes o versiones se refieren a esas fechas.

# Verificación ejecutada — 2 de noviembre de 2026

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

Entorno local macOS arm64, Python 3.12.12, CPU. Instalación editable en `.venv`.
Versiones utilizadas: NumPy 2.5.4, Gymnasium 1.4.0, pygame-ce 2.5.8 (SDL 2.32.10),
Pytest 9.2.0. Las dependencias transitivas quedan fijadas en
`configs/requirements-stage1.txt`. PyTorch, Pandas y Matplotlib no se instalaron:
pertenecen a etapas posteriores y su compatibilidad todavía no se ha verificado.

## Comprobaciones y evidencia

- `python -m pytest`: **37 pruebas pasaron**, sin fallos. Cubren caída libre contra
  solución analítica, rotación de fuerzas, signos de momentos, aceleración angular,
  frenado disponible, consumo conjunto y parcial, ausencia de empuje sin gas,
  fuerza de viento, geometría, convergencia de subpasos, conservación de presupuesto,
  estados iniciales, umbrales, tipos de fallo y tiempo analítico hasta contacto.
- Reproducibilidad verificada tanto en trayectorias con viento como en dos
  ejecuciones completas del CLI: los CSV resultaron idénticos para igual semilla.
- `python scripts/check_env.py`: contrato oficial de Gymnasium válido.
  El checker emite dos advertencias sobre límites infinitos del Box. Son
  deliberados para no recortar velocidades físicas ni declarar cotas falsas;
  no son fallos. Seno/coseno y gas sí tienen límites finitos.
- `python -m pip check`: ninguna dependencia incompatible instalada.
- Modos `human` y `rgb_array` comprobados con `SDL_VIDEODRIVER=dummy`, incluido
  evento de cierre de ventana y checker oficial con renderizado. Imagen RGB de
  800×900×3 uint8 generada e inspeccionada visualmente en
  `artifacts/render-preview.png`. Falta una sesión humana con teclado y ventana
  reales; SDL dummy verifica el código gráfico, no la interacción local real.
- Diez episodios aleatorios sin viento y diez con viento U(-80,80) N,
  semillas 42–51, completados y exportados. Los veinte terminaron por
  `out_of_bounds`, ninguno aterrizó. Son pruebas de ejecución, no una estimación
  comparativa ni evidencia de control exitoso. CSV y metadatos están en
  `artifacts/random.csv`, `artifacts/wind.csv` y archivos JSON asociados.

Las pruebas de éxito usan estados sintéticos próximos al contacto, adecuados
para verificar el criterio geométrico; no demuestran aterrizaje desde la altura
inicial de entrenamiento. Esa demostración corresponde al PID de Etapa 2.
No se ejecutaron entrenamientos ni se generaron checkpoints de redes.

## Repetición

```bash
source .venv/bin/activate
python -m pytest
python scripts/check_env.py
python -m pip check
python scripts/play.py --controller random --episodes 10 --seed 42 --output artifacts/random.csv
python scripts/play.py --controller random --config configs/env_wind.json --episodes 10 --seed 42 --output artifacts/wind.csv
```

Los artefactos generados se excluyen de Git; la evidencia de versiones y comandos
queda documentada aquí. Cambiar constantes o dependencias requiere volver a ejecutar
las comprobaciones; no se garantiza identidad bit a bit entre plataformas.

# Ampliación — Etapa 2

Se conservó la suite anterior y se añadieron pruebas de las ocho fuerzas/momentos,
signos de la cascada, gravedad, perfil de descenso, saturación, anti-windup,
reinicio, asignación con orientación y poco gas, separación de manifiestos,
reproducibilidad del CLI PID, y ausencia de métricas de contacto ante truncamiento.
`python -m pytest`: **63 pruebas pasaron**, incluidas las 37 originales. Los
detalles del controlador y la evaluación están en `pid.md` y `resultados.md`.

Gymnasium y `pip check` siguen pasando; permanecen las dos advertencias documentadas
sobre cotas infinitas. Se instalaron y utilizaron Pandas 3.1.0 y Matplotlib 3.11.2
para graficar los CSV; versiones completas en `configs/requirements-stage2.txt`.
No se instaló PyTorch. Las gráficas de trayectorias y estados fueron generadas con
Agg e inspeccionadas visualmente. No se afirma comprobación de teclado real.

# Ampliación — Etapa 3

**75 pruebas pasaron**, conservando las 63 previas. Incluyen valores conocidos de
targets DQN/Double DQN, terminales y truncamientos, gradientes/copia objetivo,
replay circular, exploración válida, checkpoints tipados y reanudación exacta de
ambos algoritmos con episodios y validación. Checker Gymnasium y `pip check`
pasaron; se conservan las dos advertencias conocidas sobre límites infinitos.

PyTorch 2.15.0 instalado en CPU; versiones completas en `requirements-stage3.txt`.
Cada algoritmo completó 10000 pasos y 2251 actualizaciones sin valores no finitos.
Las políticas recargadas reprodujeron la validación final, sin exploración ni
cambio de pesos. La referencia PID 98/100 se verificó desde sus CSV y hashes,
sin repetir la prueba final. No se modificó la dinámica ni el éxito.

Curvas `artifacts/rl_smoke_curves.png` generadas con Agg e inspeccionadas visualmente.
Reproducción DQN con `play.py --render` ejecutada en un caso de validación con
SDL dummy; no se verificó teclado físico. No se ejecutaron campañas largas.
Resultados, diagnóstico, comandos y estimación de presupuesto en `dqn.md`.

# Piloto controlado 100000

**77 pruebas pasaron** y checker Gymnasium válido antes del piloto; se añadieron
selección de mejor checkpoint y exportación inicial/mejor/final. Se ejecutaron
exactamente 100000 pasos y 24751 actualizaciones por algoritmo, desde cero y con
configuraciones equivalentes salvo algoritmo. Baseline y diez puntos de validación
sobre los mismos 32 casos: 352 filas por agente, sin uso del conjunto de prueba.

Se verificaron hashes PID/física, checkpoints best=10000 y last=100000, suma de
componentes, finitud y normalización del replay, targets terminales exactos y
métricas de contacto ausentes en las salidas finales. La reconstrucción por
transición de recompensas desde trazas coincide con los retornos guardados y
permite comparar Q con retornos descontados. Hubo deriva optimista de Q, no NaN/Inf.
Figuras generadas con Agg e inspeccionadas; no se comprobó teclado real.
Resultados y diagnóstico en `piloto_100k.md`; no se amplió presupuesto.

# Auditoría y ablación LR=0.0001

**81 pruebas pasaron**, incluyendo cuatro nuevas comprobaciones de integración:
acción almacenada frente a gather/gradientes, formas vectoriales reales sin
broadcasting, next_obs antes de reset tanto terminal como truncado, frecuencia
exacta de copia objetivo y observación idéntica en entrenamiento/evaluación.
Checker Gymnasium válido, con advertencias previamente documentadas.

No se encontró un defecto que requiriera corregir baseline. Se ejecutaron desde
cero 100000 pasos / 24751 actualizaciones por agente, cambiando solo LR. Se añadió
telemetría de norma posterior al clipping existente. Una sonda offline fija de
64 terminales y 64 no terminales se aplicó a los cuatro modelos y 11 checkpoints:
5632 registros, sin optimizar ni cambiar de transiciones. Se verificaron IDs,
target terminal=r, bootstrap terminal=0 y signos TD/error terminal opuestos.

Los pesos iniciales coinciden exactamente con los previos; configuraciones solo
difieren en LR; 352 filas de validación por agente. Normas posteriores al clipping
cumplen la cota con tolerancia numérica. Se preservaron los hashes físicos/PID y
todos los checkpoints/resultados previos. Gráficas comparativas generadas offline
e inspeccionadas. La brecha frente al retorno greedy no se presenta como error
exacto respecto a Q óptimo. Resultados y límites en `ablacion_lr1e4.md`.

# Diagnóstico offline acotado

Dos pruebas nuevas pasaron (`tests/test_offline_diagnostics.py`): identidad de la
suma proporcional de gradientes/coseno indefinido y copias independientes de
pesos/Adam, sin reponderar artificialmente grupos en ramas unitarias. No se
modificó código de aprendizaje RL ni se repitió una campaña.

Protocolo `configs/offline_interference_protocol.json`: 128 minibatches uniformes
por replay, 500 pasos supervisados por copia y 32 pruebas unitarias pareadas por
modelo con Adam guardado/nuevo. Solo beta=1; 0 pasos de entorno. Splits por (s,a)
sin duplicados exactos y terminales reservadas excluidas de los updates de prueba.
Todas las 6602 terminales se enlazaron a logs originales: última observación
exactamente igual al estado final codificado, y evento coherente. Se verificaron
pares previos del mismo episodio y hashes de todos los experimentos antes/después.
Los resultados no se presentan como aprendizaje de aterrizaje. Informe y límites
en `diagnostico_offline_interferencia.md`.

# Ablación SmoothL1 beta=10

**87 pruebas pasaron**, checker Gymnasium y pip check válidos. Las nuevas pruebas
verifican clase nn.SmoothL1Loss, pérdidas y pendientes beta=10, diferencia frente
a HuberLoss(delta=10), ausencia de multiplicación compensatoria en learn,
compatibilidad beta=1 y gradientes ponderados a beta explícito.

Se ejecutaron exactamente 100000 pasos y 24751 actualizaciones por agente desde
cero; configuraciones expandidas difieren solo en beta respecto a LR=.0001.
Pesos iniciales idénticos y mismos 32 escenarios en todos los puntos. Checkpoints
best=10000 por empate, last=100000. Los datos diagnósticos A/B/D conservan muestras
previas; C usa replay nuevo etiquetado aparte y empareja datos entre modelos.
Se computan gradientes offline a ambas escalas sin pasos de optimizador.

La reducción de loss no se usa como prueba de control: ambos algoritmos siguen
con cero éxitos, y los errores terminales/brechas se reportan sobre datos fijos.
Campos de contacto ausentes para salidas, consumo/duración separados por resultado.
Referencias físicas/PID y artefactos previos preservados por hashes. Figuras de
control, valores, fuentes y trayectorias generadas e inspeccionadas visualmente;
no se comprobó teclado real ni se ejecutó prueba final o una campaña adicional.

# Verificación de la publicación actual — 2 de noviembre de 2026

Los apartados previos son registros históricos de cada etapa, no el inventario
del estado actual. La suite completa actual obtuvo **91 passed**, con dos avisos
conocidos sobre límites infinitos del espacio Box. El checker oficial de
Gymnasium pasó y `pip check` no encontró dependencias rotas.

Se contrastaron los CSV de PID (31/32 validación, 98/100 prueba), sus resúmenes
por condición, y los CSV de entrenamiento/validación y metadatos de diez
ejecuciones RL: integración, piloto, LR, beta y currículo, por ambos algoritmos.
Ninguno registró aterrizajes. Se prepararon dos tablas compactas y tres figuras
existentes, sin alterar las fuentes locales. Los hashes consultados están en
`resultados/procedencia.json`; las fuentes `artifacts/` permanecen excluidas.
No se repitió la prueba PID reservada ni se ejecutaron nuevos experimentos RL.
Las pruebas automatizadas sí incluyen actualizaciones sintéticas y ejecuciones
mínimas para verificar aprendizaje y reanudación.

Se repitieron las 91 pruebas y el checker sobre una exportación exacta del índice
sin `artifacts/` ni archivos ignorados, usando las dependencias locales y el
código exportado mediante `PYTHONPATH`. Se construyó el wheel con
`pip wheel --no-deps --no-build-isolation --no-index` sin descargar dependencias.
Esto verifica empaquetado y ejecución desde el árbol publicado; no representa
una instalación nueva de dependencias en otra plataforma.

La auditoría del índice revisó tamaños, patrones de secretos, rutas personales,
enlaces Markdown y archivos ignorados ya indexados: sin hallazgos. Se conservaron
los bytes CSV con `.gitattributes` para no invalidar el hash PID al clonar. El árbol
publicado contiene 130 archivos, unos 1,8 MB; el mayor es una figura de 329 247 bytes.
Los 586 MB aproximados de artefactos locales se conservan fuera de Git.
