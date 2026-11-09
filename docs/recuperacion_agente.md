# Recuperación del agente: dos intervenciones y aterrizajes verificables

Se obtuvo una **política neuronal asistida** que aterriza sin explorar ni consultar al PID durante inferencia. No es DQN estándar ni aprendizaje desde cero: utiliza trayectorias PID de entrenamiento, preentrenamiento supervisado y TD con demostraciones permanentes. La recompensa, dinámica, gas inicial, catálogo de ocho acciones, frecuencia de 20 Hz, PID y criterios de contacto permanecieron intactos. No se ejecutó la prueba final durante esta intervención. La evaluación común posterior se documenta en [evaluacion_alcance.md](evaluacion_alcance.md).

Los artefactos completos están en [`artifacts/recovery_v1`](../artifacts/recovery_v1), con dos versiones separadas. La candidata recomendada es [`candidate_coverage_v2.policy.pt`](../artifacts/recovery_v1/candidate_coverage_v2.policy.pt). La primera candidata se conserva como `candidate.policy.pt`.

## Diagnóstico que motivó la ruta

Se verificaron las tablas previas, en lugar de asumir sus conclusiones: DQN con exploración ponderada descubrió tres éxitos en los pasos 2324, 4257 y 23468; Double DQN ninguno. Con replay de 150000 se conservaron tres terminales exitosos y 56 transiciones de sus episodios, muestreadas 132 y 2400 veces respectivamente, pero ambos agentes continuaron con cero éxitos greedy. Retener experiencias no bastó para preservar sus acciones en la política. Esto no demuestra que faltara exclusivamente exposición.

La auditoría anterior de recompensa ordenaba favorablemente al PID en ejemplos emparejados; no prueba corrección global del objetivo. El descuento sigue haciendo menos negativo un fracaso retrasado. No se cambió la recompensa en esta intervención.

| Hipótesis priorizada | Evidencia a favor / en contra | Comprobación discriminante | Resultado y límite |
|---|---|---|---|
| Error compartido de física, contacto o criterio | Muchas pruebas podían compartir fórmulas; el PID sí controlaba | Ocho acciones frente a ODE independiente RK4; caída libre analítica; clasificador de contacto geométrico independiente | Error máximo de estado por decisión 0,000825 SI con subpaso de 0,005 s, disminuye al refinar a 0,001 s; contacto libre coincide a <1e-9. No se encontró defecto que explique los fallos. No es una prueba global de precisión. |
| Agente incapaz de aprender en general | Cero éxitos greedy propios; targets y replay ya auditados | DQN del proyecto en CartPole y referencia independiente basada en tutorial oficial | El DQN mejora de 13,6 a 190 de retorno medio, termina en 120,2; referencia termina en 500. Hay aprendizaje, también inestabilidad. Las configuraciones difieren: no atribuir la diferencia a un único componente ni certificar todo el agente. |
| Señal de acción exitosa demasiado frágil bajo TD y escalas poco útiles | Solo 56 transiciones de episodios exitosos; su exposición no produjo una política repetible; cerca del contacto vx/10 y omega/3 son pequeños | Escalas de aterrizaje, demostraciones permanentes y margen supervisado; evaluar red sin profesor | Primera intervención supera 80% sostenido en L1 y 64/64 en comprobación. Cambiaron varios componentes deliberadamente: no es una ablación causal de cada uno. La imitación sola ya obtiene buen desarrollo. |
| Falta de soporte fuera de estados sencillos | Primera candidata: 0/32 originales; demostraciones solo desde 1,6–2,4 m frente a evaluación desde 10–12 m | Segunda/final intervención: misma metodología, ampliar soporte de entrenamiento y congelar nuevo desarrollo/comprobación | Recupera transferencia a los 32 casos originales. No demuestra generalización a viento, grandes inclinaciones ni toda la distribución del simulador. |

### Revisión del sistema completo

`scripts/audit_recovery.py` compara una ODE RK4 con fuerzas y momentos codificados de forma independiente, sin reutilizar el catálogo ni la rotación del integrador. Verifica la caída libre desde y=2 m, vy=-0,2 m/s: contacto a 0,4315963155 s, vy=-4,4339598555 m/s. Los ocho pares (fuerza N, momento Nm) son (0,0), (1800,0), (600,-360), (600,360), (1200,0), (2400,-360), (2400,360), (3000,0). Masa 100 kg, inercia 37,6667 kg m², gravedad 9,81 m/s². Un pulso lateral de 0,05 s cambia omega en 0,477876 rad/s: supera por sí solo el umbral de 0,25 y exige compensación. La discretización dificulta el control fino, pero no lo hace imposible.

Se reprodujeron PID y RL previo desde el mismo caso L1 de validación: PID aterriza en 41 decisiones con retorno 99,0339; RL termina fuera del área en 156 decisiones con -105,9309, sin contacto. Se conservan estados antes/después y acciones en `audit/reproduced_trajectories.csv`. Las métricas de contacto se dejan vacías en episodios sin contacto. El validador geométrico del informe vuelve a calcular pies, esquinas y límites cinemáticos, independientemente de la rutina de éxito del entorno.

Observación original: `[x/12,(y-1)/25,vx/10,vy/10,sin(theta),cos(theta),omega/3,gas/18000,wind/100]`. El estado físico observado incluye gas y viento; no incluye el reloj. Se mantiene la interpretación de límite temporal externo: se hace bootstrap en truncamiento, nunca en terminación física. Si se pretendiera un objetivo de horizonte finito intrínseco, faltaría el tiempo restante. Los fallos greedy inspeccionados eran salidas físicas, no timeouts: esa omisión no los explica por sí sola.

Se inspeccionaron selección/almacenamiento de acción, `gather` de Q(s,a), formas vectoriales de recompensa/target, siguiente observación anterior al reset, sincronización inicial de objetivo, ausencia de gradientes en objetivo y máscaras. Las pruebas nuevas comprueban targets de uno y diez pasos con números conocidos y redes online/objetivo que seleccionan acciones distintas. Entrenamiento y evaluación pasan por la misma transformación de observación. No se encontró un defecto concreto que justificara invalidar los baselines.

## Fuentes y decisiones

- [DQN, artículo original](https://www.nature.com/articles/nature14236): conservar replay, red objetivo y objetivo de Bellman; no interpretar su éxito en Atari como garantía en este vehículo.
- [Tutorial oficial de PyTorch](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html): contraste CartPole con implementación independiente de replay, targets, AdamW y copia suave. Adaptación sin gráficos, RNG explícito y presupuesto contado; no es una reproducción literal. Proyecto: 50000 pasos, Adam LR 3e-4, copia dura 500; referencia: 50000, AdamW 3e-4, tau 0,005. Dieciséis casos fijos por evaluación. Curvas y configuraciones en `cartpole/`.
- [DQfD, artículo original](https://arxiv.org/abs/1704.03732): fundamenta combinar demostraciones, margen supervisado y targets multietapa. Aquí se usa calentamiento por entropía cruzada y mezcla fija uniforme por estrato, **sin PER**; no se denomina reproducción exacta de DQfD.
- [Double DQN, artículo original](https://arxiv.org/abs/1509.06461): principal selecciona siguiente acción, objetivo evalúa; se transfirió el método solo después de obtener aterrizajes DQN.
- [Gymnasium: límites temporales](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/): distinguir terminación de truncamiento y documentar la suficiencia del estado respecto al horizonte.

## Método congelado y conjuntos separados

Versión1: 2048 estados nuevos de entrenamiento, 64 de desarrollo y 64 de comprobación; semillas 210000+, 220000+, 230000+. L1 sin viento: x ±0,1 m, y 1,6–2,4 m, vx ±0,03 m/s, vy [-0,3;-0,05] m/s, theta ±0,01 rad, omega ±0,005 rad/s. Ninguno tiene contacto inicial. Gas inicial habitual: 18000 Ns.

Versión2: nuevos manifiestos disjuntos, 2048 entrenamiento, 128 desarrollo y 128 comprobación; semillas 310000+, 320000+, 330000+. Mezcla fija 50% L1 y 50% altura ampliada: x ±0,3 m, y 8–13 m, vx ±0,1 m/s, vy [-1;0] m/s, theta ±0,03 rad, omega ±0,005 rad/s, sin viento. La etiqueta `original` de este nuevo manifiesto significa **soporte alto diseñado para cubrir condiciones originales**, no que copie sus 32 casos. Estos 32 siguen siendo validación previa separada. No se incorporaron estados de desarrollo, comprobación o prueba final a demostraciones ni replay.

En cada versión se ejecutó PID congelado en los primeros 1024 casos de entrenamiento. V1: 1022 éxitos, 47043 decisiones físicas, 46954 transiciones retenidas. V2: 1005 éxitos, 87099 decisiones, 84882 transiciones retenidas. Se descartan trayectorias fallidas para esta fuente supervisada. Un octavo de episodios (índice%8=0) se reserva como holdout interno y sus etiquetas/transiciones docentes nunca se muestrean para aprender. Los estados iniciales siguen perteneciendo al pool de entrenamiento online: este holdout mide imitación de trayectorias, no generalización a estados iniciales nuevos. En V2, por el entrelazado bajo/alto, ese holdout interno contiene solo L1; **no se usa su exactitud para afirmar generalización en altura**. El desarrollo y la comprobación sí tienen ambos grupos equilibrados.

La representación `landing_scales_v2` es `[x/1.5,(y-1)/2,vx/.7,vy/1,sin(theta)/.12,cos(theta),omega/.25,gas/18000,wind/100]`: transformación invertible de las mismas nueve variables, sin información futura, estado privilegiado ni PID en inferencia.

Arquitectura 128/128 ReLU, Adam LR 0,0003, gamma 0,99, SmoothL1 beta=10 sin compensar escala, clipping de norma 10, replay online 50000. Calentamiento: 10000 actualizaciones de entropía cruzada (batch256), luego 10000 TD asistidas (batch128). Después una actualización por cuatro decisiones, desde 128 transiciones online. Mezcla 64 demostraciones/64 replay online, muestreo uniforme dentro de cada fuente; no es muestreo uniforme del conjunto unido. Demostraciones permanentes; sin prioridades. Epsilon 0,05 fijo, exploración uniforme entre las ocho acciones; es una metodología nueva, no una continuación de los baselines.

Pérdida posterior al calentamiento:

`L = SmoothL1_beta10(Q(s,a), y1) + SmoothL1_beta10(Q(sE,aE), y10) + 4 mean(max_a[Q(sE,a)+0.8*1(a!=aE)] - Q(sE,aE))`.

`y1 = r + .99*(1-terminal)*Vobjetivo(s')`; `y10 = sum_{k=0}^{n-1} .99^k*r[k] + .99^n*(1-terminal_final)*Vobjetivo(s_n)`, n<=10, sin cruzar episodios. En DQN V es máximo objetivo; en Double DQN se evalúa en objetivo el argmax principal. Targets sin gradientes. El término de diez pasos y margen usan solo demostraciones de entrenamiento. La red objetivo se copia al final de clasificación, cada 500 actualizaciones offline y cada 500 decisiones online; no confundir estas unidades. El replay online usa exclusivamente las acciones del agente, sin consultas PID.

Tope por candidato 150000 decisiones online; evaluación cada 10000. Se detiene al obtener >=80% tres veces consecutivas y al menos 30000 pasos online. V2 exige >=80% **en cada grupo**, no en el agregado. Mejor checkpoint: máxima tasa mínima por grupo; empate por menor error horizontal medio entre éxitos. La comprobación adicional nunca selecciona pesos. Todos los candidatos pararon a 30000, sin consumir el tope.

## Resultados y límites

| Candidata V2 (pasos del checkpoint) | Comprobación L1 | Comprobación alta | Validación original | Fallos en comprobación |
|---|---:|---:|---:|---|
| DQN42 (20000) | 63/64 | 61/64 | 32/32 | 1 invalid_contact (level1), 3 invalid_contact (original) |
| Double42 (30000) | 62/64 | 63/64 | 30/32 | 2 invalid_contact (level1), 1 invalid_contact (original) |
| DQN43 (30000) | 62/64 | 63/64 | 32/32 | 2 invalid_contact (level1), 1 invalid_contact (original) |

Candidata principal V2, medias de todos los contactos de comprobación (incluidos fallos):

| Grupo | |vx| m/s | vy m/s | |theta| rad | |omega| rad/s | Error x m | Gas Ns | Duración s |
|---|---:|---:|---:|---:|---:|---:|---:|
| level1 | 0.089 | -0.291 | 0.0080 | 0.0101 | 0.112 | 2105.9 | 2.160 |
| original | 0.095 | -0.451 | 0.0168 | 0.0194 | 0.096 | 6007.5 | 6.115 |

Primera intervención: DQN42 consiguió 63/64, 64/64, 62/64 en desarrollo a 10k/20k/30k; checkpoint20k cargado consiguió 64/64 en comprobación L1 y 0/32 originales (salidas laterales, sin contacto). Double42 y DQN43 también cumplieron tres evaluaciones L1 >=80%, pero sus últimos checkpoints dieron 51/64 y 53/64 en comprobación, y 0/32 originales. Esos comparativos V1 usan últimos checkpoints, no el mismo criterio de selección que la candidata primaria; no son una clasificación justa de algoritmos. Se conservan íntegros, incluidas degradaciones.

La segunda intervención respondió a **cero transferencia en validación original y falta de soporte de altura**, no ajustó la primera política usando su comprobación. Se congelaron nuevos datos y se entrenó desde cero manteniendo pérdidas/escalas/hiperparámetros. La transferencia de 0/32 a aterrizajes demuestra una mejora funcional observada; no aísla causalmente el mérito de cada ingrediente ni resuelve todas las condiciones del entorno.

La clasificación supervisada sola ya obtuvo 119/128 en desarrollo V2. No se demostró que TD mejore frente a imitación: el resultado válido es una política asistida que funciona en los conjuntos medidos. Las segundas semillas comparten demostraciones y distribución; no constituyen campaña estadística ni permiten afirmar superioridad general DQN/Double DQN. Viento, inclinaciones grandes y prueba final permanecen sin evaluar. Los cuatro fallos de la candidata en comprobación son contactos con |omega| por encima de 0,25 rad/s; no se relajó ese criterio.

Curvas: [V1](../artifacts/recovery_v1/report/learning.png), [V2 por grupo](../artifacts/recovery_v1/report_coverage/learning.png), [CartPole](../artifacts/recovery_v1/report/cartpole.png). Trayectorias: [PID/RL previo/candidata desde el mismo estado](../artifacts/recovery_v1/report/matched_trajectories.png), [V2: primer ID por grupo y primer fallo](../artifacts/recovery_v1/report_coverage/representative_trajectories.png). La selección de ejemplos se basa en orden de identificador, no en retorno. Todas las trayectorias evaluadas, incluidos fallos, están en CSV; las salidas sin contacto no se presentan como aterrizajes.

## Reproducción

Desde la raíz del repositorio, con las dependencias del proyecto instaladas en `.venv`:

```sh
# Suite y checker (sin volver a entrenar)
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_recovery_env.py

# Reproducción íntegra de la segunda metodología en directorio NUEVO.
# Incluye demos, DQN42, Double42 tras evidencia, DQN43, evaluaciones y curvas.
.venv/bin/python scripts/run_recovery_coverage.py --output artifacts/recovery_coverage_reproduction --max-seconds 7200 --max-steps 5000000

# Primera metodología, si se quiere reproducir su alcance limitado a L1.
.venv/bin/python scripts/run_recovery.py --output artifacts/recovery_l1_reproduction --max-seconds 7200 --max-steps 5000000

# Evaluar candidata entregada: red cargada, greedy, sin profesor ni updates.
.venv/bin/python scripts/evaluate_recovery.py --checkpoint artifacts/recovery_v1/candidate_coverage_v2.policy.pt --scenarios configs/recovery_v2_check_scenarios.json --output /tmp/recovery_check_reproduction.csv

# Visualizar solo dos episodios (requiere pantalla local).
.venv/bin/python scripts/evaluate_recovery.py --checkpoint artifacts/recovery_v1/candidate_coverage_v2.policy.pt --scenarios configs/recovery_v2_check_scenarios.json --episodes 2 --render --output /tmp/recovery_visual_reproduction.csv

# Regenerar tablas/curvas de los resultados ya guardados; cero pasos físicos.
.venv/bin/python scripts/report_recovery.py
.venv/bin/python scripts/report_recovery_coverage.py
```

Las salidas de evaluación deben ser nuevas. Los checkpoints `.policy.pt` son para **inferencia**, aunque contienen optimizador: no incluyen replay, datos docentes, RNG Torch ni estado parcial del entorno; no permiten reanudación fiel. Se debe ejecutar el protocolo desde cero, no anunciar `--resume`. El render está implementado pero no se verificó interacción de teclado real ni se afirmó una sesión visual real durante esta entrega.

Los orquestadores usan `THRUSTER_RECOVERY_ROOT` para aislar resultados y un presupuesto compartido secuencial. Los scripts individuales de entrenamiento requieren un directorio con `budget.json` y las demostraciones; los orquestadores preparan ese estado. No lanzar trabajadores concurrentes sobre un mismo ledger. La verificación pytest de la intervención contó también sus pasos mediante `recovery_pytest_budget`.

## Verificaciones y presupuesto

La suite final pasó **114 pruebas**, con dos avisos conocidos del checker por límites infinitos del Box de observación. El checker oficial también pasó por separado. Se verificaron targets numéricos DQN/Double y n-step, máscara terminal, ausencia de gradientes objetivo, margen, exclusión de etiquetas del holdout, transformación consistente, inferencia cargada sin PID y sin cambios de pesos/RNG propio, separación de manifiestos y límites duros de presupuesto. Los clasificadores independientes confirmaron todos los resultados/contactos de **480 episodios V1 y 864 V2**. Las configuraciones V2 solo difieren de V1 en soporte/protocolo agrupado; una prueba lo verifica.

Integridad: SHA-256 de **1118 archivos anteriores** coincide íntegramente con el inventario previo; se verificaron también ambos protocolos congelados. `preservation_verified.json`, `source_version.json`, `tests_coverage_final.log` y los manifiestos guardan la evidencia. El árbol ya tenía cambios antes de esta tarea: no se revirtieron ni se atribuyeron a esta intervención.

Presupuesto total cerrado: **828,266 decisiones de entorno de 5 000 000**, incluyendo auditoría, CartPole, PID, entrenamiento, validación, comprobaciones y pasos de pruebas/checker. Seis candidatos asistidos ×30000 =180000 decisiones online; el resto incluye los otros usos, no solo entrenamiento. Tiempo conservador desde el inicio registrado: **50.55 min de 120**, incluye 10 min reservados inicialmente para revisión. Suma de etapas de ejecución medidas: **7.60 min**; no confundirla con toda la interacción. `budget.json` registra cada intento y queda cerrado contra nuevos pasos. La auditoría inicial midió 41 decisiones PID en 0,0168 s y 156 RL en 0,0582 s; los candidatos V2 tardaron aproximadamente un minuto cada uno incluyendo sus evaluaciones, según sus metadatos. Estos tiempos son de esta máquina CPU, no una estimación universal.


Se completaron exactamente dos intervenciones metodológicas. Se detiene aquí: no se inició una tercera búsqueda, campaña de millones de pasos ni prueba final. El siguiente trabajo razonable es evaluar robustez en un protocolo nuevo y explícito, conservando esta candidata; no atribuir el éxito asistido a RL desde cero.
