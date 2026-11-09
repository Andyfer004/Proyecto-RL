# Evaluación común conforme al alcance — 2026-11-04

Se completó la comparación congelada de PID, DQN/Double DQN desde cero y **DQN y Double DQN asistidos por demostraciones PID**, con condiciones iniciales variadas y viento. Es un simulador físico académico propio 2D inspirado conceptualmente en el Tesla Roadster; no reproduce parámetros ni tecnología comerciales. No se añadieron 3D, hardware, navegación u otras funciones fuera de alcance.

La [matriz de trazabilidad](trazabilidad_alcance.md) enlaza compromiso, implementación, evidencia y pendientes. La asistencia por demostraciones se documenta como **ampliación metodológica adicional** respecto al enfoque inicial. Los baselines desde cero siguen siendo la evidencia de la metodología original y no lograron aterrizar en esta evaluación.

## Congelación, selección y validez

Antes de cualquier rollout final se creó `configs/scope_final_v1_protocol.json`, con hashes SHA-256 de los checkpoints, sus fuentes originales, configuraciones, manifiestos, evaluador y código físico. Se copiaron modelos a `artifacts/scope_alignment_v1/frozen_models/`. El protocolo y su propio hash se verificaron antes y después de evaluar. **No hubo entrenamiento, selección posterior ni reajustes**. La prueba ya se consumió: entrenamiento futuro deberá tener identificador nuevo y comprobación independiente.

DQN asistido42: mejor checkpoint de desarrollo, 20000 pasos online; Double42 y DQN43: mejores a 30000. La regla previa era máxima tasa mínima entre grupos y, en empate, menor error horizontal entre contactos exitosos. La segunda semilla se conserva como sensibilidad y no reemplaza retrospectivamente a la candidata principal. Comparten demostraciones: no son réplicas independientes de generación del profesor.

Baselines estándar: `artifacts/{dqn,double_dqn}_pilot_100k/best.policy.pt`, de campañas previas de 100000 decisiones, sin demostraciones y con exploración uniforme. Ambos mejores quedaron en 10000 pasos: todos los puntos de validación tenían cero éxitos y se conserva el más temprano por empate, según la regla original. Se verificó la selección contra sus metadatos. No se eligieron baselines por el resultado final. Permanecen disponibles los últimos checkpoints y todos los experimentos posteriores.

Los presupuestos y metodologías no son iguales entre baselines y asistidos: comparación de controladores existentes, **no ablación del beneficio de demostraciones ni estimación causal de superioridad de algoritmos**. Las tasas agregadas dependen de la mezcla fijada de grupos, no describen una distribución operacional universal.

El manifiesto reservado original `pid_test_scenarios.json` se mantuvo íntegro: 100 casos, vertical/offset/tilted/wind, 25 cada uno. Ya había dado 98/100 al PID en una etapa anterior; **no es prueba virgen para PID**. Esta es la primera ejecución neural sobre esos casos. El complemento nuevo es independiente de entrenamiento, desarrollo y comprobaciones previas para todos los métodos; exclusión por semillas y estados registrada antes de evaluar.

## Metodología realmente ejecutada

Inspección de `recovery_agent.py`, `train_recovery.py`, `metrics.csv` y contadores cargados, exportada en `artifacts/scope_alignment_v1/methodology_audit.json`:

| Checkpoint | Imitación CE | TD + margen offline | TD + margen online | Total |
|---|---:|---:|---:|---:|
| DQN asistido42 | 10000 | 10000 | 4969 | 24969 |
| Double DQN asistido42 | 10000 | 10000 | 7469 | 27469 |
| DQN asistido43 | 10000 | 10000 | 7469 | 27469 |

CE es entropía cruzada sobre acciones PID. Después se minimiza `SmoothL1_beta10(Q(s,a), y1) + SmoothL1_beta10(Q(sE,aE), y10) + 4*Margen_0.8(Q(sE),aE)`. El margen imita acciones; TD usa recompensa real y bootstrap objetivo, a uno y hasta diez pasos. DQN maximiza la red objetivo; Double usa selección principal y evaluación objetivo. Terminal físico anula bootstrap; truncamiento externo lo conserva. El n-step usa trayectorias docentes y no cruza resets. Hay registros no nulos de ambas pérdidas TD y del margen; no se infiere actividad TD únicamente del nombre del archivo.

La red ya aterrizaba después de CE en desarrollo. **No se ha demostrado que TD aporte mejora frente a imitación sola**. Se trata de políticas híbridas con TD ejecutado, no evidencia de RL puro funcional ni de superioridad de TD. La fórmula, estratificación y preentrenamiento completos permanecen en [recuperacion_agente.md](recuperacion_agente.md).

Durante inferencia actúa solo la MLP, argmax sin epsilon, sobre las mismas nueve observaciones físicas. La transformación asistida es invertible y no añade información. PID recibe el mismo vector float32 decodificado a SI (incluido viento observable), nunca `env.state` ni información futura; reinicia integrales por episodio. La memoria interna del PID es propia del método. Para las redes se bloquearon consultas `PIDController.act` y se verificó invariancia de pesos online/objetivo, optimizador, replay, contadores y RNG. Los contadores de consultas docentes y actualizaciones de evaluación son cero.

Todos usan masa100 kg, gas18000 Ns, ocho acciones, 0,05 s por decisión, límite25 s, misma geometría y criterios: abs(x)<=1,5 m, abs(vx)<=0,7 m/s, vy entre -1 y0 m/s, abs(theta)<=0,12 rad, abs(omega)<=0,25 rad/s, pies dentro de plataforma y segundo apoyo a <=0,15 m, cuerpo libre. No hay estados iniciales en contacto.

## Grupos predefinidos

Complemento `configs/scope_final_v1_complement.json`, 288 casos sin filtrar por resultados, generados por `prepare_scope_evaluation.py` con RNG por caso, semillas440000+ y offsets para pares de viento:

| Grupo | Rangos SI / origen | Relación con entrenamiento asistido |
|---|---|---|
| represented_l1, 32 | x±0,1; y1,6–2,4; vx±0,03; vy[-0,3;-0,05]; theta±0,01; omega±0,005; W0 | Dentro del soporte inicial, identidades nuevas |
| represented_high, 32 | x±0,3; y8–13; vx±0,1; vy[-1;0]; theta±0,03; omega±0,005; W0 | Dentro del soporte inicial, identidades nuevas |
| new_displacement, 32 | Rango alto salvo abs(x)0,6–1,5, ambos signos | Desplazamiento inicial nuevo |
| new_dynamics, 32 | Alto salvo abs(theta)0,06–0,12; abs(vx)0,2–0,5; vy[-1,8;-0,8]; omega±0,05 | Inclinación y velocidades nuevas |
| wind_control y cuatro vientos, 5×32 | Mismos 32 estados altos, W=0,-20,+20,-40,+40 N | El control está dentro de soporte; todo viento distinto de cero es nuevo |

Los rangos corresponden al diseño, no a extremos observados redondeados. Los baselines entrenaron únicamente descensos centrados verticales desde10–12 m sin viento: su soporte es más estrecho. Los grupos reservados mezclan estados dentro/fuera del soporte asistido; se etiqueta cada caso en CSV. «Dentro del soporte inicial» no afirma cobertura de todas las trayectorias ni generalización automática.

Viabilidad del viento: fuerza constante horizontal aplicada al centro de masa, observable, sin momento ni ráfagas. A40 N, la sustentación ideal requiere inclinación `atan(40/981)=0,04075 rad` y fuerza981,82 N, por debajo de3000 N y del umbral angular0,12. A1,8 m/s descendente, frenado vertical ideal con máximo empuje requiere0,0802 m, frente a más de6,9 m de separación inicial en los grupos altos. Estas cotas muestran autoridad física; no garantizan éxito de cada política ni incluyen todas las demoras/discretización. No se cambió el entorno para conseguir casos viables.

## Resultados conservados

| Método congelado | Reservados | Complementarios |
|---|---:|---:|
| PID | 98/100 | 287/288 |
| DQN asistido 42 | 96/100 | 275/288 |
| Double DQN asistido 42 | 94/100 | 266/288 |
| DQN asistido 43 (sensibilidad) | 100/100 | 277/288 |
| DQN desde cero | 0/100 | 0/288 |
| Double DQN desde cero | 0/100 | 0/288 |

| Grupo (n por método) | PID | DQN asistido 42 | Double asistido 42 | DQN asistido 43 | DQN/DDQN desde cero |
|---|---:|---:|---:|---:|---:|
| reserved: vertical (25) | 25 | 25 | 25 | 25 | 0 / 0 |
| reserved: offset (25) | 24 | 23 | 23 | 25 | 0 / 0 |
| reserved: tilted (25) | 24 | 24 | 25 | 25 | 0 / 0 |
| reserved: wind (25) | 25 | 24 | 21 | 25 | 0 / 0 |
| complement: represented_l1 (32) | 32 | 28 | 28 | 29 | 0 / 0 |
| complement: represented_high (32) | 31 | 32 | 28 | 31 | 0 / 0 |
| complement: new_displacement (32) | 32 | 30 | 28 | 32 | 0 / 0 |
| complement: new_dynamics (32) | 32 | 31 | 29 | 29 | 0 / 0 |
| complement: wind_control (32) | 32 | 31 | 31 | 32 | 0 / 0 |
| complement: wind_m20 (32) | 32 | 31 | 31 | 32 | 0 / 0 |
| complement: wind_p20 (32) | 32 | 31 | 31 | 31 | 0 / 0 |
| complement: wind_m40 (32) | 32 | 29 | 28 | 30 | 0 / 0 |
| complement: wind_p40 (32) | 32 | 32 | 32 | 31 | 0 / 0 |

No se descartó ningún escenario ni se sustituyeron los reservados. El viento negativo40 N redujo éxitos respecto al control emparejado en 2 casos para DQN42 y 3 para Double42; +40 N añadió uno respecto a su control de31/32. Esta asimetría de una muestra no demuestra robustez universal o ventaja general de un signo de viento.

| Método | Éxitos | Contacto inválido | Cuerpo | Fuera de plataforma | Techo | Salida lateral | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|
| PID | 385 | 3 | 0 | 0 | 0 | 0 | 0 |
| DQN asistido 42 | 371 | 17 | 0 | 0 | 0 | 0 | 0 |
| Double DQN asistido 42 | 360 | 27 | 1 | 0 | 0 | 0 | 0 |
| DQN asistido 43 (sensibilidad) | 377 | 11 | 0 | 0 | 0 | 0 | 0 |
| DQN desde cero | 0 | 47 | 193 | 43 | 99 | 6 | 0 |
| Double DQN desde cero | 0 | 29 | 262 | 97 | 0 | 0 | 0 |

Los 17 fallos de DQN42 y los11 de DQN43 incumplen omega al contacto. PID también tiene tres fallos por omega. Double42 tiene22 fallos solo por omega, cinco con otras violaciones cinemáticas y uno de cuerpo; los registros incluyen violaciones simultáneas. Los baselines presentan impactos, contactos fuera de plataforma y salidas del área; no se describen como controladores funcionales. No hubo límites temporales en esta evaluación.

## Contacto y eficiencia

| Método / resultado | n | Contactos | Gas medio Ns | Duración s | Media abs(vx) m/s | vy m/s | Media abs(theta) rad | Error horizontal m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PID / éxito | 385 | 385 | 5762.4 | 5.859 | 0.099 | -0.474 | 0.017 | 0.106 |
| PID / fracaso | 3 | 3 | 6521.0 | 6.609 | 0.158 | -0.435 | 0.005 | 0.006 |
| DQN asistido 42 / éxito | 371 | 371 | 5737.2 | 5.834 | 0.090 | -0.488 | 0.019 | 0.149 |
| DQN asistido 42 / fracaso | 17 | 17 | 4961.2 | 5.045 | 0.220 | -0.402 | 0.020 | 0.189 |
| Double DQN asistido 42 / éxito | 360 | 360 | 5731.5 | 5.821 | 0.094 | -0.461 | 0.020 | 0.152 |
| Double DQN asistido 42 / fracaso | 28 | 28 | 4919.9 | 4.995 | 0.415 | -1.383 | 0.134 | 0.440 |
| DQN asistido 43 (sensibilidad) / éxito | 377 | 377 | 5896.5 | 5.994 | 0.092 | -0.457 | 0.019 | 0.138 |
| DQN asistido 43 (sensibilidad) / fracaso | 11 | 11 | 4988.7 | 5.045 | 0.177 | -0.377 | 0.025 | 0.220 |
| DQN desde cero / fracaso | 388 | 283 | 6066.7 | 6.266 | 1.698 | -15.209 | 1.637 | 3.044 |
| Double DQN desde cero / fracaso | 388 | 388 | 879.5 | 1.440 | 1.767 | -12.872 | 1.596 | 1.824 |

Las velocidades, orientación y precisión de esta tabla incluyen **solo contactos** de cada estrato, también los impactos. Para el baseline DQN hay105 episodios sin contacto: sus estados finales no se usan como aterrizajes. El CSV `summary.csv` contiene además omega, medianas y percentiles90 absolutos, por método/grupo/éxito/fracaso/causa. `episodes.csv` conserva estado inicial/final, semilla, componentes de recompensa y métricas faltantes sin contacto.

Comparación emparejada únicamente donde ambos aterrizan: DQN42 consume65,56 Ns menos que PID en media (368 pares) y presenta0,0431 m más de error horizontal; Double42 consume63,27 Ns menos (357 pares), con0,0482 m más de error. Es un intercambio descriptivo de precisión/consumo sobre ese subconjunto, no una superioridad global. Las medias de consumo de fracasos no se interpretan como eficiencia de aterrizaje. Los subconjuntos de éxitos distintos tampoco son intercambiables sin este emparejamiento.

En esta ejecución PID fue más consistente que las candidatas primarias; DQN/Double asistidos consiguieron aterrizajes en todos los grupos, con fallos residuales conservados. La semilla43 obtuvo mejores tasas aquí, pero no autoriza a elegirla después de la prueba ni a concluir superioridad estadística. El objetivo de construir y comparar controladores 2D se materializa; demostrar aprendizaje autónomo desde cero y beneficio causal de TD sigue sin resolverse.

## Artefactos, animación y comandos

Todos los resultados: `artifacts/scope_alignment_v1/final/`. Tablas descubiertas también en `docs/resultados/alcance_*`. [Éxito por grupo](../artifacts/scope_alignment_v1/final/success_by_group.png), [contacto y gas](../artifacts/scope_alignment_v1/final/contact_and_fuel.png), [trayectorias con fallos](../artifacts/scope_alignment_v1/final/representative_trajectories.png). Los ejemplos son primeros IDs predefinidos, no mejores episodios; los baselines fallidos permanecen en las figuras.

Animaciones GIF desde las trazas guardadas, primer caso +40 N predefinido: [PID](../artifacts/scope_alignment_v1/final/pid.complement-wind_p40-000.gif), [DQN asistido](../artifacts/scope_alignment_v1/final/dqn_assisted42.complement-wind_p40-000.gif), [Double asistido](../artifacts/scope_alignment_v1/final/double_assisted42.complement-wind_p40-000.gif). Se exportaron con el renderer RGB del proyecto bajo SDL dummy, sin pasos de entorno adicionales. No se verificó teclado físico ni ventana real. La duración GIF está remuestreada a10 fps; no es una medición del tiempo de cómputo.

```sh
# Verificaciones (no entrenamiento)
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_recovery_env.py

# Reproducir el MISMO protocolo y modelos congelados, salida nueva.
# No debe utilizarse para reajustar y volver a declarar independencia.
.venv/bin/python scripts/evaluate_scope.py --output /tmp/scope_final_reproduction
.venv/bin/python scripts/report_scope.py --results /tmp/scope_final_reproduction
.venv/bin/python scripts/animate_scope.py --results /tmp/scope_final_reproduction

# Reproducir estados guardados en ventana local, sin volver a evaluar.
.venv/bin/python scripts/animate_scope.py --human --methods pid dqn_assisted42 double_assisted42
```

`prepare_scope_evaluation.py` se ejecutó una vez antes de evaluar; rechaza sobrescribir protocolo o complemento. El entrenamiento no forma parte de estos comandos. Los comandos de entrenamiento asistido de la etapa anterior se conservan como reproducibilidad histórica, no como autorización para reajustar con esta prueba. Los checkpoints asistidos son de inferencia, no reanudación fiel.

## Verificaciones y pendientes expresos

118 pruebas aprobadas, incluidas cuatro nuevas sobre decodificación de observación PID, agregación con contacto ausente y consumo separado, rechazo de archivos congelados modificados e inferencia greedy con profesor bloqueado. Checker Gymnasium válido; dos avisos conocidos por Box de cotas infinitas. Se cotejaron independientemente geometría y umbrales en los2328 resultados y se comprobaron identidades de casos/estado inicial/viento entre métodos. Los hashes anteriores y de protocolo se verificaron al cierre.

Ejecución común:242465 decisiones físicas,2328 episodios,82,98 s en esta CPU; cero actualizaciones de entrenamiento. Este coste es adicional y separado del presupuesto cerrado de la intervención anterior. No se lanzó una campaña de entrenamiento ni se modificaron modelos tras ver la prueba.

Pendiente de comprobación local: prueba de teclado/ventana real en el equipo. Limitaciones experimentales: una semilla principal por algoritmo, una semilla de sensibilidad con demos compartidas, viento constante hasta±40 N, grupos finitos y parámetros académicos. No se ha evaluado hardware, ráfagas, estados extremos, 3D ni una distribución real. Futuras modificaciones exigirán experimento nuevo y evaluación independiente.
