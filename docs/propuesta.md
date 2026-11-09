# Control inteligente de propulsores para la estabilización y el aterrizaje de un vehículo mediante aprendizaje por refuerzo

Pregunta: ¿cómo se comparan DQN y Double DQN con un PID en éxito, precisión y
consumo de propulsor, bajo condiciones iniciales y viento comunes?

Objetivo general: desarrollar y evaluar controladores para estabilizar y aterrizar
un vehículo 2D con propulsores y recursos limitados. Objetivos específicos: construir
el simulador verificable; formular su estado, acciones y recompensa; implementar
PID, DQN y Double DQN; compararlos con escenarios y métricas comunes; documentar
resultados, limitaciones y reproducibilidad.

Alcance: cuerpo rígido plano, plataforma fija, gravedad, empuje discreto, rotación,
gas limitado, viento horizontal, ejecución sin gráficos y visualización opcional.
Se excluyen 3D, despegue, navegación, hardware, recipientes físicos, aerodinámica
detallada, imágenes, varios vehículos y plataformas móviles.

## Organización sugerida para tres estudiantes

1. Dinámica, entorno y pruebas físicas; revisión de geometría y contacto.
2. PID y ejecución de validación; apoyo a entrenamiento DQN/Double DQN.
3. Protocolo experimental, métricas, gráficas y documentación; apoyo a algoritmos.

Las revisiones deben ser cruzadas: quien modifica una ecuación explica unidades,
signos y pruebas a otro integrante. No se asignan tareas a personas reales desde
este repositorio; esta división sirve como propuesta del equipo.

## Hitos al cierre de la Etapa 3 (registro histórico)

- Etapa 1 implementada: simulador, Gymnasium, configuración, ejecución aleatoria y
  manual, visualización y pruebas. La visualización interactiva requiere verificar
  teclado y ventana en el equipo del usuario; las pruebas automáticas usan SDL dummy.
- Etapa 2 implementada: PID en cascada, 31/32 éxitos en validación y 98/100 en
  prueba reservada; parámetros, condiciones y fallos en `pid.md` y `resultados.md`.
- Etapa 3 implementada: DQN/Double DQN compartidos, checkpoints completos y de
  inferencia, pruebas de reanudación y entrenamientos cortos. Ninguno aterrizó
  en validación tras 10000 pasos. Diagnóstico y alcance en `dqn.md`.
- Etapa 4 pendiente: evaluación común, tablas y gráficas.
- Etapa 5: documentación inicial incluida; completar resultados tras los experimentos.

No comenzar entrenamientos largos hasta que el PID confirme controlabilidad práctica.

## Seguimiento del alcance — 2026-11-04

Los objetivos y exclusiones anteriores se mantienen como contrato. La
[matriz de trazabilidad](trazabilidad_alcance.md) contrasta su cumplimiento con
código y evidencia, y el [informe de evaluación común](evaluacion_alcance.md)
cierra la comparación sobre escenarios congelados.

La inspiración Tesla Roadster es conceptual: simulador y parámetros académicos
propios en 2D, sin reproducir física comercial.

Las políticas funcionales se denominan **DQN y Double DQN asistidos por
demostraciones PID**: combinan imitación y TD. Esta asistencia amplía la metodología
inicial como componente adicional; no sustituye la evidencia de los baselines
DQN/Double DQN desde cero. Tampoco
se ha demostrado que TD mejore frente a imitación sola. No se modificaron física,
acciones, gas ni criterios de éxito para obtener aterrizajes.
