# Metodología implementada y cierre experimental

La metodología actual está descrita con ecuaciones y etapas en [informe_final.md](informe_final.md); el protocolo final, datos y criterios en [evaluacion_alcance.md](evaluacion_alcance.md). El archivo original de propuesta se conserva en [propuesta.md](propuesta.md), con seguimiento del alcance.

Se implementaron PID en cascada, DQN/Double DQN desde cero y una ampliación asistida por demostraciones PID. Las familias se mantienen separadas. Los baselines usaron TD y exploración; los asistidos tuvieron CE inicial, TD/margen offline y TD/margen online. Las políticas finales recibieron ambas clases de actualización y en inferencia actúa únicamente la red. No se aisló el beneficio de TD frente a imitación sola. La asistencia se documenta como ampliación adicional de la metodología.

Los parámetros/checkpoints se seleccionaron con desarrollo/validación, antes de evaluar 100 casos reservados y 288 complementarios. La reserva histórica ya había sido usada para PID; el complemento fue independiente y no sustituyó esos casos. Se usaron las mismas observaciones disponibles, física, gas, acciones, frecuencia y éxito para todos. No se entrenó ni reajustó con los resultados finales.

Las métricas distinguen éxito, contacto, causa de fracaso y estados finales sin contacto. El consumo se separa por resultado y se compara también en pares comunes exitosos. Variación entre escenarios no equivale a variación entre semillas: hay una principal por algoritmo y una sensibilidad adicional DQN con demostraciones compartidas, no una campaña de tres semillas.

Las propuestas iniciales de tres semillas y 100 casos por condición no se ejecutaron como tal; no se describen como cumplidas. El protocolo realizado y sus tamaños concretos sustituyen cualquier previsión antigua sobre el estado de la implementación, sin ocultar la diferencia. Nuevos entrenamientos serían otro experimento y precisarían una comprobación independiente.

Comandos de presentación y reproducción de trazas en [demostracion.md](demostracion.md); inventario de distribución y pendientes en [entrega/README.md](../entrega/README.md).
