# Entrega v1 — instrucciones de recepción

Guía de la publicación Git: [REPOSITORIO.md](REPOSITORIO.md).

Archivo principal: [informe final](../docs/informe_final.md). Demostración sin simulación: abrir [galería HTML](../docs/demostracion.html). Guion: [demostracion.md](../docs/demostracion.md). Contrato: [propuesta](../docs/propuesta.md) y [matriz](../docs/trazabilidad_alcance.md).

El ZIP `artifacts/delivery_v1/proyecto_rl_entrega_v1.zip` contiene una raíz `Proyecto-RL/` con fuentes, scripts, pruebas, configuraciones, documentación, resultados finales, modelos seleccionados, trazas y evidencia de desarrollo asistido. Contiene `MANIFEST_ENTREGA.json` con SHA-256 de cada archivo. El archivo `.sha256` contiguo verifica el ZIP completo. No incluye entornos virtuales, Git, credenciales ni la totalidad de checkpoints/replays de campañas históricas.

Tras extraer, entrar a esa raíz, instalar dependencias según README y ejecutar:

```sh
.venv/bin/python scripts/verify_delivery.py
```

La verificación no entrena ni reevalúa. Se puede inspeccionar todo el informe y la galería sin instalar Python. La entrega inicial incluyó Markdown+HTML+CSV+PNG+GIF y código. La conversión posterior a PDF está en [output/pdf/informe_final.pdf](../output/pdf/informe_final.pdf), con 23 páginas renderizadas y revisadas. No se generó Word.

Los modelos finales no se modificaron después de probar. La familia funcional se denomina **DQN y Double DQN asistidos por demostraciones PID**. Se preservan baselines desde cero con cero éxitos y la comparación supervisada de desarrollo, sin inventar una mejora de TD.

Pendientes explícitos:

- Interacción real de teclado/ventana: SDL dummy y GIF no la acreditan. Incidencia de driver del script congelado documentada en la demostración.
- Aprendizaje funcional desde cero, beneficio aislado de TD, robustez general y adaptación automática: no demostrados; investigación futura requeriría experimento nuevo y evaluación independiente.

Las notas históricas pueden mencionar otros `artifacts/` no distribuidos. La selección versionada y la entrega cubren las dependencias del protocolo final y evidencia de recuperación, no pretende empaquetar todos los replays ni habilitar resume fiel de los agentes asistidos.
