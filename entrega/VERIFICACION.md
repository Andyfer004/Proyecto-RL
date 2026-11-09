# Verificación del cierre entregable

Fecha: 2026-11-04. No se ejecutaron nuevos entrenamientos, inferencias de evaluación ni simulaciones para este cierre.

- Protocolo, checkpoints, CSV finales y evidencias de recuperación contrastados con el inventario previo del cierre; SHA-256 sin cambios.
- 2328 filas finales comprobadas, mismos 388 casos por seis métodos; éxitos cotejados por conjunto.
- Contadores seleccionados CE/TD cotejados con checkpoints y auditoría; consumo emparejado verificado numéricamente.
- Ejemplos independientes de inercia, autoridad, pulso angular, targets DQN/Double y SmoothL1 comprobados sin optimización.
- Rutas Markdown/HTML y recursos locales de demostración comprobados; no se afirma prueba interactiva de navegador o teclado real.
- `pip check`: sin dependencias incompatibles.
- ZIP extraído a un directorio temporal: `verify_delivery.py` pasó usando las fuentes extraídas, no las del workspace original. El empaquetador verifica también cada archivo contra su manifiesto y el CRC del ZIP.

La última suite completa existente es 118 pruebas aprobadas y checker válido; no se relanzó porque contiene optimizaciones de agentes. Los logs de solo lectura de este cierre están en `artifacts/delivery_v1/verification.txt` y `extracted_verification.txt` en el workspace. El ZIP incluye `verification.txt`.

Se conserva documentada la incidencia de SDL dummy por defecto en el script histórico de animación, sin cambiar su hash ni los resultados físicos. La demostración entregable usa GIF locales. No se exportaron PDF/Word, por lo que no se declara un renderizado de esos formatos.
