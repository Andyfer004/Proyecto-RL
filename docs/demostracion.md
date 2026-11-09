# Demostración de la versión entregable

Esta demostración reproduce evidencia ya guardada. No entrena, no reajusta ni vuelve a consumir la prueba. Los números citados corresponden al protocolo congelado; no a una ejecución nueva de la presentación.

## Preparación

1. Abrir [informe_final.md](informe_final.md), [trazabilidad_alcance.md](trazabilidad_alcance.md) y [demostracion.html](demostracion.html). La galería HTML usa solo archivos GIF locales y funciona sin servidor ni conexión.
2. Desde la raíz, comprobar integridad y cifras: `.venv/bin/python scripts/verify_delivery.py`. El paquete ZIP ofrece el mismo comando usando su entorno instalado.
3. Mantener visible la tabla por condición: [alcance_resultados.csv](resultados/alcance_resultados.csv) y la figura [alcance_exitos.png](resultados/alcance_exitos.png).

## Guion de exposición, aproximadamente ocho minutos

- **Problema y alcance:** cuerpo rígido2D académico, inspirado conceptualmente en Tesla Roadster. No es modelo comercial, hardware ni simulación3D.
- **Física y acciones:** explicar gravedad, fuerza rotada, momento lateral y presupuesto de 18000 N·s. N·s no son kg de gas. Hay ocho combinaciones a 20 Hz.
- **PID:** posición→velocidad horizontal→inclinación; altura→velocidad vertical→fuerza con gravedad; orientación→momento. Se asigna al catálogo por error normalizado, con saturación y anti-windup.
- **Métodos aprendidos:** distinguir desde cero, imitación sola y DQN/Double asistidos. Mostrar los contadores CE/TD y explicar que las redes finales recibieron ambas clases de actualización. En inferencia solo actúa la red.
- **Animación:** abrir la galería y reproducir el primer caso predefinido de viento+40 N para PID y dos candidatas. Es un ejemplo fijo, no prueba suficiente por sí solo. Revisar también la figura de trayectorias, que conserva fallos de baselines.
- **Resultados completos:** PID98/100 y 287/288; DQN asistido 96/100 y 275/288; Double asistido 94/100 y 266/288. Baselines seleccionados0. Revisar grupos de inclinación/viento y causas, no únicamente el agregado.
- **Consumo y conclusión:** comparar pares donde ambos aterrizan, no gas ahorrado al impactar. PID fue más consistente. No se aisló el beneficio de las actualizaciones TD frente a la imitación por sí sola. Una red congelada no se adapta automáticamente.
- **Límites y metodología:** asistencia PID como componente adicional; pocas semillas; física ideal; viento constante; sin vehículo real ni teclado físico verificado.

## Animaciones y regeneración opcional

Los GIF enlazados se exportaron desde estados guardados, con SDL dummy. Se inspeccionaron sus imágenes y múltiples frames, no una ventana real. Se pueden regenerar sin nuevos pasos físicos:

```sh
.venv/bin/python scripts/animate_scope.py
```

Incidencia conocida: el script congelado también asigna SDL dummy por defecto con `--human`. Por eso no se usa ese comando como demostración de ventana local verificada. Si el usuario desea ensayar la ventana en macOS, puede establecer explícitamente el driver correspondiente; no se declara probado aquí:

```sh
SDL_VIDEODRIVER=cocoa .venv/bin/python scripts/animate_scope.py --human --methods pid dqn_assisted42 double_assisted42
```

La galería HTML es la ruta recomendada para la presentación entregable y no depende de ese driver. Cualquier problema de interacción real permanece documentado, sin afectar o modificar resultados finales.
