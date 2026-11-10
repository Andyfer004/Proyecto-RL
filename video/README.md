# Videos de demostración

Ocho clips 1920×1080 a 20 fps, la misma frecuencia de decisión del entorno, así que van en
tiempo real. Se generan con [`scripts/render_demo_videos.py`](../scripts/render_demo_videos.py)
a partir de las trazas congeladas de `artifacts/scope_alignment_v1/final/`: **no se simula
ni se evalúa nada nuevo al renderizarlos**.

Cada panel dibuja las cinco ventanas de éxito con su banda de tolerancia y la aguja del
valor actual, que se pone roja al salirse. El medidor de ω lleva las marcas del cuanto de
0.477876 rad/s que imprime un solo pulso lateral de 0.05 s. La cámara se acerca por debajo
de 6 m: cambia la escala del encuadre, nunca la geometría.

## Comparativos

Los cuatro controladores sobre el mismo caso, con reloj compartido.

| Archivo | Caso | Qué pasa |
|---|---|---|
| `1_comparativo_vertical.mp4` | `test-vertical-000` | PID y ambos asistidos aterrizan; el DQN desde cero sale por el techo |
| `2_comparativo_inclinado_rapido.mp4` | `complement-new_dynamics-000` | Entrada inclinada y rápida |
| `3_comparativo_viento_mas40.mp4` | `complement-wind_p40-000` | Viento de +40 N, no visto en entrenamiento |
| `4_comparativo_desplazado.mp4` | `test-offset-008` | El DQN asistido falla y el PID aterriza |

## Individuales

Un controlador a pantalla completa, con la telemetría extendida.

| Archivo | Método y caso | Para qué |
|---|---|---|
| `5_asistido_falla_por_omega.mp4` | DQN asistido, `test-offset-008` | Cuatro ventanas en verde y ω en rojo a −0.3254 rad/s |
| `6_desde_cero_sale_por_el_techo.mp4` | DQN desde cero, `test-vertical-000` | La política que nunca aprendió |
| `7_pid_con_viento.mp4` | PID, `complement-wind_p40-000` | El baseline clásico con viento |
| `8_asistido_inclinado.mp4` | DQN asistido, `complement-new_dynamics-000` | Recuperación de una entrada difícil |

## Regenerar

Desde la raíz del repositorio, con las dependencias instaladas:

```sh
python scripts/render_demo_videos.py --scenario test-vertical-000 --output video/1_comparativo_vertical.mp4
python scripts/render_demo_videos.py --scenario test-offset-008 --mode individual \
    --methods dqn_assisted42 --output video/5_asistido_falla_por_omega.mp4
```

El script acepta cualquiera de los 388 casos evaluados por su `scenario_id` y cualquier
combinación de métodos, así que se puede renderizar un caso concreto si hace falta.
Requiere `imageio-ffmpeg` además de las dependencias del proyecto.
