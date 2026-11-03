# Evidencia seleccionada del avance

Cifras contrastadas el 2 de noviembre de 2026 con los CSV y metadatos locales,
sin ejecutar nuevos experimentos. [experimentos.csv](experimentos.csv) resume
los conjuntos y resultados finales. [pid_evaluaciones.csv](pid_evaluaciones.csv)
conserva una selección de columnas de los 32 casos de validación y 100 de prueba
PID, incluidos sus fallos. El [resumen PID original](../pid_test_summary.csv)
permanece en su ruta porque forma parte de las referencias congeladas por hash.
Las demás tablas pequeñas de `docs/` conservan sus rutas para mantener referencias
en informes y scripts; no se publican las trazas originales extensas.

## Resultados y experimentos sin aterrizajes RL

| Experimento | Pasos por agente | Validación final DQN / Double DQN | Resultado relevante |
|---|---:|---|---|
| Integración corta | 10 000 | 0/32 / 0/32 | Verifica integración; no demuestra aterrizaje |
| Exploración lenta | 100 000 | 0/32 / 0/32 | Ambos salen por techo; deriva optimista de Q |
| LR=0.0001 | 100 000 | 0/32 / 0/32 | Persiste el problema de control y calibración |
| SmoothL1 beta=10, LR=0.0001 | 100 000 | 0/32 / 0/32 | DQN sale por techo; Double DQN: 27 fuera de plataforma y 5 contactos inválidos |
| Currículo, nivel 1 | 50 000 | 0/128 / 0/128 | Parada predefinida; en nivel 1, 0/32 cada uno; no se entrenaron niveles 2/3 |

Cero aterrizajes en todos los registros de entrenamiento y validación RL
revisados. Una semilla por algoritmo (42); los casos fijos de validación se
repiten entre puntos. En currículo, los 128 casos finales agrupan 32 originales y
32 por cada nivel 1/2/3; no son 128 casos de nivel 1. No se ejecutó prueba final RL.
El diagnóstico offline es una prueba de capacidad e interferencia sin episodios
nuevos; sus ajustes supervisados no son políticas de aterrizaje.

El PID obtuvo 31/32 en validación y **98/100 en prueba reservada**; 25/25 vertical,
24/25 desplazamiento, 24/25 inclinación y 25/25 viento. Los dos fallos superaron
el límite de velocidad angular. Los conjuntos PID y RL son diferentes.

## Figuras seleccionadas

Trayectorias PID sobre la prueba reservada; verde éxito, rojo fallo:

![Trayectorias PID de los 100 casos de prueba](pid_trayectorias.png)

Ablación beta=10 frente a beta=1, ambas con LR=0.0001. El panel de éxito sigue
en cero; las curvas de TD y duración no demuestran aterrizaje:

![Validación, consumo y diagnóstico de la ablación beta](beta10_control.png)

Currículo detenido al terminar nivel 1. Validación e historial de entrenamiento
muestran cero éxitos. Se inspeccionaron también niveles 2/3 sin entrenarlos:

![Curvas del currículo y magnitudes Q y TD](curriculum_curvas.png)

## Procedencia y archivos locales excluidos

[procedencia.json](procedencia.json) es un **registro local de auditoría**: hashes
de las fuentes consultadas, incluidas figuras copiadas sin alterar. Las rutas
`artifacts/` de ese registro no vienen incluidas en el repositorio. Tampoco se
incluyen los archivos mencionados por los manifiestos históricos
`configs/reference*.json` ni se garantiza que sus hashes antiguos correspondan
al código actual: algunos congelan versiones anteriores a nuevas etapas.

Los checkpoints, replay buffers, estados de reanudación, trazas por paso,
logs completos y figuras restantes se conservan localmente bajo `artifacts/`,
excluidos mediante `.gitignore`. Las tablas y figuras de esta carpeta sí están
publicadas. Los comandos de los informes que leen rutas históricas necesitan
esos archivos locales; los comandos básicos de README generan salidas nuevas.
