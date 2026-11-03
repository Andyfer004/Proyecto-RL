# Etapa 2: PID en cascada y validación de aterrizajes

> Archivo técnico de una etapa del proyecto. Las rutas `artifacts/` citadas son
> registros locales excluidos de Git; no vienen incluidas en el clon. La evidencia
> publicada y el estado actual están en [resultados/README.md](resultados/README.md).

## Revisión de la capacidad de control

Se conservan sin cambios el entorno, física, ocho acciones, frecuencia de decisión,
recompensa, masa, gas y umbrales de Etapa 1. La tabla de `diseno.md` se verifica
para las ocho acciones en las pruebas nuevas: empuje local vertical de
0/1800/600/600/1200/2400/2400/3000 N y momentos de
0/0/-360/+360/0/-360/+360/0 N m. La gravedad requiere 981 N de sustentación.
Con todos los motores la aceleración neta ascendente es 20.19 m/s²; el central
solo produce 8.19 m/s². La fuerza horizontal del propulsor es -T sinθ:
θ positivo desplaza hacia la izquierda. Los dos signos de momento permiten
inclinar, frenar la rotación y volver hacia vertical. La fuerza y el momento
están acoplados: no existe un actuador horizontal independiente ni torque puro.

La resolución importa: un lateral genera 360/37.6667=9.5575 rad/s² y un pulso
completo de 0.05 s cambia ω en 0.477876 rad/s. Es mayor que el umbral de éxito
0.25 rad/s. Con gas suficiente, ω en los instantes de decisión vive en la red
ω_inicial+n·0.477876; para estados iniciales arbitrarios no se puede imponer
exactamente ω=0. El contacto puede interrumpir un pulso. Corregir un error pequeño
justo al tocar puede invalidar el aterrizaje aunque posición y ángulo sean buenos.

### Presupuesto de impulso

En descenso vertical: J=m(v_final-v_inicial+g·T). Con inclinación, el impulso total
necesario es al menos el vertical y aumenta aproximadamente como 1/cosθ cuando
la inclinación es constante. La rotación diferencial también condiciona el gasto.

Ejemplo continuo ideal con y inicial=11 m (10 m de recorrido de los apoyos),
vy inicial=-1 m/s, velocidad de crucero=-2 m/s y contacto=-0.4 m/s:

1. Acelerar hacia abajo a -2 m/s² durante 0.5 s: recorrer 0.75 m con 781 N.
2. Crucero durante 3.425 s: recorrer 6.85 m con 981 N.
3. Frenar a +0.8 m/s² durante 2 s: recorrer 2.4 m con 1061 N.

Tiempo 5.925 s; impulso 5872.425 N s, frente al presupuesto 18000 N s.
Son fuerzas medias ideales realizables aproximadamente con pulsos; no es una
predicción exacta del PID cuantizado. En prueba, el consumo medio por condición
fue de 6257–6511 N s, incluyendo fallos. Existe margen en estas condiciones,
pero sostenerse indefinidamente consume todo el presupuesto en unos 18.35 s.

## Implementación y unidades

Código: `src/thruster_landing/agents/pid.py`; configuración: `configs/pid.json`.
`PIDController.act(State, wind_force)` lee únicamente variables observables en SI.
Devuelve un índice del catálogo; no posee referencia al entorno ni escribe su
estado. En el CLI actúa una vez cada `decision_dt` (0.05 s). `reset()` limpia
integrales, medición anterior y diagnósticos antes de cada episodio.

- Posición horizontal: vx*=clip(-k_x x, ±v_x_max).
- PID de velocidad horizontal: a_x*=clip(k_p(vx*-vx)+I-k_d·a_x_medido,
  ±a_x_max). La aceleración medida es diferencia de velocidades dividida por dt.
- Descenso: h=max(y-foot_drop,0); vy*=-min(v_max,
  sqrt(v_contacto²+2 a_frenado h)). No se fuerza vy*=0: se evita quedar flotando
  justo encima del contacto.
- PID de velocidad vertical produce a_y*; la solicitud de fuerza incorpora
  gravedad: Fy*=m(g+a_y*). Se limita a [0,T_max·max(cosθ,0)].
- Fx*=m a_x*-W si la compensación de viento está habilitada. W es la misma fuerza
  observable del entorno, sin conocimiento de perturbaciones futuras.
- θ*=clip(-atan2(Fx*,max(Fy*,0.1)), ±θ_lim(h)). θ_lim interpola entre 0.04 rad
  cerca del suelo y 0.2 rad sobre 3 m. Esto limita la maniobra horizontal durante
  la aproximación final, sin modificar el criterio de éxito.
- PID de orientación: α*=clip(k_p wrap(θ*-θ)+I-k_d ω, ±τ_max/I_cuerpo).
  El momento solicitado es τ*=I_cuerpo α*.

Derivada de la medición, sin salto derivativo por cambio de consigna. Ganancias
horizontales/verticales: kp en s^-1, ki en s^-2, kd adimensional; orientación:
kp en s^-2, ki en s^-3, kd en s^-1. El lazo externo de posición tiene kp en s^-1.
Las contribuciones integrales están en unidades de aceleración (m/s² o rad/s²).
`integral_limit` fija una cota numérica para esas contribuciones en cada lazo.
Por defecto orientación es PD (ki=0) y los lazos de velocidad son PI (kd=0).
Todos admiten las tres ganancias en configuración.

La integral se actualiza con ki·error·dt y se recorta. Se congela si el error
empuja más hacia una saturación continua y puede descargarse si cambia de signo.
Sin gas se devuelve apagado y se reinicia el controlador para no integrar sin
autoridad. Este anti-windup es local a cada lazo y su cota: no implementa una
retropropagación exacta de la saturación de inclinación ni del error de asignación
discreta. La cuantización sigue produciendo ciclos límite, visibles en las gráficas.

## Asignación a acciones

Para cada acción se obtiene la fuerza **global** y el momento reales con
`wrench(action, θ, config)`. Si G no alcanza para un período completo, se multiplican
por q=min(1,G/(caudal_impulso·dt)): representan su contribución media disponible.
No se modifica ni se reparte el gas desde el controlador; eso sigue a cargo de
la física. Con G=0 se elige 0.

Se minimiza:

`w_x ((Fx_a-Fx*)/T_max)² + w_y ((Fy_a-Fy*)/T_max)² + w_tau ((τ_a-τ*)/τ_max)²`.

Los denominadores son las autoridades nominales en N y N m; no cambian cuando
se acaba gas. Pesos configurables por defecto 0.5,1,4. Desempate determinista por
menor índice. No hay pulsos de duración arbitraria: se aplica la acción completa
hasta la próxima decisión, salvo que la física agote gas o termine el episodio.
La asignación usa orientación actual; no es una búsqueda predictiva de trayectorias.

## Escenarios y ajuste

Los manifiestos se materializaron antes del ajuste mediante
`scripts/make_pid_scenarios.py`; ese script se niega a sobrescribirlos.
`pid_validation_scenarios.json`: ocho escenarios por condición, semillas
1000–1007/1100–1107/1200–1207/1300–1307. `pid_test_scenarios.json`: 25 por condición,
semillas 9000–9024/9100–9124/9200–9224/9300–9324. No comparten semillas.
Cada manifiesto fija valores concretos, no solo rangos de un generador.

| Condición | Estado y perturbaciones |
|---|---|
| vertical | x=vx=θ=ω=0; y U(10,12) m; vy U(-1,0) m/s; sin viento |
| offset | lo anterior más x U(-0.8,0.8) m y vx U(-0.15,0.15) m/s |
| tilted | offset más θ U(-0.18,0.18) rad y ω U(-0.03,0.03) rad/s |
| wind | tilted más fuerza horizontal constante U(-40,40) N |

Ajuste manual únicamente con validación. La primera versión tenía ki vertical=0.1
e integral_limit=1: la cuantización sesgaba la velocidad hacia arriba, prolongaba
el vuelo a unos 18.4 s y agotaba el gas. Tras revisar estados y gas, se compararon
tres candidatos sobre los 24 casos sin viento:

| Candidato | Diferencias frente a la versión inicial | Éxitos / 24 |
|---|---|---|
| integral (elegido) | ki vertical=2, integral_limit=5 | 23 |
| fast | anterior + kp vertical=6, contacto=0.6, kd orientación=8 | 14 |
| strong | fast + kp orientación=120, kd orientación=10 | 18 |

Se mantuvieron los demás valores iniciales. La validación completa del elegido
obtuvo 31/32; el viento posterior obtuvo 8/8. No se ajustó usando la prueba final.
No se cambiaron física ni umbrales para conseguir éxitos. Parámetros finales
congelados antes de abrir prueba: hash SHA256 de la configuración serializada
ordenando claves `8aa3ded23f55e2a6f9b4c5666ffad0742777b4695b214776b51eaa240969cf54`.

## Ejecución y evidencia

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,render,analysis]'
python -m pytest
python scripts/check_env.py
python scripts/play.py --controller pid --config configs/env.json --pid-config configs/pid.json --scenarios configs/pid_validation_scenarios.json --trace --output artifacts/pid_validation.csv
python scripts/play.py --controller pid --config configs/env.json --pid-config configs/pid.json --scenarios configs/pid_test_scenarios.json --trace --output artifacts/pid_test.csv
python scripts/plot_pid.py artifacts/pid_test.csv
python scripts/play.py --controller pid --pid-config configs/pid.json --scenarios configs/pid_validation_scenarios.json --episodes 3 --render --output artifacts/pid_visual.csv
```

Sin `--render` no se importa Pygame. Sin manifiesto, `--seed` y `--episodes` usan
los rangos del entorno. Con manifiesto, las semillas y opciones proceden del
archivo; `--episodes` puede limitar al prefijo, y se ejecutan todos por defecto.
Se registran configuración completa, versiones, manifiesto y hashes en metadatos.
Para repetir las versiones: instalar `configs/requirements-stage2.txt`, seguido de
`python -m pip install --no-deps -e .` en Python 3.12.

El CSV por episodio contiene semilla/condición/estado inicial, resultado exacto,
retorno, consumo, duración, error horizontal final y altura final del centro.
Las columnas `contact_*` quedan vacías si no hubo contacto. Las causas separan
éxito, contacto inválido, cuerpo, fuera de plataforma, salida del área y tiempo.
`.summary.csv` separa todos los episodios, contactos y éxitos, indicando n.
`.traces.csv` guarda estados al final de cada intervalo; acción y consignas
corresponden al intervalo **previo**, calculadas desde su estado inicial. La fila
t=0 no tiene acción. Si una sesión se interrumpe, las gráficas filtran los episodios
incompletos; no se cuentan como fallos ni éxitos.

Gráficas: todas las trayectorias de centro de masa, coloreadas por éxito/fallo;
estados del **primer** escenario de cada condición, sin elegir los mejores;
y consumo/acciones de esos mismos escenarios. Los umbrales dibujados solo se
exigen al contacto. Los datos y gráficas generados siguen fuera de Git. La tabla
resumida de prueba queda también en `docs/pid_test_summary.csv`.

## Límites de la conclusión

Se demostró aterrizaje con los rangos indicados, no robustez global. El viento es
uniforme, observable y constante; ±40 N equivale a 0.4 m/s² sobre 100 kg, menor
que el rango opcional ±80 N del entorno anterior. Los dos fallos de prueba fueron
por ω al contacto: no justifican relajar el umbral. No se corrigió el controlador
usando esos resultados. Veinticinco casos por condición no garantizan éxito en
otros estados, ráfagas o inclinaciones grandes. La cuantización, acoplamiento
entre empuje y torque, y aproximación de fuerza congelada siguen presentes.

No se ha probado teclado físico. Los controles manuales existentes y sus pruebas
se conservan. Las comprobaciones gráficas automáticas utilizan SDL dummy; las
gráficas de investigación utilizan el backend Agg. No se implementaron DQN ni
Double DQN, ni se ejecutó entrenamiento.

La suite completa pasó **63 pruebas**, conservando las 37 de Etapa 1. El checker
oficial de Gymnasium y `pip check` también pasaron. Se mantienen las dos
advertencias conocidas sobre cotas infinitas de observación.
