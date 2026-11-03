# Diseño físico y formulación del entorno — Etapa 1

## Convenciones y aproximaciones

Modelo académico propio, sin Box2D ni código de LunarLander. No reproduce ningún
vehículo comercial ni tecnología de Tesla. Un solo cuerpo rígido rectangular en
un plano; coordenadas globales x hacia la derecha, y hacia arriba. Unidades SI:
metros, segundos, kilogramos, newtons, radianes. Ángulo cero vertical, positivo
antihorario. El origen global está en el centro de la plataforma y a nivel del
suelo. El origen local está en el centro de masa.

Masa fija 100 kg; ancho 1.4 m; altura 1.6 m; inercia de rectángulo uniforme
I=m(w²+h²)/12 = 37.6666667 kg m². Los apoyos sin masa están en
(-0.6,-1.0) y (+0.6,-1.0) m locales. Los cuatro vértices del cuerpo están en
(±0.7,±0.8) m. El centro objetivo en reposo es (0,1) m.
La masa y la inercia no disminuyen al gastar gas: se modela un presupuesto
abstracto de impulso, no termodinámica, presión, caudal másico ni un recipiente.

## Propulsores y autoridad de control

| Propulsor | Punto local (m) | Fuerza local (N) | Momento (N m) |
|---|---|---|---|
| Central | (0,-0.8) | (0,1800) | 0 |
| Izquierdo | (-0.6,-0.8) | (0,600) | -360 |
| Derecho | (+0.6,-0.8) | (0,600) | +360 |

Todos son binarios. No hay cambios directos de posición, velocidad ni ángulo.
El movimiento horizontal se consigue inclinando el vehículo: para θ>0 el empuje
produce una componente hacia la izquierda. El diferencial lateral controla θ.
Las posiciones se definen con el ancho del brazo lateral y la mitad de la altura
configurados, y los momentos siempre se calculan mediante r_x F_y-r_y F_x.

| Acción | Activaciones | Empuje local vertical (N) | Momento (N m) |
|---|---|---|---|
| 0 | Apagado | 0 | 0 |
| 1 | Central | 1800 | 0 |
| 2 | Izquierdo | 600 | -360 |
| 3 | Derecho | 600 | +360 |
| 4 | Ambos laterales | 1200 | 0 |
| 5 | Central + izquierdo | 2400 | -360 |
| 6 | Central + derecho | 2400 | +360 |
| 7 | Todos | 3000 | 0 |

Peso 981 N; empuje máximo/peso=3.058. Con vehículo vertical, la aceleración neta
máxima ascendente es 20.19 m/s² (central solo: 8.19 m/s²). A velocidad descendente
de 1 m/s bastan 0.025 m para detenerse con empuje máximo, sin demora ni inclinación.
Incluyendo una demora de decisión de 0.05 s, la velocidad sería 1.4905 m/s y la
distancia posterior de frenado 0.055 m. La altura mínima inicial de los apoyos es
aproximadamente 8.95 m para |θ|≤0.08 y y≥10; existe margen amplio en el rango
inicial sencillo. Esto verifica autoridad de frenado, no que cualquier política
pueda aterrizar desde cualquier estado.

## Dinámica e integración

R(θ)=[[cosθ,-sinθ],[sinθ,cosθ]]. F_global=R(θ) Σ F_local.
Aceleraciones: a=(F_global+(W,-mg))/m; α=Σ(r×F)/I.
Viento W es una fuerza horizontal aplicada al centro de masa, sin momento.
Se sortea W~U(wind_mean-wind_amplitude,wind_mean+wind_amplitude) al reset y
permanece constante durante todo el episodio. También se puede fijar con
`options={"wind_force": 40.0}`. Es observable: no hay estado oculto del viento.
No se incluyen arrastre, sustentación ni ruido adicional.

Decisiones cada 0.05 s (20 Hz), diez subpasos de 0.005 s. El empuje se transforma
usando el ángulo al comienzo de cada subpaso. Se integran aceleraciones congeladas:
p'=p+v·h+a·h²/2; v'=v+a·h, y análogamente θ y ω. El ángulo se lleva a [-π,π).
La caída libre es exacta salvo redondeo; la traslación bajo rotación tiene error
discreto que se reduce con subpasos. No se supone conservación de energía durante
una activación: los propulsores realizan trabajo.

Gas G inicial=18000 N s. Consumo: ΔG=Σ||F_i||·Δt, sumando empujes, sin
cancelar fuerzas ni momentos. Si falta gas se integra empuje únicamente durante
G/Σ||F_i|| segundos y luego caída libre durante el resto. El impulso total no
supera G. Todos los propulsores activos se apagan simultáneamente al agotarse.
El gasto durante el contacto se corta en el instante estimado del evento.
La capacidad equivale a unos 18.35 s de sustentación ideal a peso fijo.

## Contacto y finalización

El suelo completo está a y=0 y la plataforma ocupa x∈[-2.5,2.5]. Se transforman
los seis puntos geométricos del vehículo; el primer punto que alcanza y≤0 produce
contacto. No se espera a que el centro cruce el suelo. Se detecta el cruce en cada
subpaso y se localiza por bisección (35 iteraciones). No se modela rebote ni fricción:
el primer contacto termina el episodio.

Éxito exige cuerpo por encima del suelo; ambos apoyos dentro de la plataforma;
|x_centro|≤1.5 m; |vx|≤0.7 m/s; -1≤vy≤0 m/s; |θ|≤0.12 rad;
|ω|≤0.25 rad/s; ambos apoyos a no más de 0.15 m sobre el suelo en ese instante.
Este último umbral admite el pequeño desfase del primer apoyo cuando hay inclinación;
no simula una fase posterior de asentamiento. Todos los umbrales son configurables.

Terminaciones: `success`, `body_contact`, `missed_platform`, `invalid_contact`,
`out_of_bounds`. Cualquier punto fuera de |x|<12 m o sobre y<25 m causa salida.
`time_limit` es truncamiento a 25 s, separado de terminación física. Un contacto
al mismo tiempo que el límite temporal tiene prioridad como terminación.
Gas cero permite seguir cayendo o desplazándose hasta una finalización real.

El detector usa extremos de subpasos y bisección; no es detección continua exacta
para rotaciones extremas capaces de cruzar y regresar dentro de un solo subpaso.
Los parámetros académicos y subpasos pequeños reducen ese riesgo. Cambiar a
velocidades o fuerzas extremas exige un estudio de convergencia y barrido continuo.

## Observación, reset e información

Estado físico: (x,y,vx,vy,θ,ω,G), más W constante. Observación float32 de 9 valores:

| Índice | Valor | Escala |
|---|---|---|
| 0 | x relativo a plataforma | 12 m |
| 1 | y relativo a altura del centro al aterrizar: y-1 | 25 m |
| 2,3 | vx,vy | 10 m/s |
| 4,5 | sinθ,cosθ | 1 |
| 6 | ω | 3 rad/s |
| 7 | G/G_capacidad | 18000 N s |
| 8 | W | 100 N |

Las escalas normalizan sin recortar; posición, velocidad y viento pueden exceder
±1. El espacio Box declara cotas infinitas para esos elementos, ±1 para seno y
coseno, [0,1] para gas. `info` conserva los valores SI, tiempo, viento, consumo,
resultado, métricas al contacto y componentes de recompensa. No se devuelven
buffers mutables compartidos entre pasos.

`reset(seed=...)` usa solamente `self.np_random` para estado y viento. Rangos:
x∈[-1,1], y∈[10,12], vx∈[-0.2,0.2], vy∈[-1,0], θ∈[-0.08,0.08],
ω∈[-0.02,0.02]. `options.initial_state` admite sobreescrituras parciales en SI.
Estados iniciales en contacto o fuera del área se rechazan. La semilla del catálogo
aleatorio se fija separadamente con `action_space.seed(seed)`.

El reloj se observa en info y queda fuera de la observación de RL: el horizonte
es un límite artificial del recolector, no una variable oculta de la dinámica.
La transición física y recompensa dependen solo del estado observado y la acción;
el bootstrap futuro debe usar `terminated`, sin anularlo por `truncated`.
Si se trata el plazo como parte del objetivo de un MDP finito, habrá que añadir
el tiempo restante a la observación y revisar el bootstrap.

## Recompensa diagnóstica

Potencial Φ(s)=-(w_x |x|/12+w_h max(y-1,0)/25).
Progreso = w_progreso(Φ(s')-Φ(s)); no se multiplica por un γ implícito.
El resto suma terminal (+100 éxito, -100 fallo),
-w_vel h·||v||/10, -h(w_θ |θ|/π+w_ω |ω|/3),
-w_gas ΔG/G_capacidad y -w_tiempo h. Pesos exactos en `configs/env.json`.
El límite temporal no recibe premio terminal ni se presenta como impacto.

Solo se penaliza gas consumido. La penalización temporal hace costoso mantenerse
suspendido y el progreso acumulado telescopa, sin premio recurrente por flotar;
el horizonte acota el retorno. Esto no garantiza una política óptima de aterrizaje:
la calibración debe comprobarse en validación antes del experimento final.
Éxito se decide con geometría y umbrales, independientemente de la recompensa.
