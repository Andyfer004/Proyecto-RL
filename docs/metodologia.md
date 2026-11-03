# Protocolo de experimentación (PID implementado; comparación RL pendiente)

La ejecución de Etapa 2, sus splits materializados, ajuste y resultados reales se
documentan en `pid.md`. Lo que sigue describe el protocolo previsto para comparar
PID con DQN y Double DQN; esa comparación todavía no está implementada.

Separar entrenamiento, validación para parámetros PID/hiperparámetros/recompensa,
y prueba reservada. Guardar manifiestos de estados iniciales, semilla y viento;
no usar prueba para escoger parámetros. En este modelo el viento es constante por
episodio: el mismo valor corresponde a la misma secuencia para todos los métodos.
Si se añade viento variable, predefinir secuencias o procesos observables comunes.

Condiciones previstas: descenso casi vertical sin viento; desplazamiento e
inclinación moderados; viento horizontal configurable. La primera etapa incluye
configuración fácil y su variante con viento, sin afirmar validación de control.
Si se incorpora currículo, deberá usar las mismas etapas y presupuestos en los
dos algoritmos RL; la versión actual no contiene currículo.

Valores iniciales configurables de las etapas futuras: tres semillas de entrenamiento
por algoritmo, cien estados predefinidos de prueba por condición, mismo número de
interacciones para DQN y Double DQN. Evaluar sin exploración ni actualizaciones y
con el mismo catálogo y período de decisión para PID y RL. Separar variación de
semillas de entrenamiento de variación de escenarios; no tratar los episodios de
una misma red como entrenamientos independientes.

Registrar: éxito; gas consumido; error horizontal, velocidades, ángulo y velocidad
angular al contacto; duración; tipo de fallo; retorno. En episodios sin contacto,
las métricas de contacto son ausentes, nunca cero. Presentar tasa de éxito y métricas
sobre todos los episodios (para contacto, informar n de contactos disponibles),
así como gas/error/duración condicionados a éxito. Informar n de episodios,
n de éxitos, n de semillas, variación por semilla e incertidumbre. No excluir
fallos silenciosamente al calcular consumo medio. Usar resúmenes por semilla
antes de agregar, y no interpretar una recompensa alta como éxito.

El CLI de Etapa 1 exporta CSV y un JSON de configuración, versiones y argumentos.
CSV almacena estado inicial, final, viento y semilla. Son trazas de comprobación,
no una comparación de controladores. Para publicaciones futuras guardar también
commit de Git, manifiesto del split, configuración del agente y presupuesto total.

## Diseño del PID (implementación y validación en pid.md)

Control horizontal en cascada: error x → vx deseada → aceleración horizontal
→ θ deseado (negativo para aceleración derecha). Orientación θ/ω → momento.
Descenso: altura disponible y velocidad de frenado → vy deseada → fuerza vertical.
Limitar inclinación, velocidades, fuerza y momento; integrales con anti-windup
por saturación y reinicio por episodio. Comparar la fuerza y momento solicitados
con los de cada acción mediante error cuadrático normalizado por autoridad máxima;
usar el mismo catálogo cada 0.05 s, sin alterar el estado directamente.

Ajustar solo en un conjunto fijo de validación sencilla. Antes de entrenar RL,
exigir aterrizajes observados en ese conjunto; si no ocurren revisar física,
cuantización, tiempos y ganancias. No confundir autoridad de frenado analítica
con éxito demostrado del PID.

## Implementación RL (Etapa 3 documentada en dqn.md)

MLP pequeña CPU, replay, red objetivo, epsilon-greedy, minibatches, Huber y clipping
de gradiente. DQN: max_a Q_objetivo(s',a). Double DQN: elegir argmax con Q_principal
y evaluar esa acción con Q_objetivo. Targets sin gradientes y máscara basada en
terminated, preservando bootstrap en truncated. Guardar configuración, pesos,
optimizador, generadores aleatorios y contadores para reanudación reproducible.
Pruebas futuras deben distinguir numéricamente ambos targets y validar checkpoints.
