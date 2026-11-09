# Contenido versionado de la entrega

El repositorio incluye el código y la evidencia necesarios para inspeccionar y reproducir el protocolo final sin entrenar otra vez. No se modificaron física, modelos ni resultados al preparar la publicación.

## Incluido en Git

- Código de simulación, PID, DQN/Double DQN, variantes asistidas, diagnósticos y pruebas.
- Configuraciones, manifiestos de escenarios y protocolo final congelado.
- Informe final PDF con carátula UVG, LaTeX editable y figuras; informe Markdown, demostración HTML y GIF locales.
- Modelos seleccionados de inferencia: PID configurado, DQN y Double desde cero, DQN y Double asistidos y sensibilidad DQN43.
- Los2328 registros finales, sus trazas completas, resúmenes, comparaciones emparejadas, causas y verificaciones de invariancia.
- Metadatos y CSV seleccionados del desarrollo asistido, figuras y auditoría; estadísticas docentes, sin subir los buffers binarios regenerables.
- Logs de verificación anteriores copiados como texto en `entrega/verificaciones/`. No son pruebas nuevas ni entrenamientos ejecutados para publicar.

`publicacion_manifest.json` enumera cada artefacto publicado con SHA-256. La selección se agrega explícitamente a Git porque los directorios generales de resultados y los pesos están ignorados por defecto; las nuevas ejecuciones no se versionan automáticamente.

## Excluido

Entornos virtuales, cachés, borradores de maquetación, replays, checkpoints completos de reanudación, datos docentes binarios regenerables y campañas históricas brutas no seleccionadas. Los manifiestos históricos pueden referirse a archivos locales no publicados; no deben confundirse con la lista de contenido del clon.

## Comprobar un clon limpio

Desde la raíz del clon:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,render,rl,analysis]'
python scripts/verify_repository.py
```

La verificación comprueba presencia y hashes, los2328 resultados, contadores CE/TD y cifras emparejadas. No simula, no entrena y no vuelve a ejecutar la prueba. Para las versiones exactas del entorno verificado, usar `configs/requirements-stage3.txt` como explica README.

Sin Python se pueden abrir `output/pdf/informe_final.pdf` y `docs/demostracion.html`. Esta última usa las animaciones y figuras locales incluidas en el clon.

## Regeneración documental

```sh
# Solo tablas y figuras desde CSV existentes.
python scripts/report_scope.py

# Solo GIF desde estados guardados, sin nuevas decisiones físicas.
python scripts/animate_scope.py

# Opcional: PDF a partir del informe; requiere Node y Tectonic.
mkdir -p tmp/pdfs output/pdf
python scripts/prepare_pdf_figures.py
node scripts/build_report_pdf.mjs
tectonic --keep-logs --outdir output/pdf tmp/pdfs/informe_final.tex

# ZIP local; no es necesario subirlo a Git.
python scripts/build_delivery.py
```

Las regeneraciones pueden variar metadatos internos de PDF/PNG y sus hashes binarios según versiones; no modifican la evidencia experimental original. No sustituir silenciosamente un artefacto congelado. El protocolo final ya se consumió: nuevas modificaciones de entrenamiento necesitarán experimento y evaluación independientes.

Los checkpoints asistidos son de inferencia, no de reanudación fiel. Los scripts históricos de entrenamiento se conservan para estudio/reproducción en directorios nuevos; no forman parte de la comprobación del clon.
