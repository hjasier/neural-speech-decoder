<!--
PLANTILLA. Sustituye TODO lo que va entre <angulos>. Borra este comentario.

Un README se lee en treinta segundos o no se lee. Quien lo abre quiere saber
tres cosas: qué es esto, cómo lo ejecuto, y funcionó o no. En ese orden.

Recuerda que en noviembre otra pareja va a clonar este repositorio y va a
intentar reproducir una métrica SIN PREGUNTAROS NADA. Este fichero es lo único
que van a tener.
-->

# <Título del proyecto>

Replicación de **<Autores, año · Título del paper>** sobre **<dataset>**,
con <aplicativo: buscador semántico / detector en vídeo / lo que sea> construido encima.

- **Paper:** <enlace a arXiv o DOI>
- **Implementación de referencia:** <enlace, o «ninguna: el paper se implementa desde cero»>
- **Seguimiento de experimentos:** <enlace al proyecto de W&B — obligatorio>
- **Autoría:** <Nombre 1>, <Nombre 2>

## Resultado

| Métrica | Paper | Nuestra réplica | Escala |
|---|---|---|---|
| PER greedy | Pendiente de comparación | Pendiente de entrenamiento completo | 19 sesiones T12 |

<Una frase honesta sobre la diferencia. Una desviación bien explicada puntúa
igual que un acierto; una desviación sin explicar, no.>

## Instalación

```bash
git clone <url>
cd <repo>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

```bash
python scripts/get_data.py --dataset speech_2023 --files competitionData.tar.gz
```

Descarga `competitionData` de Dryad y verifica su SHA-256 oficial: 3,67 GB
comprimidos y unos 6,95 GB extraídos, con licencia CC0. Los `.mat` deben
quedar en `data/speech_2023/competitionData/{train,test,competitionHoldOut}/`.
Para extraer el archivo verificado:

```bash
python - <<'PY'
import tarfile
with tarfile.open("data/speech_2023/competitionData.tar.gz") as archive:
    archive.extractall("data/speech_2023", filter="data")
PY
```

La configuración `configs/speech_original.yaml` selecciona las 19 sesiones
del baseline TensorFlow publicado: **6.640 ensayos de train y 680 de test**.
Desde la raíz del repositorio, con las dependencias instaladas:

```bash
python scripts/get_phoneme_resources.py --config configs/speech_original.yaml
python scripts/prepare_phoneme_labels.py --config configs/speech_original.yaml
python scripts/prepare_speech_data.py --config configs/speech_original.yaml
```

La preparación procesa una sesión cada vez. Guarda señales en el orden
`tx1 → spikePow`, normalizadas con estadísticas de train del mismo día,
en 38 archivos NPY que ocupan **2,44 GB adicionales**. No aplica suavizado
ni ruido: se harán al formar la entrada del modelo. Los manifiestos conservan
estadísticas, límites de los ensayos, referencias y hashes. Todo queda en
`data/`, ignorado por Git; no se modifican los `.mat` originales.

`construir_loaders(cfg)` ya devuelve lotes reales para esta configuración.
Entrenamiento elige una sesión uniformemente por lote y recorre sus ensayos
barajados, con un RNG independiente del modelo. Cada lote conserva señales
con padding cero, fonemas concatenados sin padding, longitudes reales e IDs
de sesión. **El ID 0 es un fonema válido y el blank CTC es 40.** El segundo
loader recorre el test publicado, sin barajar ni descartar el último lote;
no constituye una validación independiente.

## Ejecución

```bash
# Baseline de habla: datos reales, cinco GRU de 1.024 unidades, CTC y PER.
python run.py --config configs/speech_original.yaml

# Prueba acortada con presupuesto de dos horas; no es la réplica completa.
python run.py --config configs/speech_2h.yaml

# Evaluar pesos guardados, sin entrenar; sustituir ID_RUN por el ID impreso.
python run.py --config configs/speech_original.yaml --solo-evaluar \
    --checkpoint checkpoints/speech_original/ID_RUN/ultimo.pt
```

Para entrenar en una máquina de Colab, abre
[`notebooks/04_entrenamiento_colab.ipynb`](notebooks/04_entrenamiento_colab.ipynb).
Sube ese archivo a Colab, selecciona GPU y ejecuta sus celdas en orden. Contiene
un snapshot de las fuentes verificado por SHA-256, de modo que funciona sin
dar acceso de Colab al repositorio. No incluye datos, pesos ni credenciales.
Instala un entorno aislado con Python 3.12.11 y las dependencias fijadas,
descarga y verifica el corpus público y llama al mismo `run.py`. Conserva
arquitectura, sesiones y lote efectivo; fuerza CUDA para evitar entrenar por
accidente en CPU. W&B usa la clave de Secretos de Colab o una entrada oculta.

La celda 4 elige el perfil. Por defecto usa `configs/speech_2h.yaml`: misma
arquitectura, normalización, 19 sesiones y lote efectivo de 64, con
microlotes de 16, hasta 2.001 pasos, learning rate de 0,02 a 0 en 2.000 pasos
y evaluación completa cada 250 pasos, además de la primera y la última.
El autor observó 250 pasos en 11 minutos con microlotes de 4: al mismo ritmo,
2.001 pasos serían unos 88 minutos. La velocidad y memoria del perfil de
16 todavía no se han medido; no se promete una mejora concreta.

`entrenamiento.max_segundos: 5400` añade un límite suave de 90 minutos al
bucle, incluidas sus evaluaciones periódicas y guardados. Se comprueba entre
actualizaciones y se termina con evaluación y checkpoint del paso alcanzado;
si una evaluación periódica ya agotó el presupuesto, se reutiliza ese
resultado. Preparación, lote en curso, evaluación final, guardado y
sincronización pueden añadir tiempo. W&B registra los pasos previstos,
completados y si terminó por tiempo. El perfil completo mantiene sus
10.001 pasos, microlotes de 4, evaluación cada 50 y ningún límite temporal.
El perfil de dos horas no equivale al entrenamiento completo ni cumple la
regla docente de menos de 20 minutos.

Cambiar un YAML no modifica un run ya iniciado. Para usar el perfil nuevo
en una sesión de Colab ya preparada, detén la celda 9 y ejecuta las celdas
2 (fuentes actualizadas), 4 (configuración y experimento nuevo) y 9; no
repitas instalación ni preparación de señales. El entrenamiento empieza
desde cero: el checkpoint anterior se conserva para evaluación, pero aún
no se reanuda el optimizador.

El notebook solicita acceso a Drive para guardar checkpoints, fuentes,
configuración, entorno y log; las señales se preparan en el disco temporal
de Colab. La celda 6 usa por defecto el archivo original desde Drive: busca
en las carpetas habituales y después por nombre en Mi unidad, con límites
de profundidad, carpetas y tiempo. También acepta la ruta completa obtenida
con «Copiar ruta» en el panel Archivos de Colab o una ruta relativa a Mi unidad.
Si no lo encuentra, muestra cómo indicar su ubicación; solo intenta descargar
de Dryad cuando se selecciona expresamente ese origen. Comprueba tamaño y
SHA-256 antes de publicar la copia local y reutiliza una copia local válida.
La salida stdout/stderr de los scripts se muestra en la celda y los errores
incluyen el motivo, además del código de salida. Colab puede interrumpir la
sesión: los checkpoints
actuales permiten evaluar e inferir, pero todavía no reanudar el optimizador.
La duración completa no está medida. No se ha ejecutado este entrenamiento
en Colab ni se ha vuelto a entrenar en el PC para crear el notebook.

Si cambias las fuentes o la configuración, actualiza la copia del notebook
antes de subirlo de nuevo:

```bash
python scripts/export_colab.py
```

`configs/speech_original.yaml` reproduce los parámetros del [comando offline de los
autores](https://github.com/fwillett/speechBCI/blob/522a9605ccf0818def80b6fd80007598e790408e/AnalysisExamples/rnn_step2_trainBaselineRNN.ipynb)
y su [implementación TensorFlow](https://github.com/fwillett/speechBCI/blob/522a9605ccf0818def80b6fd80007598e790408e/NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py):
ruido blanco (SD 1) y offset por frase (SD 0,2) antes del suavizado, Adam con
epsilon-hat 0,1, tasa lineal de 0,02 a 0 en 10.000 pasos, clipping de norma
global a 10 y L2 de 1e-5 en kernels de entrada GRU y de la capa diaria activa.
El bucle original incluye los índices 0–10.000: el YAML pide 10.001
actualizaciones y la última tiene tasa cero. Los parámetros están en el YAML.
La pérdida CTC se promedia por frase,
sin dividir cada pérdida por su longitud de referencia.

Para ajustarse a la memoria de la RTX 2060 se conserva el **lote efectivo de
64 frases**, procesándolo en grupos de 4 (`entrenamiento.micro_batch_size`).
Se acumula la suma CTC de cada grupo dividida por las 64 frases; L2,
clipping, Adam y el avance del learning rate se aplican una sola vez por lote
completo. No se recortan frases ni se reduce la arquitectura. Train conserva
la extensión temporal del padding del lote original para añadir ruido antes
del suavizado. El orden de los números aleatorios y el redondeo pueden cambiar
respecto a procesar las 64 frases simultáneamente.

Evaluación también usa grupos de 4 (`evaluacion.micro_batch_size`), recortando
solo padding temporal sobrante y manteniendo todos los ejemplos y fonemas.
W&B registra los picos de memoria CUDA asignada y reservada en MiB; también
se imprimen con el avance del entrenamiento. El proceso usa dos hilos de CPU
para limitar su carga en el equipo de escritorio.

Cada evaluación registra CTC y **PER greedy = errores de edición totales /
fonemas de referencia totales**, incluyendo SIL, con blank 40 y sin contar
el padding. PER se guarda como proporción (0,25 equivale al 25 %) y puede
superar 1 si hay muchas inserciones. Se registran también los recuentos y si
se recorrió el test completo. No se calcula WER: falta el decodificador de
palabras/modelo de lenguaje.

El test publicado se monitoriza tras la primera actualización, cada 50 pasos
después y al terminar, como en la referencia. Se guarda
`ultimo.pt` por ID de ejecución, sin seleccionar el mejor por PER de test.
Es una diferencia explícita respecto al original, que selecciona el mejor
checkpoint sobre ese conjunto. Tampoco hay aún una validación independiente.
El checkpoint contiene pesos, configuración, sesiones, vocabulario,
estadísticas de normalización de train, firma de los datos y métricas;
`ultimo.json` conserva los recuentos y las métricas. `--solo-evaluar` exige
checkpoint y comprueba su compatibilidad. Estos checkpoints permiten evaluar
e inferir; todavía no permiten reanudar el optimizador.

W&B está configurado en **online**: sincroniza configuración, pérdidas, tasa,
norma de gradiente, tiempos y métricas con el proyecto `tiiac-2026` de la
cuenta autenticada. Antes del primer entrenamiento:

```bash
wandb login
```

Para trabajar sin conexión se puede usar `wandb.modo: offline` en el YAML;
el historial queda bajo `runs/tracking/wandb/` y se puede subir después:

```bash
wandb sync runs/tracking/wandb/offline-run-FECHA-ID
```

Se puede establecer la entidad en el YAML para usar un equipo concreto.
La ejecución de habla rechaza el modo disabled y se detiene si falta W&B.
`ejecucion.dispositivo: auto` elige CUDA cuando está disponible; también admite
`cpu` o `cuda`. La versión completa no tiene un tiempo de entrenamiento medido
ni cumple aún el límite docente de 20 minutos; la configuración reducida está
pendiente. `python run.py --smoke` y `configs/default.yaml` conservan la plantilla
sintética anterior y **no verifican el pipeline de habla**. El aplicativo sigue
pendiente de conexión con el checkpoint de habla.

La primera comprobación de integración del 19 de octubre usó dos
actualizaciones con lotes de dos frases y la arquitectura completa, en CPU
y RTX 2060. No comprobaba la memoria necesaria para lotes de 64: el primer
entrenamiento completo falló al llegar a un lote con 904 bins. Con acumulación
en grupos de 4, el run de comprobación llegó al menos al paso 51, guardó un
checkpoint y recorrió las 680 frases de test. PyTorch registró picos de
1.342 MiB asignados y 2.098 MiB reservados en la RTX 2060. Otra comprobación
con cinco frases reales, sin ruido ni dropout, verificó CTC y todos los
gradientes al dividir el lote en 2+2+1 (diferencia máxima absoluta de gradiente
2,87e-5), incluyendo L2 una sola vez.

La comprobación prevista de 64 pasos quedó interrumpida por el bloqueo y
reinicio del PC; no se completó. Tras el reinicio se comprobó el CRC del
checkpoint guardado y se revisaron estos registros sin volver a entrenar.
El registro del sistema no permite atribuir una causa concluyente al bloqueo.
En el equipo local, las comprobaciones de cómputo se ejecutarán de una en una.
El entrenamiento completo y su comparación de calidad siguen pendientes.
No se añadieron archivos de pruebas; el lanzador de Colab se añadió después
por petición expresa del autor.

## Estructura

```
configs/      Todos los hiperparámetros. Ninguno escrito a fuego en el código
src/data/     Carga, splits y transformaciones
src/models/   La arquitectura replicada
src/training/ Bucle de entrenamiento y evaluación
src/app/      La capa de servicio: carga un checkpoint y responde
scripts/      Descarga de datos y utilidades
tests/        Tests de forma, invariancia y reproducibilidad
run.py        Punto de entrada único
```

## Reproducibilidad

- Semilla fijada en `src/utils/seed.py`, valor en `configs/`.
- Dependencias con versión exacta en `requirements.txt`.
- Presupuesto de cómputo declarado: **<tiempo real> en <hardware>**.
- Qué se recortó respecto al paper y qué se pierde con ello: <explicación>.

La [prueba de tiempo con GRU y frases reales](docs/prueba_tiempo_entrenamiento.md) contiene el código, las medidas en RTX 2060 y la estimación del recorte que queda por comprobar.

La [presentación del pitch](docs/presentacion_pitch/index.html) resume el problema, el plan y el aplicativo en tres diapositivas blancas con animaciones. [Cómo abrirla y ensayar](docs/presentacion_pitch/README.md).

## Decisiones donde el paper no especifica

<La sección que distingue replicar de copiar. Lista lo que tuvisteis que
decidir vosotros y con qué criterio. Sale de la ficha de replicación.>

| Ambigüedad del paper | Qué decidimos | Por qué |
|---|---|---|
| <no dice el schedule de lr> | <cosine con warmup de 5 épocas> | <es lo habitual en la familia de trabajos citada> |

## Tests

```bash
pytest tests/ -q
```

## Uso de IA

Documentado en [`AI_USAGE.md`](AI_USAGE.md), según la guía docente.
