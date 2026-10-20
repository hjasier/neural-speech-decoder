# T15: propuesta propia con Brain-to-voice

Estado a 20 de octubre de 2026: fuentes y esquema revisados; **sin grabaciones
locales ni entrenamiento**. La petición HEAD al archivo de control devolvió
HTTP 403. Es la respuesta del servidor de Dryad; no se ha descargado un ZIP
válido ni se ha intentado cargar un modelo preentrenado.

## Datos disponibles y relación con el proyecto

[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f) publica
características intracorticales ya procesadas, frases indicadas al participante,
temporización del habla y objetivos acústicos LPCNet. La versión ampliada de
mayo de 2026 contiene un ZIP de control de 710.843.261 bytes y el principal
de 26.158.469.306 bytes. Los manifiestos y SHA-256 están en
`docs/datasets/voice_2025/manifest.json`.

El [lector oficial](https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025/blob/cff4875decd35ff0f98a944035377a25b43ba79c/lib/dataloader.py)
concatena `spikepow_trials` y `threshcross_trials` en una entrada de 512
características; carga `target_speech_feats_word` como objetivo de 20 dimensiones.
El [config oficial](https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025/blob/cff4875decd35ff0f98a944035377a25b43ba79c/configs/config.yaml)
usa contexto de 600 ms y ventanas cada 10 ms. Son esquemas leídos en el código;
aún no están comprobados contra los MAT de nuestro disco.

Tener datos, código y pesos publicados no impide una aportación propia. La
aportación que se propone aquí es **otra pregunta experimental y su evaluación**.
T12 y T15 necesitan adaptadores distintos, también en el orden de las entradas.
No se transferirán directamente los pesos T12 por coincidir algunas dimensiones.

## Primera prueba propuesta: temporización

Pregunta: ¿cuánto contexto necesita una GRU pequeña para localizar los períodos
de habla intentada y las transiciones habla/silencio?

- Entrada: exclusivamente características neuronales del participante.
- Etiquetas: intervalos de `speech_indices.mat`, después de verificar nombres,
  unidades, inclusividad de los índices y correspondencia ensayo a ensayo.
- Comparación: predictor constante, regresión logística y GRU unidireccional.
- Partición: sesiones completas distintas si hay varias; bloques completos si
  solo disponemos del control. Separar **antes** de generar ventanas. El control
  de una sesión no permite afirmar generalización entre sesiones.
- Normalización adicional con train; auditar el procesamiento ya suministrado
  antes de atribuir al sistema una latencia de funcionamiento real.
- Selección: umbral e hiperparámetros con validación; test reservado hasta el final.
- Medidas: F1, precisión/recall, PR-AUC, falsas activaciones por minuto, error
  temporal de inicio/fin y tiempo de inferencia. Declarar la tolerancia temporal
  antes de evaluar; no escogerla mirando el test.
- Controles: desplazar etiquetas por ensayo y comparar contra información temporal
  del protocolo para comprobar si el modelo aprovecha la estructura de la tarea.

No se emplearán micrófono, IMU ni audio objetivo como entradas neuronales. Una
red unidireccional por sí sola no acredita causalidad de todo el sistema: las
características entregadas ya incluyen suavizado y normalización cuyo soporte
temporal habrá que revisar.

## Segunda prueba, después de validar la temporización

Comparar ridge temporal y GRU para predecir características LPCNet alineadas,
sin copiar el transformer del repositorio. Medir error por característica,
correlación y coste. La síntesis audible requeriría además vocoder y comprobación
de su compatibilidad; no se incluye como resultado conseguido hoy. Los objetivos
publicados son objetivos de entrenamiento de los autores, no necesariamente una
grabación natural intacta del participante.

## Obtención reproducible

```bash
python scripts/get_data.py --dataset voice_2025 --files t15_control_experiment.zip
# Si Dryad responde 403, descargar en el navegador y colocar el archivo en
# data/voice_2025/t15_control_experiment.zip, sin cambiar su nombre ni contenido.
python scripts/get_data.py --dataset voice_2025 --files t15_control_experiment.zip --verify-only
```

Antes de programar el adaptador, inspeccionar los README internos y las variables
MAT del ZIP verificado. El control es un primer candidato por tamaño, pero **no
se da por hecho que tenga las mismas etiquetas/objetivos que el archivo principal**.
Si no permite construir la etiqueta, se conserva para auditoría y se prepara una
sesión del archivo principal. No se inventarán etiquetas a partir de un nombre de
archivo ni se reutilizarán las particiones de T12.

Falta: descarga válida, auditoría de variables/alineación, elección documentada
de sesiones/bloques, implementación y resultados. No presentar este documento
como una segunda réplica entrenada.
