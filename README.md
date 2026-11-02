# Decodificación de habla con señales cerebrales

Réplica en PyTorch de **Willett et al. (2023), A high-performance speech
neuroprosthesis**, con las grabaciones del paciente **T12**.

- **Paper:** [A high-performance speech neuroprosthesis](https://doi.org/10.1038/s41586-023-06377-x).
- **Implementación de referencia:** [speechBCI](https://github.com/fwillett/speechBCI).
- **Experimentos:** [W&B · tiiac-2026](https://wandb.ai/aaurre-deusto/tiiac-2026).
- **Autoría:** Asier Aurre.

## Resultado

| Métrica | Paper, habla vocalizada | Réplica causal | Escala de la réplica |
|---|---:|---:|---|
| PER, error de fonemas | 19,7 % | 31,90 % | T12, 1.460 frases de validación |
| WER, error de palabras | 23,8 % | 37,95 % | Mismas frases, con trigrama |

El sistema decodifica online, por bloques de 80 ms. La llegada de señal se
simula con grabaciones; el paper la recibe de un implante en uso.
Estas métricas se calcularon por bloques sobre validación, con otra partición
y normalización, por lo que las condiciones de evaluación son distintas.
Resultados y ejecución en los notebooks [12, modelo causal](notebooks/12_entrenamiento_online_colab.ipynb)
y [13, trigrama](notebooks/13_trigrama_colab.ipynb).

## Instalación

Python 3.12 y `curl` para descargar los datos.

```bash
git clone https://github.com/hjasier/tic_proyecto.git
cd tic_proyecto
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

```bash
python scripts/get_data.py --dataset speech_2023 --files competitionData.tar.gz
tar -xzf data/speech_2023/competitionData.tar.gz -C data/speech_2023
python scripts/get_phoneme_resources.py --config configs/entrenos_online/speech_online.yaml
python scripts/prepare_phoneme_labels.py --config configs/entrenos_online/speech_online.yaml
python scripts/prepare_speech_data.py --config configs/entrenos_online/speech_online.yaml
```

El script descarga `competitionData` de [Dryad](https://doi.org/10.5061/dryad.x69p8czpq)
y verifica su SHA-256. Son 3,67 GB comprimidos, unos 6,95 GB extraídos y
2,44 GB de señales preparadas. Licencia CC0 1.0. Se utilizan 19 sesiones:
5.180 frases para entrenar, 1.460 para validar y 680 reservadas como test.

## Ejecución

```bash
# Comprobar la instalación con datos sintéticos.
python run.py --smoke

# Entrenar el modelo causal
wandb login
python run.py --config configs/entrenos_online/speech_online.yaml

# Abrir la demo local en CPU, con los pesos y el trigrama preparados.
python scripts/demo_speech_http.py
```

El [notebook 12](notebooks/12_entrenamiento_online_colab.ipynb) prepara el entorno
de Colab, entrena y evalúa el modelo. El [13](notebooks/13_trigrama_colab.ipynb)
prepara el trigrama y su runtime. Los datos y pesos no se incluyen en Git.
Sus rutas se configuran en los perfiles de la [demo](configs/entrenos_online/speech_demo_http.yaml)
y del [trigrama](configs/entrenos_online/speech_trigrama_compacto.yaml).

Con el servidor arrancado: [visor](http://127.0.0.1:8766/) y
[llamadas](http://127.0.0.1:8766/llamada). Las llamadas requieren las dependencias
de `requirements-llamadas.txt`, FFmpeg con Flite y las claves `LIVEKIT_URL`,
`LIVEKIT_API_KEY` y `LIVEKIT_API_SECRET` en el `.env` privado.
La voz del micrófono activa una muestra aleatoria; la respuesta es independiente de lo que se diga.

## Estructura

```text
configs/       Configuración de modelos y experimentos
src/data/      Carga, particiones y preparación
src/models/    GRU y salidas de fonemas
src/training/  Entrenamiento y evaluación
src/app/       Visor y llamadas
scripts/       Descargas y utilidades
notebooks/     Exploración y ejecuciones en Colab
tests/         Pruebas
run.py         Entrada de entrenamiento y evaluación
```

## Reproducibilidad

- Semilla en `src/utils/seed.py`, configuración en `configs/` y dependencias fijadas en `requirements.txt`.
- Modelo causal: **10.001 pasos en 34 min 52 s de bucle**, con validaciones y guardados, en una **A100 de 40 GB**. Run [jaovk23p](https://wandb.ai/aaurre-deusto/tiiac-2026/runs/jaovk23p).
- Los pesos se seleccionan por PER de validación. El test queda fuera de la selección.
- La réplica incluye GRU + CTC y trigrama no incluye rescoring con OPT por el momento.

## Decisiones donde el paper no especifica

El paper y su código describen la base del sistema. Aquí se recogen las
decisiones propias de la adaptación y los cambios respecto a esa base,
incluidos los que afectan a aspectos que sí están especificados.

| **Aspecto del paper o de la adaptación** | **Qué he decidido** | **Por qué** |
|---|---|---|

| Aspecto | Qué he decidido | Por qué |
|---|---|---|
| Cantidad de datos | El paper usa 10.850 frases de entrenamiento. He usado 19 sesiones, con 6.640 frases antes de separar validación. | Elegí este recorte de datos con la idea de que cupiese en el límite de los recursos |
| Partición y selección de pesos | De las 6.640 frases, uso 5.180 para entrenar y reservo los últimos bloques de cada sesión, con 1.460 frases, para validar. Elijo los pesos con menor PER en validación y dejo aparte las 680 frases de test. | Así puedo elegir el modelo sin tocar el test, que queda para la evaluación final. Mantengo la misma partición para comparar las variantes con los mismos datos. |
| Implementación y arquitectura base | He pasado la red de TensorFlow a PyTorch. Uso cinco capas GRU de 512 unidades, ventanas de 14 bins y avance de 4, como el modelo online del paper. Su baseline offline usa 1.024 unidades y ventanas de 32, en el mio sigo usando la misma configuración que en el online | También usé 512 unidades en la versión offline para que consumiera menos memoria y fuera más rápida en mi ordenador, aunque pudiera perder algo de precisión. |
| Salidas intermedias | He añadido salidas en las capas 2 y 3, además de la final, con pesos CTC de 0,25/0,25/0,50. También he preparado versiones independientes de dos y tres capas para comparar. | Quería comprobar si podía obtener respuestas más rápidas usando menos capas y cuánta precisión perdería. Son pruebas experimentales, la demo sigue usando el modelo de cinco capas. |
| Normalización de la señal | Calculo la media y la desviación con los datos de entrenamiento de cada sesión y las guardo con el modelo. | Preferí empezar con una normalización fija para simplificar el sistema y tener primero la demo funcionando. Dejé para más adelante la adaptación a los cambios de la señal durante el día que utiliza el paper. |
| Memoria durante el entrenamiento | Proceso cada lote de 64 frases en cuatro grupos de 16, acumulando los gradientes antes de actualizar el modelo. | Así puedo entrenar con menos memoria de GPU sin tener que reducir la red ni el tamaño del lote. |
| Modelo de lenguaje y OPT | Uso la GRU con el trigrama original de 125.000 palabras. Dejo para más adelante el 5-grama y OPT, que el paper utiliza para mejorar el resultado offline. | Quería tener primero la demo funcionando en la CPU de mi ordenador. OPT tiene 6.700 millones de parámetros y añadirlo exige bastante memoria. Además, el resultado online del 23,8 % con el que comparo tampoco utiliza OPT. |
| RAM del trigrama | Lo he convertido a ConstFst y he cambiado su lectura para cargar en RAM solo las partes que se van usando, sin recortar el modelo. | No podía usar el trigrama original en mi ordenador sin quedarme sin memoria, con mi implementación el archivo puede ocupar más en disco, pero necesita menos RAM para funcionar. |




Las dos arquitecturas de referencia se pueden consultar en la
[configuración online de los autores](https://github.com/fwillett/speechBCI/blob/522a9605ccf0818def80b6fd80007598e790408e/NeuralDecoder/neuralDecoder/configs/model/gru_stack_inputNet.yaml)
y en su [notebook del baseline offline](https://github.com/fwillett/speechBCI/blob/522a9605ccf0818def80b6fd80007598e790408e/AnalysisExamples/rnn_step2_trainBaselineRNN.ipynb).
La normalización y OPT se describen en los [métodos suplementarios](docs/41586_2023_6377_MOESM1_ESM.pdf).
Los cambios de la réplica están en los [perfiles causales](configs/entrenos_online/)
y en el [conversor del trigrama](scripts/compact_speech_trigram.py).

## Tests

```bash
pytest tests/ -q
```

## Uso de IA

Documentado en [AI_USAGE.md](AI_USAGE.md).
