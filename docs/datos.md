# Datos para el proyecto de neuroprótesis del habla

Inventario consultado el **30 de septiembre de 2026**. Proyecto elegido:
*A high-performance speech neuroprosthesis* (Willett et al., 2023).
El alcance de investigación contempla los datos completos y otros participantes;
la entrega docente será una configuración reproducible de ese trabajo.

## Qué datos están disponibles

Este catálogo reúne cuatro publicaciones públicas directamente relevantes para
decodificación intracortical de habla. No es un inventario exhaustivo de todos
los datasets de lenguaje, EEG, MEG y ECoG existentes. Los manifiestos de
`docs/datasets/` contienen **todos los archivos de la publicación vigente**,
bytes exactos, enlaces oficiales y SHA-256, obtenidos mediante la API de Dryad.
Las cuatro publicaciones declaran licencia CC0 1.0 en sus metadatos.

| Publicación | Clave para descargar | Tamaño total comprimido | Aportación al proyecto |
|---|---|---:|---|
| [Willett 2023, T12](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) | `speech_2023` | 80,441 GB | Réplica principal y análisis de las grabaciones originales publicadas |
| [Card 2024, T15; datos ampliados en 2025](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) | `speech_2024_t15` | 11,595 GB | Segundo participante; entrenamiento y evaluación por sesiones |
| [Kunz 2025, habla interna](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j) | `inner_speech_2025` | 10,799 GB | Habla intentada, imaginada y otras conductas; incluye cuatro participantes |
| [Wairagkar 2025, síntesis de voz; ampliación de mayo de 2026](https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f) | `voice_2025` | 26,869 GB | Señal neural, objetivos acústicos alineados y modelos de síntesis |

Total: **129,704 GB**, antes de extraer y sin añadir cachés, conversiones,
entornos ni checkpoints. GB significa 1.000 millones de bytes. Los GiB de
algunas herramientas del sistema usan otra unidad.

## Publicación principal: todos sus archivos

Usar la entrega de **1 de septiembre de 2023**, denominada versión 4 en el
README científico. El identificador de versión de la API es `253293`;
la numeración interna de la API no coincide con el nombre de la entrega.

| Archivo | GB comprimidos | Contenido / utilidad |
|---|---:|---|
| `competitionData.tar.gz` | 3,670 | Frases preparadas para aprendizaje automático y particiones |
| `sentences.tar.gz` | 14,803 | Grabaciones de frases con más contexto experimental |
| `diagnosticBlocks.tar.gz` | 0,612 | Repeticiones de siete palabras para estudiar cambios entre días |
| `tuningTasks.tar.gz` | 0,941 | Fonemas, palabras y movimientos orofaciales aislados |
| `derived.tar.gz` | 8,062 | Intermedios para TensorFlow y modelo de referencia |
| `languageModel.tar.gz` | 14,106 | Recursos del modelo de lenguaje de tres gramos |
| `languageModel_5gram.tar.gz` | 38,246 | Recursos del modelo de cinco gramos |

También hay siete archivos README, catalogados en el manifiesto.
Fuente: [publicación de Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq).

**Los 3,67 GB no son un piloto de 200 frases.** Constituyen el paquete preparado
para desarrollar decodificadores sobre las particiones publicadas. Según su
[README oficial](https://datadryad.org/downloads/file_stream/2547362), provienen
de `sentences`, pero solo incluyen el periodo de habla de cada ensayo y excluyen
las tareas de vocabulario de 50 palabras y algunos bloques repetidos. `sentences`
incluye también periodos anteriores al habla y audio del micrófono. Su
[README](https://datadryad.org/downloads/file_stream/2547361) indica que se
retiraron los primeros dos bloques para la competición: no esperar recuperar
allí las transcripciones ocultas. Por ello,
hay solapamiento: **no concatenar ambos como ejemplos independientes**.

El README describe `sentenceText`, `spikePow`, `tx1` a `tx4` y `blockIdx`.
Las características neurales están agrupadas en intervalos de 20 ms y tienen
256 canales por familia de características. Recomienda comenzar con `spikePow`
y `tx1` de los primeros 128 canales, correspondientes al área 6v. Combinar esas
dos familias produce 256 características de entrada; canal y característica
no son sinónimos. Son características ya extraídas, no voltajes sin procesar.

Las etiquetas describen el texto que la participante intentaba decir.
El audio residual no debe suponerse una grabación inteligible de referencia.
Las particiones `train` y `test` se organizan por bloques de cada día, mientras
que `competitionHoldOut` tiene transcripciones ocultas. Esta separación prueba
otros bloques de días conocidos; no demuestra generalización a días nunca vistos.
Fuente: [protocolo oficial](https://github.com/fwillett/speechBCI#traintestcompetitionholdout-partitions).

## Otros participantes y tareas

### T15: datos para texto

La publicación ampliada contiene cinco archivos, todos en el manifiesto:

- `t15_copyTask_neuralData.zip`: **11,051 GB**, datos neurales para entrenar.
- `t15_pretrained_rnn_baseline.zip`: **0,485 GB**, modelo de referencia.
- `t15_copyTask.pkl`: **57,79 MB**, resultados del experimento.
- `t15_personalUse.pkl`: **1,13 MB**, resultados de uso conversacional.
- `README.md`.

Los `.pkl` de resultados no sustituyen al archivo de señales neurales.
El [repositorio de los autores](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text#data)
describe **10.948 frases, 45 sesiones y 20 meses**, con 512 características
neurales por intervalo de 20 ms, almacenadas en HDF5. Hay `train`, `val` y
`test`, pero **el test público no incluye las etiquetas de texto ni de fonemas**.
Para medidas locales usar la validación etiquetada, reservando parte de ella
como evaluación final si se necesita un test propio y documentando el protocolo.
No ajustar un modelo repetidamente contra el conjunto que después se reporte
como evaluación final.

T12 y T15 tienen entradas y emplazamientos de electrodos distintos. Mi propuesta
es comenzar con modelos separados y después probar una representación común
con capas de entrada específicas de cada participante. No se puede cargar el
checkpoint de T12 y esperar que funcione directamente sobre T15.

### Habla interna

La [publicación de Kunz](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j)
contiene datos de T12, T15, T16 y T17. Los participantes presentes en cada tarea
deben comprobarse al inspeccionar los archivos; no todas las tareas incluyen
a todos. `sentenceData.zip` (3,932 GB) es la parte más cercana al decodificador
de frases. `isolatedVerbalBehaviors.zip` (3,525 GB) y
`interleavedVerbalBehaviors.zip` (0,777 GB) permiten estudiar diferencias entre
conductas. Otros archivos contienen conteo, memoria, controles y un experimento
de activación mediante palabra clave.

La imaginación del habla cambia la tarea respecto al habla intentada. Conviene
registrar conducta, participante y sesión, y medir cada combinación por separado.
Comprobar posibles ensayos o sesiones repetidos entre publicaciones antes de
mezclar datos de los mismos participantes. El manifiesto lista sus doce archivos.

### Síntesis de voz

La [publicación de Wairagkar](https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f)
incluye `t15_neural_data.zip` (26,158 GB), `t15_control_experiment.zip`
(0,711 GB) y un README. La ampliación del **8 de mayo de 2026** publica
características neurales de múltiples sesiones, texto, tiempos de habla,
objetivos acústicos LPCNet alineados y modelos TensorFlow de referencia.
Permite investigar síntesis directa, además del camino texto a voz.
El experimento de control incorpora señales adicionales de micrófono e IMU.
No confundir el número de archivos o los modelos con nuevas grabaciones
independientes: identificar solapamientos de T15 antes de agregarlos a otras fuentes.

## Fuentes adicionales y acceso

- [Metzger 2023, ECoG y avatar](https://www.nature.com/articles/s41586-023-06443-4):
  los datos neurales están sujetos a solicitud mediante
  [Zenodo](https://doi.org/10.5281/zenodo.8200782). Los resultados de figuras
  disponibles públicamente no equivalen a un dataset de entrenamiento.
- [Littlejohn 2025, síntesis continua](https://www.nature.com/articles/s41593-025-01905-6):
  su apartado de datos describe acceso restringido para ECoG, enlaza
  [Harvard Dataverse](https://doi.org/10.7910/DVN/8TQKC8) y reutiliza el dataset
  intracortical de Willett 2023. No contarlo como otro dataset intracortical
  independiente; el conjunto EMG enlazado es otra modalidad.
- [Fogg 2026, preprint de decodificación entre seis participantes](https://doi.org/10.64898/2026.07.23.739430):
  referencia para investigar adaptación entre personas. No incorporado al catálogo
  de descargas: no se ha verificado aquí un depósito público de todos sus datos.
- EEG, MEG, ECoG, habla escuchada y EMG necesitan otra preparación y protocolo.
  Son líneas adicionales de investigación, no entradas intercambiables para
  el modelo intracortical elegido.

## Qué propongo descargar

Mi orden práctico, sujeto al almacenamiento elegido:

1. `competitionData` completo de T12 y datos neurales de T15: aproximadamente
   **14,721 GB** comprimidos. Añadir el checkpoint de T15 permite probar una referencia.
2. `sentences`, `diagnosticBlocks` y `tuningTasks`: **16,357 GB** más para
   contexto experimental y estabilidad temporal.
3. El conjunto de habla interna: **10,799 GB** para ampliar las conductas y personas.
4. Los datos de síntesis: **26,869 GB** para investigar salida acústica.
5. Recursos de lenguaje e intermedios según la implementación que se utilice.

Descargar todos los recursos de lenguaje puede consumir mucho más espacio que
los datos neurales. El [decodificador antiguo](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text#python-environment-setup-for-ngram-language-model-and-opt-rescoring)
declara aproximadamente 60 GB de RAM para tres gramos y 300 GB para cinco.
La alternativa [KenLM/Flashlight de los mismos investigadores](https://github.com/Neuroprosthetics-Lab/phoneme-to-words-lm)
declara menos de 10 GB de RAM para su decodificador y admite salida incremental.
Es una opción a medir, no un rendimiento verificado en nuestro equipo.

## Descargar y verificar

Ejecutar desde la raíz del repositorio:

```bash
python3 scripts/get_data.py --list-all

python3 scripts/get_data.py --dataset speech_2023 --files competitionData.tar.gz
python3 scripts/get_data.py --dataset speech_2024_t15 --files t15_copyTask_neuralData.zip t15_pretrained_rnn_baseline.zip

# Todos los archivos de UNA publicación; no extrae nada.
python3 scripts/get_data.py --dataset speech_2023 --files all

# Carpeta alternativa: se crea una subcarpeta por publicación.
python3 scripts/get_data.py --dataset inner_speech_2025 --files all --dest ../datos-bci

# Verificar un archivo descargado manualmente y colocado en data/speech_2023/.
python3 scripts/get_data.py --dataset speech_2023 --files competitionData.tar.gz --verify-only
```

Necesita Python 3 y `curl`, sin instalar dependencias de entrenamiento. Guarda
archivos incompletos como `.part`, reanuda y verifica bytes y hash antes de
dar una descarga por válida. Comprueba que la selección completa cabe dejando
5 GiB libres; **esa reserva no garantiza espacio para la extracción**.
No sobrescribe archivos completos con un hash incorrecto.
Los datos quedan ignorados por Git dentro de `data/`; evitar versionarlos si
se utiliza un destino alternativo dentro del repositorio.

La extracción y el número exacto de ensayos por sesión se verificarán sobre los
archivos descargados. Tener un manifiesto no significa tener las grabaciones.

En esta sesión las peticiones directas desde terminal devolvieron HTTP 403,
mientras que la descarga individual desde el navegador sí funcionó para los
README. El script deja ese error visible y permite verificar archivos
descargados manualmente; no se ha confirmado que Dryad acepte `curl` desde
este equipo. La descarga agrupada llegó a mostrar progreso pero no produjo
un archivo local confirmado; usar la descarga individual y verificar el hash.

## Criterio de funcionalidad

Propongo una aplicación que procese características neurales sin leer la
transcripción de referencia durante la predicción, mantenga estado entre
ventanas y produzca texto incremental. La evaluación debería medir PER/WER,
latencia, velocidad sostenida, errores de inicio/final de habla y estabilidad
en nuevas sesiones. Una red bidireccional debe declararse como offline o
medirse con su ventana de anticipación; reproducir visualmente una frase
precalculada no demuestra funcionamiento en streaming.

La evaluación entre participantes necesita calibración o un procedimiento
explícito de adaptación. Estas pruebas con grabaciones permiten construir un
decodificador útil y reproducible; conectar un implante y comprobar el circuito
con una persona requiere un entorno de adquisición al que aquí no tenemos acceso.

## Adquisición inicial verificada

El 30 de septiembre de 2026 se descargó `competitionData.tar.gz` por la
descarga individual del navegador, se verificó su SHA-256 contra Dryad y se
guardó en `data/speech_2023/`. También se descargaron y verificaron sus siete
README. Se extrajo con el filtro seguro de `tarfile` en
`data/speech_2023/competitionData/`; quedan disponibles los `.mat` originales.
El índice real del archivo está en
`docs/datasets/speech_2023/archive_index.json`:

| Partición | Archivos `.mat` |
|---|---:|
| `train` | 24 |
| `test` | 24 |
| `competitionHoldOut` | 15 |

Total: **63 archivos**, **6.949.015.400 bytes** de contenido al extraer.
Cada archivo contiene múltiples ensayos; estos números no son cantidades
de frases. El número exacto de ensayos, las formas de las matrices y las
etiquetas se comprobarán con el cargador de MATLAB antes de entrenar.
Los otros archivos grandes del catálogo todavía no se han descargado.
