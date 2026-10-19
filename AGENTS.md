# Contexto del proyecto

<!--
Fichero de contexto para asistentes de IA (Claude Code, Copilot, Cursor…).
Se explica en la sesión A1 del 9 de noviembre.

Dos criterios para escribirlo:
  1. Va lo que le dirías a alguien que se incorpora hoy al proyecto.
  2. Si te encuentras repitiendo la misma corrección tres veces, esa
     corrección va aquí.

Dos páginas se leen enteras. Quince, no.
-->

## Qué es esto

Proyecto elegido: replicación offline de *A high-performance speech
neuroprosthesis* (Willett et al., 2023) sobre datos intracorticales públicos de
Dryad. Evaluación con PER y WER; aún no hay resultados ni una métrica objetivo
cerrada. Aplicativo previsto: emular la llegada de señal en tiempo real a
partir de grabaciones, decodificar por bloques y mostrar texto durante la
recepción. El usuario aclaró este enfoque el 12 de octubre de 2026. La
comparación con la referencia sirve para evaluar el sistema.

El usuario decidió cambiar desde handwriting el 30 de septiembre de 2026.
Quiere explorar los datos completos y otros participantes para desarrollar
un decodificador funcional. En esta fase autoriza explorar escalas mayores
sin limitar el diseño de la versión completa a la T4. El 10 de octubre de
2026 aclaró que el límite docente de entrenamiento de 20 minutos sigue
vigente: se harán dos versiones con el mismo código y configuraciones en
`configs/`. La versión de entrega debe cumplir ese límite, con los recortes
de datos o modelo necesarios y una posible menor calidad. La versión
completa usará todos los datos de entrenamiento previstos y el tiempo
necesario, sin ese límite de 20 minutos. Ambas deben respetar las demás
restricciones del proyecto. La viabilidad, las métricas y el consumo de
cómputo siguen pendientes de medir. El inventario está en `docs/datos.md`.

## Restricciones del curso

- **Prueba corta:** `python run.py --smoke` en menos de 5 min en CPU.
- **Entrenamiento de entrega:** menos de 20 minutos; medir el tiempo con
  la configuración y el hardware utilizados antes de afirmar que cumple.
- **Entrenamiento de la versión completa:** sin el límite de 20 minutos,
  manteniendo las demás restricciones y registrando tiempo, recursos y
  métricas por separado de la versión de entrega.
- Dispone de una RTX 2060 y contempla Colab de pago con 200 unidades de
  cómputo; no asumir la contratación ni una GPU concreta, ni prometer
  tiempos o consumo sin medir.
- Todo entrenamiento se registra en Weights & Biases.
- Las semillas se fijan **solo** en `src/utils/seed.py`.
- Los hiperparámetros viven **solo** en `configs/`. Ninguno escrito a fuego.

## Cómo se ejecuta

```bash
python run.py --smoke                          # pipeline entero en miniatura
python run.py --config configs/default.yaml    # entrenamiento completo
pytest tests/ -q                               # tests
python tools/tiiac_check.py .                  # comprobador de reproducibilidad
```

## Convenciones

- Sin rutas absolutas, nunca.
- Toda función pública lleva docstring y anotaciones de tipo.
- **La normalización usa estadísticas de entrenamiento, no del conjunto completo.**
  Lo contrario es una fuga de datos: no da error e infla la métrica.
- El `data augmentation` se aplica a train, jamás a validación ni a test.
- `src/app/` no importa nada de `src/training/`: el aplicativo solo carga un
  checkpoint y responde.

## Lo que NO hay que tocar

- `src/data/splits.py`: los splits están congelados desde el hito H3.
- <lo que decidáis vosotros>

## Estado actual

- Hecho: inventario de cuatro publicaciones de Dryad y manifiestos oficiales
  con tamaños, enlaces y SHA-256 en `docs/datasets/`; descarga reanudable y
  verificación en `scripts/get_data.py`.
- Hecho: `competitionData.tar.gz` y sus siete README descargados y
  verificados con SHA-256 oficial; corpus extraído en
  `data/speech_2023/competitionData/` (24 archivos de train, 24 de test,
  15 de competitionHoldOut). Datos ignorados por Git.
- En curso: adquisición e inspección de grabaciones; comprobar archivos
  locales antes de afirmar que un dataset está disponible en el equipo.
- Hecho: lector real `src/data/speech.py` y notebooks de exploración. Inventario
  por metadatos: 8.800/880/1.200 ensayos en train/test/holdout. El lector no
  normaliza ni entrena. `scripts/explore_data.py`, `docs/primeros_pasos.md`
  y los tests específicos del lector mencionados anteriormente no existen
  en el checkout del 19 de octubre.
- Hecho: modelo `src/models/speech_gru.py`, integrado en la factoría y con
  `configs/speech_original.yaml`: baseline offline TensorFlow, cinco GRU
  unidireccionales de 1.024 unidades, kernel 32 y stride 4; selección oficial
  de 19 sesiones (6.640 frases de train). El paquete completo tiene 24 sesiones.
  El modelo recibe características ya normalizadas y suavizadas. Los comandos
  de preparación y entrenamiento están en README.
- Los notebooks se conservan como trabajo anterior. El usuario ha pedido
  continuar con código de la entrega, sin nuevos notebooks.
- Hecho: `src/data/preprocessing.py` extrae tx1 antes de spikePow, ajusta
  normalización por sesión leyendo exclusivamente train y restaura esas
  estadísticas sin volver a ajustarlas. Incluye el suavizado gaussiano del
  original, sobre tensores PyTorch, para aplicarlo después de la futura
  ampliación con ruido. Parámetros en `configs/speech_original.yaml`.
- Hecho: `src/data/phonemes.py` convierte las referencias con g2p-en 2.1.0,
  siguiendo la limpieza y SIL del original. IDs 0–39; blank CTC 40. Rechaza
  etiquetas ocultas. Recursos NLTK fijados por commit y SHA-256, preparados
  con `scripts/get_phoneme_resources.py`; conversión posterior sin red.
  `scripts/prepare_phoneme_labels.py` genera los JSON en data/ para las 19
  sesiones: 6.640 referencias de train y 680 de test, con manifiesto de
  versiones, vocabulario y hashes. No carga matrices neuronales ni entrena.
- Hecho: `scripts/prepare_speech_data.py` guarda las señales normalizadas en
  38 NPY (2,44 GB) con estadísticas de train, referencias y hashes. No añade
  ruido ni suaviza. `src/data/speech_dataset.py` lee ensayos por memmap,
  verifica la correspondencia con las etiquetas y su longitud mínima CTC.
  `construir_loaders` devuelve lotes de habla: padding cero solo para señal,
  objetivos concatenados, longitudes e IDs de sesión. El entrenamiento elige
  uniformemente una sesión por lote; el muestreo usa un RNG independiente
  creado en `src/utils/seed.py`. El segundo loader es el test publicado,
  no una validación independiente. Los datos preparados están fuera de Git.
- Hecho: `run.py --config configs/speech_original.yaml` conecta datos reales,
  ruido solo en train, suavizado, CTC, Adam con epsilon-hat de TF, L2 y clipping.
  Evalúa PER greedy agregado por fonemas sobre el test publicado, sin WER ni
  validación independiente. Registra W&B (online por defecto), tiempos y
  recuentos. Guarda ultimo.pt por run con pesos, configuración, normalización
  y firma de los datos; --solo-evaluar exige checkpoint compatible. No selecciona
  el mejor checkpoint por PER de test ni reanuda entrenamiento todavía.
  Comprobado con dos actualizaciones reales en CPU y RTX 2060, recarga con
  CTC/PER idénticos en CPU y evaluación completa de las 680 frases en GPU.
  El sandbox ocultaba CUDA; fuera de él sí está disponible la RTX 2060.
  Estas ejecuciones cortas no son el entrenamiento completo ni un resultado
  de calidad. Sus configs, pesos e historiales W&B están ignorados por Git.
- El primer run completo sincronizó W&B pero agotó la memoria de la RTX 2060
  al llegar a un lote de 64 frases con 904 bins (paso 21 con seed 0). La
  comprobación anterior usaba lotes de dos frases y no cubría esa memoria.
  Se mantiene batch_size=64 y se acumulan gradientes con micro_batch_size=4;
  CTC se pondera por el número total de frases, L2/clipping/Adam se aplican
  una vez por lote efectivo. Evaluación también usa microlotes y conserva
  todo el test. No recortar padding temporal de train: el ruido y suavizado
  se definen sobre la extensión del lote original. Cambia el orden del RNG
  frente al lote monolítico, sin reducir datos, arquitectura o precisión.
- Pendiente: entrenamiento completo, configuración reducida y aplicativo.
  `run.py --smoke` y default.yaml conservan la plantilla sintética y no
  comprueban el pipeline de habla. README aún contiene campos de la
  plantilla. La propuesta vigente de habla está en `docs/propuesta_final.md`;
  `docs/propuestas.md` conserva las propuestas históricas.
- Entorno local: `pip check` señala un conflicto previo entre
  `googleapis-common-protos==1.75.5` (requiere protobuf >=6.33.5) y
  `protobuf==5.29.6`. Ninguno fue cambiado al instalar g2p-en. Las
  comprobaciones de etiquetas y CTC pasan; W&B 0.18.7 importa correctamente.
- No mezclar `competitionData` y `sentences` como muestras independientes;
  comprobar solapamientos de participantes/sesiones entre publicaciones.
- Los tests publicados de T15 no incluyen etiquetas. Los datos de T12 y T15
  requieren adaptadores específicos; no asumir entradas intercambiables.
