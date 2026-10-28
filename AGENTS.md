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

Decodificador offline en PyTorch basado en la GRU + CTC de Willett et al.
(2023), con datos intracorticales T12 de `competitionData`. La métrica actual
es PER de fonemas; el modelo de lenguaje para obtener texto y medir WER
queda pendiente. No es una reproducción del sistema clínico completo.

La propuesta estudia salidas tras las capas 2/3/5 y su compromiso entre PER
y latencia. El aplicativo previsto emulará la recepción de grabaciones por
bloques y mostrará la decodificación. Alice es una exploración anterior.

## Restricciones del curso

- **Prueba corta:** `python run.py --smoke` en menos de 5 minutos en CPU.
  Para E1 basta demostrar entorno, pipeline real, explicación y W&B;
  el entrenamiento completo no está limitado a 20 minutos.
- Todo entrenamiento se registra en Weights & Biases, proyecto `tiiac-2026`.
- Las semillas se fijan **solo** en `src/utils/seed.py`.
- Los hiperparámetros viven **solo** en `configs/`. Ninguno escrito a fuego.
- Entrenamiento y cálculos GPU en Colab. No usar GPU local, tampoco para
  inferencia o demos, por los bloqueos del PC. Comprobaciones CPU pequeñas,
  con CUDA oculta y dos hilos; no entrenar de nuevo para verificar cambios menores.

## Cómo se ejecuta

Desde la raíz, con el entorno activado. En Colab,
`notebooks/09_comparativa_t12_colab.ipynb` permite elegir variante y semilla.
Se conservan `notebooks/07_baseline_t12_colab.ipynb` para el baseline y
`notebooks/08_salidas_t12_colab.ipynb` para la prueba de 200 pasos.

```bash
python run.py --smoke                          # plantilla sintética, no T12
python run.py --config configs/default.yaml    # modelo de la plantilla
python run.py --config configs/speech_comparacion.yaml  # T12 con validación
pytest tests/ -q                               # tests
python tools/tiiac_check.py .                  # comprobador de reproducibilidad
```

En comprobaciones CPU locales, anteponer
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`.
Los datos y pesos no se copian automáticamente entre carpetas de trabajo.

## Convenciones

- Sin rutas absolutas en el proyecto.
- Toda función pública lleva docstring y anotaciones de tipo.
- **La normalización usa exclusivamente train.** Ajustarla antes de separar
  validación produce fuga de datos.
- El `data augmentation` se aplica a train, jamás a validación ni a test.
- Seleccionar `mejor_val.pt` por mínimo PER de val completa, primero en empate.
  Reservar test para la evaluación final explícita; no confundir PER y WER.
- `src/app/` no importa nada de `src/training/`: el aplicativo solo carga
  un checkpoint y responde. `run.py` orquesta y los notebooks son lanzadores.
- No mezclar protocolos, presupuestos ni pruebas sintéticas con resultados reales.
  `competitionData` y `sentences` se solapan: no sumarlos como muestras nuevas.
- Stage y commits manuales. Revisar el diff antes de editar y conservar cambios
  ajenos. No publicar ramas, abrir PRs ni hacer merges sin indicación explícita.
- La rama de integración es `ml-train`. Cada feature parte de ella después
  de integrar la anterior; conservar los commits individuales al hacer merge.
- Registrar las intervenciones relevantes en `AI_USAGE.md`, en primera persona,
  con el contexto y la ayuda recibida, para revisión antes de comitear.
  No forzar entradas por ajustes menores ni restaurar las descartadas en revisión.
  Conservar las entradas revisadas; no inventar decisiones ni análisis de resultados.
- Mantener este archivo breve: actualizar lo vigente y quitar lo obsoleto.
  Historiales, resultados y explicaciones extensas van en `docs/`.

## Lo que NO hay que tocar

- `docs/datasets/speech_2023/particion_t12_v1.json`: partición congelada;
  conservar su contenido y hash. La lógica vive en `src/data/speech_protocol.py`.
- Datos originales, pesos y salidas de notebooks ejecutados: conservarlos;
  no regenerar snapshots sobre ejecuciones guardadas ni versionar credenciales.

## Estado actual

- Hecho: baseline GRU + CTC, pipeline y W&B; protocolo T12 de 19 sesiones,
  con 5.180 frases train, 1.460 val y 680 test, selección de pesos por val
  y lanzador Colab propio.
- Hecho: salidas opcionales tras las GRU 2 y 3; cada perfil ejecuta solo
  el prefijo elegido y se conserva la salida final del baseline.
- Hecho: CTC conjunta con pesos 0,25/0,25/0,50 en configuración y métricas
  por salida en W&B. Se selecciona un mismo checkpoint por PER de val de
  la salida final; los pesos guardan también las cabezas intermedias.
- Hecho: verificador de recarga por perfil sobre val y selección explícita
  con `run.py --solo-evaluar --split val`. Perfil de 200 pasos disponible
  en `configs/speech_salidas_e1.yaml`.
- Hecho: notebook 08 con perfiles de 200 y 10.001 pasos, guardado en Drive
  y recarga independiente en val. El notebook 07 conserva el baseline anterior.
- Hecho: controles independientes de dos y tres capas y selector de variante
  y semilla en `scripts/configure_speech_comparison.py`. El plan vive en
  `configs/speech_comparativa.yaml` y exige datos y presupuesto comunes.
- Hecho: notebook 09 para ejecutar una combinación de variante y semilla,
  con W&B, guardado en Drive y recarga independiente sobre val.
- Hecho: `scripts/evaluate_speech.py` y celda 12 del notebook 09, con PER
  global y por sesión, predicciones y controles de señal original/cero/permutada.
  El diagnóstico usa val por defecto; los controles se aplican en inferencia.
- Hecho: `scripts/benchmark_speech.py` y celda 13 del notebook 09, con carga,
  forward y señal cruda a fonemas medidos por separado sobre muestras comunes.
  En la red conjunta todos los pesos siguen residentes al elegir una salida.
- Hecho: notebook 10 con actividad neuronal y selector de salidas desde
  checkpoints compatibles. La configuración local usa CPU y el notebook
  oculta CUDA; mantenerlo así al incorporar versiones históricas posteriores.
- Hecho: `scripts/plot_speech_results.py` genera figuras y tablas desde
  registros compatibles, con hashes de procedencia. Las figuras de referencia
  están en `docs/figuras/t12_referencia/`; no interpretan los resultados.
- Hecho: entorno local fijado. El registro original de
  reproducibilidad conserva las comprobaciones y hashes de aquel contexto.
- Hecho: notebook 08 ejecutado en T4, run `xz71t4vd`, 200 actualizaciones
  y recarga independiente de las salidas 2/3/5, sin actualizar pesos ni usar test.
  Registro y curva originales en `docs/resultados/speech_salidas_e1_xz71t4vd.json`
  y `docs/figuras/t12_e1_xz71t4vd/`.
- Hecho: demo CPU conectada a esos pesos; controles y medidas históricas en
  `docs/resultados/*_xz71t4vd_mediciones.json`, figuras en `docs/figuras/t12_e1_mediciones/`.
  Se generan desde `configs/speech_figuras_e1.yaml` sin cargar el modelo.
  La demo ejecutada original se conserva aparte en `docs/resultados/*_demo_ejecutada.ipynb`.
- Hecho: notebook 09 conserva el baseline interrumpido `m6sn4um1`, con
  5.801 pasos sincronizados de 10.001 y 117 evaluaciones completas de val.
  Su registro y curva están en `docs/resultados/speech_baseline_m6sn4um1.json`
  y `docs/figuras/t12_baseline_m6sn4um1/`; no hay cierre normal del entrenamiento.
- Hecho: el pipeline local lee `WANDB_API_KEY` desde `.env` si no está
  definida en el entorno. Solo carga esa variable, sin imprimirla ni añadirla
  a la configuración. La lectura desde Drive se incorporará con el notebook 11.
- Hecho: `entrenamiento.pt` guarda modelo, Adam, paso, estado aleatorio,
  muestreo, historial y selección. `run.py --reanudar RUTA` exige datos,
  fuentes y configuración compatibles. Los pesos antiguos son de inferencia.
  Online conserva el run de W&B; offline usa segmentos con un ID lógico común.
- Pendiente: incorporar el perfil y el lanzador reanudable de Colab.
