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

Desde la raíz, con el entorno activado. El entrenamiento T12 se ejecuta en Colab
con `notebooks/07_baseline_t12_colab.ipynb` para el baseline y
`notebooks/08_salidas_t12_colab.ipynb` para la variante con tres salidas.

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
- Pendiente: lanzador de la comparación y medidas de calidad y coste.
