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
- Hecho: lector real `src/data/speech.py` e informe reproducible con
  `scripts/explore_data.py`, `configs/exploration.yaml` y entorno ligero
  Python 3.12 (`requirements-exploration.txt`). Guía en `docs/primeros_pasos.md`.
  Inventario por metadatos: 8.800/880/1.200 ensayos en train/test/holdout.
  Validación de valores limitada a la sesión seleccionada; cuatro pruebas
  específicas del lector pasan. No normaliza ni entrena.
- Preferencia del usuario: exploración en notebooks Python para ver y
  ejecutar celdas. Entrada principal `notebooks/01_exploracion_t12.ipynb`;
  diez celdas de código ejecutadas con gráficos incrustados. Kernel local
  `trabajo-integrado-exploracion` en `.venv` (Python 3.12). El HTML es alternativo.
- Pendiente: conectar el lector con batches de longitud variable, fonemas,
  modelo, entrenamiento, métricas y aplicativo. El pipeline de `run.py`
  sigue siendo la plantilla sintética. README actualizado; las propuestas
  conservan ideas históricas de handwriting y necesitan revisión posterior.
- No mezclar `competitionData` y `sentences` como muestras independientes;
  comprobar solapamientos de participantes/sesiones entre publicaciones.
- Los tests publicados de T15 no incluyen etiquetas. Los datos de T12 y T15
  requieren adaptadores específicos; no asumir entradas intercambiables.
