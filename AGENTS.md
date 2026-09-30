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
cerrada. Aplicativo previsto: decodificación y visualización de grabaciones,
con transcripción y comparación con la referencia.

El usuario decidió cambiar desde handwriting el 30 de septiembre de 2026.
Quiere explorar los datos completos y otros participantes para desarrollar
un decodificador funcional. En esta fase autoriza explorar escalas mayores
sin limitar el diseño a la T4; no dar por medida la viabilidad ni por aceptados
los cambios de requisitos docentes. El inventario está en `docs/datos.md`.

## Restricciones del curso

- **Regla 5/20:** `python run.py --smoke` en menos de 5 min en CPU;
  el entrenamiento completo en menos de 20 min en una T4 de Colab gratuito.
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
- Pendiente: adaptar datos, modelo, entrenamiento, métricas y aplicativo;
  todos siguen siendo placeholders. README y propuestas conservan la ficha
  inicial de handwriting y necesitan una actualización posterior.
- No mezclar `competitionData` y `sentences` como muestras independientes;
  comprobar solapamientos de participantes/sesiones entre publicaciones.
- Los tests publicados de T15 no incluyen etiquetas. Los datos de T12 y T15
  requieren adaptadores específicos; no asumir entradas intercambiables.
