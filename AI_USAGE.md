# Bitácora de uso de IA

Obligatoria según la guía docente. Vale **4 puntos de E2**.

**No se puntúa usar poco.** Se puntúa el criterio: una entrada que dice
«se lo pedimos, nos dio esto, lo rechazamos porque…» vale más nota que no
haber usado ninguna herramienta.

**La línea roja:** la IA se puede usar para comprender y generar código,
citándola. **No** para el análisis automático de los resultados ni para la
discusión reflexiva sobre el trabajo. Eso es lo que evalúa la asignatura, y se
comprueba en la defensa.

---

| Fecha | Tarea | Herramienta | Qué devolvió | Qué hicimos y por qué |
|---|---|---|---|---|
| 2026-09-30 | Revisar el cambio a speech e inventariar datos públicos | Codex | Catálogo con fuentes, manifiestos oficiales de cuatro publicaciones y script de descarga/verificación; señaló solapamientos y etiquetas ocultas | Trabajo solicitado por el autor. Datos, métricas y adaptación del modelo pendientes de inspección; las recomendaciones de alcance quedan para revisión humana. |
| 2026-09-30 | Primer contacto con competitionData del paciente T12 | Codex | Crear un notebook para ver y leer los datos de una sesión | Usado para ver cómo están organizados los ensayos y las señales antes de empezar a entrenar. |
| 2026-10-11 | Ampliar las pruebas con fonemas, palabras y frases de T12 | Codex | Generó código para comparar señales y canales, probar centroides y LDA, validar por bloques y entre días, mezclar etiquetas y revisar las frases. | El usuario marco la ruta a explorar. |
| 2026-10-11 | Medir el tiempo de entrenamiento y guardar la prueba en el repo | Codex | Generó y ejecutó un piloto de GRU + CTC en RTX 2060, guardó las medidas por paso y el código, y calculó escenarios de duración. Preparó un borrador para explicar el recorte. | Yo pedí medir el tiempo, usar la GPU y guardar la prueba para valorar mantener 5 capas con 2.000 frases. El recorte sigue pendiente de entrenamiento completo y de mi revisión; no se ha comprobado aún PER ni el límite en T4. |
| 2026-10-12 | Preparar una presentación nueva del proyecto | Codex | Generó una presentación HTML blanca de tres diapositivas, diagramas animados, una maqueta con entrada de señal por bloques y un borrador de guion basado en los datos y la prueba de tiempo existentes. Contrastó la comparación de calidad con el paper y aclaró la diferencia entre WER y accuracy. Corrigió una primera cifra sin cálculo por una expectativa de ≈70 % en 100 − WER, explicando la aproximación con la curva de datos de Fig. 5i. Separó los riesgos del aplicativo en dos animaciones: procesamiento lento con señal pendiente y predicciones equivocadas al ritmo normal. | Yo marqué los tres temas, el fondo blanco, las animaciones y que solo aparezca lo que voy a explicar. Después pedí numerar las diapos, quitar el motivo de elección, orientar el aplicativo a emular tiempo real, dar una cifra esperada con una razón detrás y mostrar los dos fallos por separado. El ≈70 % traslada de forma aproximada una pérdida de ≈7 puntos de WER observada al reducir días de datos en el paper; supone entrenamiento y modelo de lenguaje comparables, y no es un resultado nuestro. El guion queda para adaptarlo con mis palabras; el tiempo se muestra como estimado y las predicciones, el retraso y la cola de las animaciones como simulados. |

---

<!--
Ejemplos del nivel que se espera (bórralos cuando tengas los tuyos):

| 12-nov | Escribir el Dataset con el subset del paper | Claude Code | Clase completa con __len__ y __getitem__ | **Aceptado con cambios.** Normalizaba con estadísticas del conjunto entero; lo cambiamos a las de train para no meter fuga |
| 15-nov | «¿Por qué mi loss no baja?» | Claude | Cinco hipótesis | **Rechazado como análisis.** Lo usamos como lista de comprobación; el diagnóstico (lr 100× alto) lo encontramos con el LR range test |
-->
