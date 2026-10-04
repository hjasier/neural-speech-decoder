# Fuentes y créditos

## Artículos y datos

- Willett, F. R. et al. (2023). [A high-performance speech neuroprosthesis](https://doi.org/10.1038/s41586-023-06377-x). *Nature*. Paper elegido, método GRU y CTC, participante T12 y resultados originales: 62 palabras/minuto y 23,8 % de WER en la condición de habla intentada con vocalización y vocabulario de 125.000 palabras.
- Willett, F. R. et al. (2021). [High-performance brain-to-text communication via handwriting](https://doi.org/10.1038/s41586-021-03506-2). *Nature*. Antecedente de decodificación de intentos de escritura. La cifra de 18 palabras/minuto aparece como comparación en el artículo de 2023; corresponde a otro estudio.
- [Datos públicos del experimento de 2023 en Dryad](https://doi.org/10.5061/dryad.x69p8czpq). Paquete `competitionData`: 8.800 ensayos de train en 24 sesiones, 880 ensayos etiquetados de test y 1.200 con etiquetas ocultas. Son datos de una participante y no permiten afirmar generalización a otras personas.
- Feghhi et al. (2025). [Time-Masked Transformers with Lightweight Test-Time Adaptation for Neural Speech Decoding](https://arxiv.org/abs/2507.02800). Referencia para la comparación propuesta con un Transformer causal y enmascaramiento temporal.
- Li et al. (2024). [Brain-to-Text Decoding with Context-Aware Neural Representations](https://arxiv.org/abs/2411.10657). Referencia para explorar transiciones entre fonemas y la parte de lenguaje.

## Figuras del paper

`assets/paper-cover.png`, `implant.png`, `paper-pipeline.png` y `paper-results.png` se extraen del PDF original de Willett et al. (2023) aportado por el usuario. Corresponden a la primera página y detalles de las figuras 1a, 2a y 2b–c, respectivamente. Los detalles se han recortado y rasterizado; no se han alterado sus datos ni etiquetas. El artículo se distribuye con licencia [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

## Visualizaciones de registros reales

Los tres mapas y las curvas de entrada proceden de `data/speech_2023/competitionData/train/` y se prepararon con el lector del repositorio:

| Recurso | Sesión | Índice del ensayo en el archivo | Frase de referencia |
|---|---|---|---|
| `session-1.png` | t12.2022.04.28 | 21 | A good attitude is unbeatable. |
| `session-2.png` | t12.2022.06.23 | 0 | It went the other way. |
| `session-3.png` | t12.2022.08.25 | 0 | Is that pollution? |

Se muestran las primeras 128 columnas de `spikePow`, correspondientes a 6v. Se aplica `log1p` para visualizar; cada mapa usa su propia escala hasta el percentil 99. Las curvas muestran los canales 9, 31, 64 y 100, reescalados entre los percentiles 2 y 98 para facilitar su lectura. No son predicciones, ni una alineación con fonemas, ni el preprocesamiento de un modelo entrenado. Las frases y las escalas difieren: estos ejemplos no constituyen por sí solos una medida controlada de drift.

`assets/data-source.json` y `assets/data.js` conservan los metadatos y las curvas que utiliza la presentación. El selector muestra números de ensayo comenzando por 1; esta tabla conserva los índices comenzando por 0 del lector Python.

## Ilustraciones generadas

`assets/brain.png` y `assets/conversation.png` fueron generadas con la herramienta de imágenes de Codex para esta presentación. El cerebro es conceptual; los puntos no representan la ubicación real de los implantes. Las dos mujeres son personas ficticias para ilustrar una conversación; ninguna representa a T12. Los prompts exactos se guardan en [assets/generation-prompts.json](assets/generation-prompts.json).

Los esquemas animados de cursor, escritura, señal, GRU, CTC y enmascaramiento son ilustraciones didácticas realizadas en HTML, CSS y SVG. La salida «hello» es un ejemplo del método, no una predicción propia. La pantalla del explorador es una propuesta de interfaz con visualizaciones reales y una transcripción pendiente de integrar.

## Software y tipografía

- [Reveal.js](https://revealjs.com/) 6.0.2, licencia MIT. Distribución oficial obtenida del registro npm y comprobada con su integridad SHA-512. Copia de la licencia en `vendor/reveal/LICENSE` y manifiesto en `vendor/manifest.json`.
- [Manrope](https://github.com/sharanda/manrope), licencia SIL Open Font License; copia en `assets/fonts/OFL.txt`. Tipografía cargada en local.
- Mapas estáticos generados con Matplotlib. La presentación no necesita Python ni una conexión a los datos originales para ejecutarse.
