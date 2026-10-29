# Primera ronda T12 · semilla 0

29 de octubre de 2026. Cuatro entrenamientos completos de 10.001 pasos,
con recarga, controles de señal y benchmark verificados. Esta tabla reúne
las observaciones de validación; no contiene test ni conclusiones científicas.

| Modelo | Salida | PER val | Mediana forward | Señal cruda en RAM → fonemas |
|---|---:|---:|---:|---:|
| Baseline | 5 | 27,69 % | 5,077 ms | 6,131 ms |
| Conjunta | 2 | 29,53 % | 2,541 ms | 3,525 ms |
| Conjunta | 3 | 28,89 % | 3,387 ms | 4,431 ms |
| Conjunta | 5 | 28,10 % | 5,092 ms | 6,167 ms |
| Independiente 2 | 2 | 29,60 % | 2,538 ms | 3,568 ms |
| Independiente 3 | 3 | 27,89 % | 3,291 ms | 4,342 ms |

Cada PER agrega errores sobre los mismos 41.571 fonemas de 1.460 frases.
La conjunta comparte un único checkpoint elegido por su salida 5: las tres
filas no son tres entrenamientos ni selecciones independientes por cabeza.
Las redes independientes se seleccionan por su propia salida final.

Los tiempos son medianas de 38 frases de val, lote 1 y cinco repeticiones
en A100-SXM4-40GB. Las firmas de entrada y el protocolo coinciden. Se
reprodujeron recuentos y medianas desde las tablas originales de W&B.
No incluyen lectura .mat desde disco; las repeticiones no son semillas.
Los JSON originales de benchmark con UUID de GPU siguen en Drive: faltan
por copiar antes de generar las figuras con el validador estricto existente.

Datos reutilizables: [CSV](t12_semilla0_resumen.csv) y
[registro con fuentes y hashes](t12_semilla0_resumen.json). Registros completos:
[baseline](speech_baseline_ofxo8x9a.json), [conjunta](speech_conjunta_7b2n7v3o.json),
[independiente 2](speech_control_2_153amw3i.json) e
[independiente 3](speech_control_3_yzcwdxn1.json).
