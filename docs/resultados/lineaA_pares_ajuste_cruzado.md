# Línea A: pares fonemas → frase por ajuste cruzado

Partición `t12_bloques_v2`; split de val y bloques de train que cada modelo no vio.
Train y val salen de la misma receta causal; solo cambia el pliegue apartado.
Train apartado usa la normalización por bloque de train; val, la rodante.

| Modelo | Pliegue | Split | Frases | PER exportado (%) | PER de val en train (%) |
|---|---:|---|---:|---:|---:|
| t12_causal_pliegue0_seed0 | 0 | train | 1400 | 24,67 | 25,83 |
| t12_causal_pliegue1_seed0 | 1 | train | 1460 | 23,95 | 25,91 |
| t12_causal_pliegue2_seed0 | 2 | train | 1360 | 23,91 | 25,99 |
| t12_causal_pliegue3_seed0 | 3 | train | 1320 | 24,43 | 25,95 |
| t12_causal_pliegue4_seed0 | 4 | train | 1360 | 24,04 | 25,76 |
| t12_causal_seed0 | — | val | 1900 | 23,86 | 23,86 |
| t12_causal_seed1 | — | val | 1900 | 24,98 | 24,98 |
| t12_causal_seed2 | — | val | 1900 | 25,02 | 25,02 |
| t12_causal_seed3 | — | val | 1900 | 25,09 | 25,09 |
| t12_causal_seed4 | — | val | 1900 | 25,12 | 25,12 |

| Split | Frases | PER agregado (%) |
|---|---:|---:|
| train | 6900 | 24,20 |
| val | 9500 | 24,81 |
