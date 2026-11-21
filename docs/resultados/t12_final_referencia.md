# Tabla de referencia T12, partición t12_bloques_v2

Split: val. Generada por `scripts/build_reference_table.py`; no editar a mano.
PER de val: evaluación completa del entrenamiento. WER y PER por bloques: evaluación
causal con el trigrama. Media ± desviación muestral entre semillas (mínimo–máximo).

| Variante | Modelo | Normalización | Semillas | PER val (%) | WER (%) | Semillas con WER |
|---|---|---|---:|---:|---:|---:|
| t12_causal | 512/14 causal | rodante | 5 | 24,81 ± 0,54 (23,86–25,12) | 28,20 ± 0,69 (27,01–28,76) | 5 |
| t12_offline | 1024/32 centered | rodante | 5 | 20,80 ± 0,07 (20,75–20,92) | — | 0 |
| t12_causal_bloque | 512/14 causal | bloque | 3 | 23,91 ± 0,45 (23,41–24,27) | 26,53 ± 0,68 (25,76–27,05) | 3 |
| t12_causal_train_fija | 512/14 causal | train_fija | 3 | 32,09 ± 0,57 (31,43–32,47) | 39,12 ± 1,29 (37,62–39,90) | 3 |

Por trabajo. Intervalos al 95 %, bootstrap sobre frases.

| Trabajo | Run | Paso | PER val (%) | WER (%) | WER vocalizada (%) | WER silenciosa (%) | PER por bloques (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| t12_causal_seed0 | ddmfz9oq | 9901 | 23,86 | 27,01 [25,89; 28,15] | 27,09 [25,82; 28,36] | 26,71 [24,42; 29,13] | 23,86 [23,21; 24,49] |
| t12_causal_seed1 | a52ft9pz | 9751 | 24,98 | 28,76 [27,63; 29,92] | 28,97 [27,67; 30,26] | 27,91 [25,52; 30,41] | 24,98 [24,34; 25,60] |
| t12_causal_seed2 | boetvrq2 | 9801 | 25,02 | 28,50 [27,36; 29,65] | 29,01 [27,68; 30,33] | 26,45 [24,05; 28,97] | 25,02 [24,40; 25,64] |
| t12_causal_seed3 | c0mu8nug | 9951 | 25,09 | 28,35 [27,20; 29,52] | 28,50 [27,22; 29,78] | 27,74 [25,26; 30,32] | 25,09 [24,46; 25,72] |
| t12_causal_seed4 | sy2g8c02 | 9951 | 25,12 | 28,38 [27,26; 29,51] | 28,43 [27,16; 29,71] | 28,21 [25,70; 30,79] | 25,12 [24,49; 25,73] |
| t12_offline_seed0 | ozp0c1g7 | 9751 | 20,79 | — | — | — | — |
| t12_offline_seed1 | c03xy5om | 10001 | 20,92 | — | — | — | — |
| t12_offline_seed2 | 2xm6sgxu | 9951 | 20,77 | — | — | — | — |
| t12_offline_seed3 | 5h7vi0lg | 9151 | 20,80 | — | — | — | — |
| t12_offline_seed4 | 963yxi02 | 9701 | 20,75 | — | — | — | — |
| t12_causal_bloque_seed0 | mm610yn2 | 9751 | 23,41 | 25,76 [24,69; 26,79] | 26,03 [24,83; 27,23] | 24,66 [22,46; 26,92] | 23,41 [22,81; 24,00] |
| t12_causal_bloque_seed1 | fmd5oakq | 9751 | 24,05 | 26,78 [25,69; 27,84] | 27,13 [25,89; 28,34] | 25,34 [23,14; 27,66] | 24,04 [23,44; 24,63] |
| t12_causal_bloque_seed2 | g4ljte2k | 9801 | 24,27 | 27,05 [25,95; 28,15] | 27,38 [26,11; 28,63] | 25,68 [23,41; 28,06] | 24,28 [23,67; 24,87] |
| t12_causal_train_fija_seed0 | v5iqxuzd | 9801 | 31,43 | 37,62 [36,21; 39,04] | 36,70 [35,15; 38,24] | 41,37 [37,84; 44,92] | 31,43 [30,62; 32,25] |
| t12_causal_train_fija_seed1 | cz8jubsh | 9801 | 32,36 | 39,83 [38,41; 41,26] | 38,74 [37,17; 40,30] | 44,27 [40,86; 47,77] | 32,36 [31,57; 33,15] |
| t12_causal_train_fija_seed2 | 877drtr6 | 9801 | 32,47 | 39,90 [38,48; 41,33] | 38,93 [37,37; 40,44] | 43,85 [40,34; 47,33] | 32,48 [31,66; 33,30] |
