# Alice EEG: piloto exploratorio y auditoría pendiente

**Exploración histórica.** El autor aclaró después que quiere adaptar nuestro
decodificador GRU + CTC a Alice. La línea activa está en `alice_ctc.md`.
Este objetivo de regresión lo había propuesto el asistente. Se ejecutó una comparación técnica sobre sub-001.
La referencia `sound_power` presenta una anomalía que debe aclararse antes
de interpretar las métricas como recuperación de información acústica.

## Qué se ha ejecutado

Entrada: ventanas de 100–600 ms después del inicio publicado de cada palabra,
con baseline de −200–0 ms, 60 canales EEG, referencia media, banda 0,5–20 Hz
y remuestreo offline a 64 Hz. Se excluyen VEOG y Aux5; no entran audio, texto,
posición de la palabra ni etiquetas lingüísticas al modelo. Los onsets publicados
se utilizan para extraer las ventanas: no es un decodificador autónomo.

Objetivo provisional: `log10(sound_power)` de los eventos oficiales. Se comparan
media de train, ridge sobre cuatro promedios temporales y GRU de dos capas
de 32 unidades (14.657 parámetros), entrenada desde cero en PyTorch. Es una
adaptación propia de la familia recurrente; no reproduce el experimento del
paper de Alice ni transfiere los pesos T12.

Separación por segmentos del audiolibro: 1–8 train, 9–10 validación, 11–12 test.
Los segmentos se separan antes del filtrado; se descartan márgenes de 2 s.
Normalización ajustada solo en train. Ridge y época GRU se eligen por MSE de
validación; el código calcula test al terminar. Quedan 1.380/324/272 ventanas.
El test se ha inspeccionado después para auditar el objetivo; este experimento
es de desarrollo y no proporciona una confirmación independiente.

## Mediciones del run 1ujf6ldx

| Modelo | Pearson en test | R² en test | MSE normalizado |
|---|---:|---:|---:|
| Media de train | No definido: constante | −0,01794 | 1,05141 |
| Ridge | 0,10095 | −0,00840 | 1,04156 |
| GRU | 0,36657 | 0,12966 | 0,89896 |

Son medidas frente a **esa columna provisional**. No equivalen a exactitud,
reconstrucción de voz ni generalización entre personas. Los 199 desplazamientos
de etiquetas por segmento son controles exploratorios; no una conclusión sobre
decodificación cerebral. No se han realizado ICA ni rechazo manual de artefactos.

CPU, dos hilos: 8,16 s de bucle GRU con validación; 0,0174 s de ajustes ridge;
10,77 s desde preparación hasta construir el informe, sin descarga, arranque de
W&B, figuras ni guardados finales. Completó 15 épocas y restauró la época 9.
Este tiempo corresponde a Alice; T12 sigue teniendo un run de unos 45 minutos.

La recarga de `gru.pt` y `ridge.npz`, restaurando normalización, reproduce las
métricas de las 272 ventanas; error máximo frente al CSV en log10 de 4,54e−7
para GRU y cero para ridge. No se volvió a entrenar para comprobarlo.

## Por qué se ha puesto el resultado en revisión

La figura muestra largos tramos casi constantes de `sound_power` al inicio
de cada segmento. En los primeros 20 s de los doce segmentos, sus medianas
son aproximadamente 3,8e−9; en el resto pasan a aproximadamente 0,0012–0,0031.
Es una observación posterior al entrenamiento, no un criterio de selección.

Se ha comprobado que `build_events.py` copia `SndPower` del CSV original,
unido por `Order`; no se ha determinado la causa de ese patrón. En el WAV
oficial del segmento 1 hay señal en sus primeros segundos. El CSV original
aporta onsets y offsets dentro de cada audio; el desplazamiento entre sus
onsets y los EEG varía menos de 2 ms dentro de cada segmento. Queda por validar
la definición/alineación acústica de `SndPower`, o definir explícitamente un
objetivo nuevo calculado desde el audio. **No se ha implementado ni entrenado
esa alternativa.** No se deben escoger nuevos parámetros mirando este test.

## Reproducir únicamente la comprobación técnica

Desde la raíz, con el entorno del proyecto instalado:

```bash
python scripts/get_alice.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python run.py --config configs/alice_cpu.yaml
python scripts/evaluate_alice.py --run-dir runs/alice_power/ID_RUN
python scripts/report_alice.py --run-dir runs/alice_power/ID_RUN
```

Se descargan once archivos oficiales de NEMAR v1.1.3, 91.219.921 bytes,
verificados por tamaño y hash SHA-256 o git-blob según el manifiesto.
La configuración CPU registra W&B offline en el proyecto `tiiac-2026`.
La sincronización posterior usa la entidad de la cuenta autenticada.
Datos, pesos, predicciones y registro W&B permanecen ignorados por Git.

El notebook 05 se ha orientado a la adaptación CTC solicitada por el autor;
no ejecuta este piloto anterior. Las fuentes y comandos anteriores permiten
conservarlo para auditoría. El notebook 04 con las salidas T12 se conserva.

## Fuentes

- [NEMAR nm000180, v1.1.3](https://www.nemar.org/dataset/nm000180), licencia CC-BY-4.0.
- [Brennan y Hale, 2019](https://doi.org/10.1371/journal.pone.0207741).
- [Conversión de eventos](https://github.com/nemarDatasets/nm000180/blob/v1.1.3/code/build_events.py).
- [CSV original](https://github.com/nemarDatasets/nm000180/blob/v1.1.3/sourcedata/AliceChapterOne-EEG.csv).
- Catálogo fijado y hashes: `docs/datasets/alice_eeg/catalogo.json`.
- Mediciones y procedencia: `docs/resultados/alice_1ujf6ldx.json`,
  `alice_1ujf6ldx_recarga.json` y `alice_1ujf6ldx_auditoria.json`.

La interpretación científica y la discusión reflexiva de la entrega quedan
a cargo del autor; este documento registra operaciones y mediciones.
