# Primera ejecución acortada de habla en T4

20 de octubre de 2026. Run [unjiabqs](https://wandb.ai/aaurre-deusto/tiiac-2026/runs/unjiabqs), `speech_2h_colab`,
perfil `configs/speech_2h.yaml`. Datos completos en
[`speech_2h_unjiabqs.json`](speech_2h_unjiabqs.json).

## Resultado observado

| Medida | Valor |
|---|---|
| Hardware | Tesla T4; PyTorch 2.5.1+cu124; CUDA 12.4 |
| Pasos completados | 2.001 de 2.001, sin parada por tiempo |
| PER del test publicado | **32.22 %**: 6.129 errores / 19.024 fonemas |
| CTC del test | 31.681987, media por frase |
| Test recorrido | 680 de 680 frases |
| Bucle, evaluaciones y guardados | 2702.8273 s, aproximadamente 45 min 03 s |
| Actualizaciones, sin evaluaciones periódicas | 2598.1002 s, aproximadamente 43 min 18 s |
| Evaluación final | 5.6600 s |
| Pico asignado / reservado | 1994.10 / 3010 MiB |

Tiempos del resumen W&B; el bucle excluye instalación, preparación de datos
y parte de la apertura/cierre del proceso. Las cifras proceden de las salidas
del notebook guardadas por el autor. No se ha repetido el entrenamiento ni
se ha abierto el checkpoint remoto desde este equipo.

## Configuración y alcance

Cinco GRU unidireccionales de 1.024 unidades; 19 sesiones de T12, 6.640 frases
de train y 680 de test. Normalización ajustada solo con train, semilla 0,
lote efectivo 64 y microlotes 16 en train y test.

Frente al perfil completo, se reducen los pasos de 10.001 a 2.001 y el
calendario lineal de learning rate de 10.000 a 2.000 pasos, de 0,02 a 0.
Se evalúa cada 250 pasos, además de al principio y al final. El presupuesto
suave de 5.400 s no se alcanzó. Se conserva el último checkpoint, sin elegir
el mejor por PER de test.

PER mide ediciones de fonemas e incluye SIL. WER y la decodificación de
palabras mediante un modelo de lenguaje quedan pendientes. Se monitoriza
el test publicado; no hay validación independiente. Es una ejecución
acortada, no el entrenamiento completo del original. **45 minutos superan
el límite docente de 20**: todavía falta medir una configuración que lo cumpla.

## Evolución registrada

Valores redondeados como en consola. Los pasos se obtienen de la cadencia
del bucle verificado en el snapshot. Las métricas finales precisas están en
el JSON.

| Paso | PER | CTC |
|---|---|---|
| 1 | 90.11 % | 192.7532 |
| 251 | 65.39 % | 67.8787 |
| 501 | 49.30 % | 51.5660 |
| 751 | 42.44 % | 43.1930 |
| 1001 | 38.78 % | 38.5757 |
| 1251 | 36.28 % | 36.2293 |
| 1501 | 33.99 % | 33.4561 |
| 1751 | 32.62 % | 32.1373 |
| 2001 | 32.22 % | 31.6820 |

## Procedencia y archivos

- Notebook con las salidas: `notebooks/04_entrenamiento_colab.ipynb`.
- SHA-256 del paquete: `e9e9abefd71b6c290671f06be2508070932deb9837d264071d505b95cde72530`.
- Commit base declarado: `40101370a0a32ac3c49f2bc2508c107b23530593`;
  los hashes de cada fuente fijan además su contenido exacto.
- Experimento, relativo a Mi unidad: `speech_bci/experimentos/20261020_122153_5effafcb`.
- Pesos, relativos a Mi unidad: `speech_bci/checkpoints/speech_original/unjiabqs/ultimo.pt`.
- `ultimo.json` conserva paso y métricas; `ultimo.pt` incluye pesos,
  configuración, normalización y procedencia de los datos.

El log indica sincronización de W&B. Los pesos se conservan en Drive;
la sincronización no registra artefactos de pesos enviados a W&B.
La celda 10 consultó el resumen; **recargar y reevaluar el checkpoint sigue
pendiente**.

## Siguiente comprobación en Colab

En la misma sesión preparada, tras la celda 10 que define `selected`:

```python
ejecutar(["run.py", "--config", CONFIG, "--solo-evaluar", "--checkpoint", str(selected)])
```

Carga los pesos y recorre el test sin entrenar. Contrastar PER, 6.129 errores,
19.024 fonemas y CTC aproximada 31,681987. Abre otro run de W&B para registrar
esa evaluación. No continúa el optimizador ni sustituye los pesos del run
original. Registrar la salida antes de marcar la recarga como verificada.
