# Prueba de tiempo del entrenamiento

Quiero tener una idea de cuánto tarda entrenar antes de decidir cuánto recortar.
Esta prueba mide el coste de los pasos de entrenamiento. Todavía no sirve para saber si el modelo aprende bien.

## Qué he probado

La prueba se ejecutó el 11/10/2026 en una RTX 2060. Se usaron 48 frases de `t12.2022.04.28.mat`, solo de train, con semilla 42.
Las frases de esta muestra duran entre 4,54 y 17,88 segundos; la mediana es 12,35 segundos.

La entrada combina `spikePow` y `tx1` de 6v: 256 características por intervalo de 20 ms. Se normaliza con esa sesión de train y se aplica suavizado.
El texto se convierte a fonemas con [CMUDict](https://github.com/cmusphinx/cmudict). Se dejaron fuera 4 frases con palabras que no estaban en el diccionario.

Se probaron cuatro GRU con CTC, en FP32. Cada una hizo 2 pasos de calentamiento y 12 pasos medidos, con cálculo de pérdida, gradientes y actualización de pesos.
El tiempo por paso incluye preparar el lote, pasarlo a la GPU y registrar las medidas en W&B offline.

| Capas × unidades | Lote | Parámetros | Tiempo medio por paso | Memoria reservada por PyTorch |
|---|---:|---:|---:|---:|
| 2 × 128 | 8 | 4.879.017 | 28,4 ms | 998 MiB |
| 3 × 256 | 8 | 8.868.649 | 59,0 ms | 1.096 MiB |
| 5 × 512 | 8 | 21.276.201 | 130,4 ms | 1.748 MiB |
| 5 × 1024 | 4 | 55.129.129 | 180,3 ms | 1.792 MiB |

Las cuatro pruebas terminaron sin errores. El proceso tenía un límite de 3 GiB de RAM, sin swap, cuota de dos CPU y 75 segundos de duración.
PyTorch podía reservar como máximo el 35 % de la memoria de GPU disponible para CUDA. El contexto CUDA y el escritorio usan memoria adicional.

## Qué tiempo sale al ampliar la prueba

La cuenta es: `ceil(frases / lote) × épocas × tiempo medio por paso × 1,25`.
Ese 25 % es un margen. No he ejecutado todavía estos entrenamientos completos.

| Capas × unidades | 500 frases, 20 épocas | 2.000 frases, 20 épocas | 8.800 frases, 20 épocas |
|---|---:|---:|---:|
| 2 × 128 | 0,75 min | 2,96 min | 13,02 min |
| 3 × 256 | 1,55 min | 6,14 min | 27,02 min |
| 5 × 512 | 3,42 min | 13,58 min | 59,75 min |
| 5 × 1024 | 9,39 min | 37,56 min | 165,25 min |

El recorte que quiero comprobar es **mantener 5 capas, usar 512 unidades por capa y 2.000 frases de train**, con lote de 8.
Con 20 épocas como ejemplo, la estimación es de unos 14 minutos de entrenamiento en esta GPU.
Mantener 5 capas con 1.024 unidades sería bastante más lento: unos 38 minutos para las mismas 2.000 frases.

Esto aún no demuestra que cumpla los 20 minutos de clase. Falta medir la preparación completa, la validación y el guardado, y repetirlo en una T4.
Tampoco sabemos si 20 épocas serán suficientes: hay que entrenar y medir PER. La prueba de menos de 5 minutos en CPU también sigue pendiente.
Solo se ha cronometrado una sesión, con frases largas, y la GRU del piloto no es una réplica exacta del código de los autores.

## Dónde están el código y las medidas

- Código: [scripts/medir_tiempo_gru.py](../scripts/medir_tiempo_gru.py).
- Configuración para repetirlo: [configs/piloto_tiempo_gru.yaml](../configs/piloto_tiempo_gru.yaml).
- Medidas originales de cada uno de los 12 pasos: [carpeta de resultados](pruebas/tiempo_gru_2026-10-11/).
- En esa carpeta también están los índices de las 48 frases, la configuración original y el entorno.

Los datos muestreados, el diccionario y los registros completos de W&B se conservan localmente en `runs/piloto_tiempo_gru/`, que está ignorado por Git.
Las medidas de Git son las de la ejecución original. El script se ha adaptado para usar rutas del repo y poder repetir la cuenta.

## Cómo repetir la prueba

Desde la carpeta del repo, con las dependencias de `requirements.txt` y `competitionData/train` descargado:

```bash
# Prepara las 48 frases; descarga y verifica el diccionario si falta.
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 -p CPUQuota=200% \
  timeout -k 5s 75s env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python scripts/medir_tiempo_gru.py --config configs/piloto_tiempo_gru.yaml --mode prepare

# Cronometra la GRU de 5 capas y 512 unidades.
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 -p CPUQuota=200% \
  timeout -k 5s 75s env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  WANDB_SILENT=true WANDB_CONSOLE=off CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python scripts/medir_tiempo_gru.py --config configs/piloto_tiempo_gru.yaml --mode benchmark --variant grande

# Recalcula las estimaciones originales sin entrenar ni usar la GPU.
.venv/bin/python scripts/medir_tiempo_gru.py --config configs/piloto_tiempo_gru.yaml \
  --mode summary --results-dir docs/pruebas/tiempo_gru_2026-10-11
```

Las otras variantes son `pequena`, `mediana` y `referencia`. Estos comandos con límites usan systemd en Linux; no son instrucciones para Colab.
