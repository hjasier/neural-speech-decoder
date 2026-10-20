# Alice EEG con GRU + CTC en CUDA T4

Perfil ampliado solicitado por el autor el 20 de octubre de 2026. Reutiliza
nuestro `SpeechGRU` y el entrenamiento CTC de T12, con pesos nuevos y entrada
EEG. El notebook es un lanzador de `run.py`; el modelo y el bucle están en
`src/`. El perfil anterior de un participante en CPU se conserva como
referencia exploratoria, con sus métricas y checkpoint propios.

## Ejecutar en Colab

1. Abrir la versión actualizada de `notebooks/05_alice_colab.ipynb` y conectar
   una VM con **GPU T4**, también si se utiliza VS Code.
2. Ejecutar las celdas en orden. Se comprueban GPU y versiones; se reutiliza
   el entorno T12 compatible o se instala Python 3.12.11 y Torch 2.5.1 CUDA 12.4
   en un entorno aislado. CUDA es obligatorio: no hay cambio automático a CPU.
3. Montar Drive, descargar y verificar las fuentes, preparar EEG/fonemas y
   proporcionar la clave W&B en la entrada oculta. Proyecto: `tiiac-2026`.
4. Ejecutar la celda de entrenamiento. Al terminar, restaura el mejor
   checkpoint, evalúa test y controles, verifica la recarga en otro proceso
   sin actualizar pesos y genera la figura.

Solo hay que subir el notebook: incluye un snapshot verificable del código.
Se actualiza desde el repositorio con
`python scripts/export_colab.py --perfil alice`. El notebook 04 conserva
sus salidas y snapshot T12.

## Configuración

Todos los parámetros están en `configs/alice_ctc_t4.yaml`.

| Elemento | Perfil ampliado |
|---|---|
| Participantes | 27, con adaptador de entrada por persona y GRU compartida |
| Entrada | 60 canales EEG buenos, referencia media, banda 0,5–20 Hz, 64 Hz |
| Arquitectura | Mismo `SpeechGRU`: proyección 60, Softsign, kernel 8/stride 2, cinco GRU unidireccionales de 512 |
| Parámetros | 7.950.893 |
| Salida | 40 símbolos incluyendo SIL; blank CTC 40 |
| Lotes | 64 fragmentos de un participante elegido uniformemente; acumulación en microlotes de 16 |
| Optimizador | Adam-TF, LR lineal 0,02→0 en 10.000 pasos, L2 y clipping heredados |
| Presupuesto | Hasta 10.001 actualizaciones; límite suave de 3.600 s de bucle |
| Validación | Completa tras el primer paso, cada 250 y al terminar; selección por CTC |
| Evaluación | Microlotes 16, test completo con el mejor checkpoint restaurado |
| Precisión | Float32, sin AMP |

El tiempo incluye validaciones y guardados del bucle. La parada se comprueba
entre pasos; descarga, preparación, evaluación final, controles, recarga y
sincronización añaden tiempo. **No se ha medido aún el entrenamiento en T4**:
no se garantiza completar los 10.001 pasos en una hora ni una mejora del PER.
Este es el perfil ampliado; no acredita el límite docente de 20 minutos.
El checkpoint permite reevaluar; aún no reanuda el optimizador.

## Selección y protocolo de datos

Fuente: [NEMAR nm000180 v1.1.3](https://www.nemar.org/dataset/nm000180),
licencia CC-BY-4.0. El catálogo fija versiones, tamaño y hash oficial de
**168 archivos, 2.465.566.644 bytes (2,47 GB)**. Son EEG de escucha pasiva
del mismo capítulo; no habla intentada ni producción de voz.

De los 31 participantes disponibles marcados `analyzed=yes`, se excluyen
`sub-026`, `sub-034`, `sub-035` y `sub-036`: falta el segmento 12 y hay
marcas temporales duplicadas entre segmentos. Se conserva el detalle de
los eventos en `docs/datasets/alice_eeg/auditoria_t4.json`.

Los 27 restantes tienen doce segmentos, eventos estrictamente crecientes,
palabras verificadas contra el CSV original y los mismos 60 canales EEG
buenos en el mismo orden. Se excluyen VEOG y canales auxiliares, incluidos
AUD/Aux5; el modelo solo recibe EEG. Los binarios se verifican antes de
preparar; las matrices preparadas también tienen SHA-256.

- **Train 1–8, val 9–10 y test 11–12 en todos los participantes.** Ninguna
  instancia de palabra anotada del test entra en train a través de otra
  persona. Esto mide participantes conocidos y tramos de relato reservados;
  no generalización a personas nuevas.
- Filtrado causal y remuestreo por segmento. En este perfil los límites
  van desde la primera palabra hasta el final de la última palabra anotada;
  se descartan fragmentos que invaden márgenes de 2 s. Este cambio recorta
  algunos fragmentos respecto al perfil CPU histórico.
- Media/desviación exclusivamente con train **de cada participante**.
  No se ajusta nada usando val/test ni se modifica `src/data/splits.py`.
- Fragmentos de hasta ocho palabras, sin cruzar segmento, frase o huecos
  de anotación. Las palabras delimitan el EEG y su referencia; el modelo
  no recibe tiempos de fonemas. Mismo g2p-en y convenciones SIL de T12.

La auditoría de metadatos y referencias prevé **5.022/1.188/918 fragmentos**
de train/val/test: 186/44/34 por participante. El preparador guarda los
recuentos reales. CTC exige suficiente salida temporal, incluyendo las
separaciones entre símbolos iguales; no se truncan objetivos incompatibles.

## Resultados que se guardan

Drive contiene un directorio nuevo `speech_bci/experimentos/alice_t4_…/`
con snapshot, configuración efectiva, entorno CUDA, logs, tracking y runs.
Los originales y señales preparadas permanecen en el disco de la VM, sin
leerlos por Drive durante el entrenamiento.

Cada run guarda `mejor_val.pt`, `resultado.json` y `predicciones_test.json`.
Se registran pasos, tiempo, hardware, picos CUDA asignados/reservados,
historial de validación, PER agregado, macro y por participante. Los controles
usan EEG cero, un prior de fonemas ajustado solo con train y secuencia vacía.
El diagnóstico posterior sin SIL acompaña al PER principal incluyendo SIL;
no se utiliza para seleccionar pesos. No hay WER ni modelo de lenguaje.

## Comprobado antes de Colab

Se verificaron los hashes de canales, cabeceras y eventos de los 27 elegidos,
su correspondencia con el texto, límites, separación por segmentos y
longitudes CTC. Se prepararon los EEG reales de **sub-001 y sub-003**:
372/88/68 fragmentos. Con ese lector, un lote con ambas personas y el modelo
completo de 27 adaptadores produjo logits `[2, 80, 41]` y CTC finita en CPU
con dos hilos, **sin actualizaciones de pesos**. Se comprobó el muestreador
de lotes de 64 de un único participante. Evidencia:
`docs/resultados/alice_ctc_t4_validacion_local.json`.

Los otros 25 EEG se descargarán y verificarán en Colab. No se ha ejecutado
el notebook en una VM remota ni medido aún memoria, duración o calidad del
perfil CUDA. El resultado CPU anterior no es el resultado de este perfil.
