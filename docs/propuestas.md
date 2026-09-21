# Ficha de propuestas de proyecto · Hito H1

**Entrega: domingo 20 de septiembre, 23:59, en la Tarea «H1 · Propuestas de proyecto».**
**Decisión: domingo 27 de septiembre (hito H2).** Sin visto bueno no se sigue adelante.

Hasta **tres propuestas, en orden de preferencia**. La primera tiene que estar completa; la segunda y
la tercera son vuestras alternativas si la primera no pasa la puerta. Las tres tienen las mismas cuatro
comprobaciones.

---

## Cómo se entrega

1. Copiad esta página entera a un fichero `docs/propuestas.md` de vuestro repositorio y rellenadla ahí.
   Es Markdown: las tablas se editan en cualquier editor, y así queda en la historia de _commits_.
2. Marcad las casillas cambiando `[ ]` por `[x]` **solo cuando sea verdad**. Una casilla marcada sin
   evidencia en la tabla de debajo cuenta como no marcada.
3. Haced _commit_ y _push_, y en la Tarea de ALUD pegad **la URL del fichero en GitHub**. Entrega uno,
   cuenta por los dos.
4. El 27 de septiembre tendréis la decisión de cada propuesta en la retroalimentación de la Tarea. La
   aprobada se convierte en vuestro `docs/viabilidad.md`, que es el que se corrige con E1.

---

## Las dos vías

**Vía A · lista curada.** Uno de los doce papers ya comprobados, que están en el capítulo _Lista de
papers_ de este Libro: ResNet, U-Net, YOLO (v1), Vision Transformer, CycleGAN, Grad-CAM, SimCLR,
ejemplos adversarios (FGSM), Word2Vec, Transformer, BERT (solo preentrenamiento) y Neural
Collaborative Filtering. Sabemos que caben en la regla 5/20 y que tienen un aplicativo natural encima.
La ficha se rellena igual, pero es rápido. **Máximo dos parejas por paper**, por orden de entrega.

**Vía B · propuesta propia.** Cualquier paper o caso industrial que os interese, si la ficha pasa las
cuatro comprobaciones. Da más trabajo al principio, y ese trabajo de acotar es justo lo que venís a
aprender aquí.

---

## Datos del proyecto

|                                     |                                                                       |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Nombres**                         | Asier Aurre                                     |
| **Repositorio**                     | https://github.com/hjasier/tic_proyecto                           |
| **Rol profesional al que apuntáis** | Aún no lo tengo decidido; me interesan distintos sectores y quiero explorar la intersección entre IA y neurotecnología. |

---

## Propuesta 1

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | *High-performance brain-to-text communication via handwriting*       |
| **Autores y año**          | Francis R. Willett, Donald T. Avansino, Leigh R. Hochberg, Jaimie M. Henderson y Krishna V. Shenoy (2021) |
| **Enlace**                 | https://doi.org/10.1038/s41586-021-03506-2                             |
| **Vía**                    | B (propuesta propia)                                                     |
| **Por qué este y no otro** | Últimamente estoy bastante interesado en la neurotecnología, interfaces cerebro-computador y sistemas de IA para comunicación asistida. Esta propuesta tiene datos reales, código de referencia y un aplicativo chulo. |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace: https://doi.org/10.5061/dryad.wh70rxwmv
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | Data from: *High-performance brain-to-text communication via handwriting* |
| **Enlace de descarga**                | https://doi.org/10.5061/dryad.wh70rxwmv |
| **Tamaño**                            | 1,41 GB; 1.000 frases, 43.501 caracteres y 10,7 horas de actividad neural; dos matrices de 96 electrodos. |
| **Licencia**                          | CC0 1.0, según los metadatos del dataset de Dryad. |
| **¿Hace falta registro o solicitud?** | No se solicita acceso académico, hay descarga pública desde Dryad. |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | Sí: https://github.com/fwillett/handwritingBCI |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | No aplica: sí hay código de referencia. El RNN, el etiquetado temporal y la evaluación se describen en el paper y el repositorio. |
| **Qué NO especifica el paper**                           | El código y los notebooks cubren gran parte de la implementación: etiquetado con HMM, entrenamiento e inferencia de la RNN, modelos de lenguaje y evaluación. Por eso no tendría que reconstruir la arquitectura desde cero. Lo que no queda cerrado para mi réplica es el coste real en un entorno actual: el repositorio está basado en TensorFlow 1.15, necesita GPU para las GRU y la ampliación sintética de datos puede generar unos 100 GB. No se cuanto puede llegar a tardar el entrenamiento en una T4 ni una receta para ejecutar exactamente mi recorte de 200 frases. Tendré que dejar fijados el entorno, los pasos que ejecuto, los intermedios que reutilizo y el efecto de quitar GPT-2. |

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | Dataset completo: 1.000 frases, 43.501 caracteres, 10,7 horas y 192 electrodos. El pipeline incluye etiquetado HMM, RNN y modelos de lenguaje; el artículo no da un tiempo directamente comparable con una T4. |
| **Escala que voy a hacer**                    | Piloto con 200 frases y datos preprocesados; RNN reducido y 10 épocas equivalentes. Elimino GPT-2 y me centro en el decodificador principal. |
| **Tiempo estimado del entrenamiento completo** | Objetivo: menos de 15 minutos en una T4 de Colab. Aún no medido; lo cronometraré sobre una copia limpia con el subset y la configuración fijados. |
| **Qué se pierde al recortar**                  | Tendré menos datos, menor capacidad de generalización entre bloques temporales y previsiblemente un CER/WER peor. No reproduciré la corrección completa con GPT-2. |
| **Hardware que voy a usar**                   | Google Colab gratuito con GPU NVIDIA T4; también haré pruebas ligeras en mi RTX 2060 y, si tengo acceso, en un equipo con RTX 4090. |

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | Aplicación de demostración , crearía una web en la que se carga una muestra aleatoria, creo un visor del input neural y muestro la transcripción progresiva generada por el modelo y comparación con el texto real. |
| **Quién lo usaría y para qué** | Investigadores o estudiantes que quieran probar un decodificador cerebro-texto sin disponer de un implante; sirve para generar interes por el sector. |
| **Qué necesita del modelo**    | Entrada: características neurales preprocesadas por ventana. Salida: probabilidades de caracteres y texto decodificado. Latencia objetivo: inferior a un segundo por ventana; se medirá en la aplicación. |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | Las dependencias antiguas de TensorFlow/CUDA y el pipeline completo pueden no ser reproducibles dentro del presupuesto; también puede haber errores al convertir los datos de Dryad. |
| **Qué haría si pasa**                            | Usaré los intermedios públicos del dataset, reimplementaré únicamente el RNN en PyTorch moderno, reduciré el subset y eliminaré el modelo de lenguaje. Si aún no cabe, pasaré a una baseline lineal y lo declararé como recorte. |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 2 (alternativa)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding* |
| **Autores y año**          | Thomas Müller, Alex Evans, Christoph Schied y Alexander Keller (2022) |
| **Enlace**                 | https://arxiv.org/abs/2201.05989<br>Guía práctica: https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/ |
| **Vía**                    | B (propuesta propia) |
| **Por qué este y no otro** | Al principio pensé en replicar el NeRF clásico, pero sus propios autores indican que entrenarlo puede tardar varias horas o incluso uno o dos días. Con los límites 5/20 no creo que una réplica clásica saliera bien. Instant-NGP mantiene la idea de NeRF, pero usa una codificación hash multirresolución y CUDA, así que es mucho más viable. Además, me permitiría trabajar en representación 3D, CUDA y visión por computador con un aplicativo visual muy claro. Aunque ya he trabajado en otros proyectos más similares a este. |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Usaría la escena de ejemplo `data/nerf/fox` del repositorio oficial.
- [x] La licencia permite el uso académico para investigación y evaluación no comercial, según la licencia del repositorio oficial.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | Escena NeRF de ejemplo `fox` incluida en el repositorio oficial de Instant-NGP. |
| **Enlace de descarga**                | https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox |
| **Tamaño**                            | Una escena con 50 imágenes y `transforms.json`; mediré el tamaño exacto al descargarla. |
| **Licencia**                          | NVIDIA Source Code License-NC: uso no comercial para investigación o evaluación; conservaré la licencia en el repositorio. |
| **¿Hace falta registro o solicitud?** | No; el repositorio y la escena son públicos y descargables directamente. |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | Sí: https://github.com/NVlabs/instant-ngp |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | No aplica: hay código de referencia y el paper describe la codificación hash, el MLP y el renderizado volumétrico. |
| **Qué NO especifica el paper**                           | El paper y el repositorio dejan bastante bien definida la implementación central: codificación hash, MLP, renderizado volumétrico, configuraciones y escenas de ejemplo. El problema no es que falte la arquitectura, sino que los tiempos dependen mucho del hardware y de la configuración. El paper habla de entrenamientos de pocos segundos en la GPU de los autores, pero no indica cuánto tardará una T4 con una sola escena, menos pasos y una tabla hash o un MLP reducidos. Tampoco fija nuestro entorno exacto de CUDA y dependencias ni el visor web que construiré. Documentaré el commit, el entorno, la configuración, los pasos, el tiempo y las métricas. |

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**. 

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | Evalúa varias primitivas y escenas; para NeRF usa la codificación hash multirresolución y entrenamiento acelerado con CUDA. Los autores reportan entrenamientos de pocos segundos en su hardware, una RTX 3090. |
| **Escala que voy a hacer**                    | Una sola escena (`fox`), resolución reducida y entre 10.000 y 20.000 pasos; usaré una configuración pequeña de la tabla hash y del MLP. |
| **Tiempo estimado del entrenamiento completo** | Objetivo: menos de 10 minutos en una T4 de Colab. |
| **Qué se pierde al recortar**                  | Solo estudiaré una escena y no reproduciré todas las tareas del paper. Una tabla hash y una resolución menores pueden reducir el detalle y la calidad de las vistas nuevas. |
| **Hardware que voy a usar**                   |  Google Colab gratuito con GPU NVIDIA T4; también haré pruebas ligeras en mi RTX 2060 y, si tengo acceso, en un equipo con RTX 4090 |

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construyo encima**       | Aplicación nativa que permite sacar fotos de un obtejo , se suben a un servidor , este entrena el modelo y lo devuelve a la apliacción como entorno 3d manipulable |
| **Quién lo usaría y para qué** | Estudiantes o investigadores que quieran inspeccionar una reconstrucción 3D sin instalar el entorno de entrenamiento. |
| **Qué necesita del modelo**    | Entrada: imágenes . Salida: imagen renderizada desde una cámara nueva. La inferencia debe ser interactiva después del entrenamiento. |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | La compilación de CUDA/C++ o la compatibilidad entre la GPU, CUDA y tiny-cuda-nn; también puede fallar el ajuste de las poses|
| **Qué haría si pasa**                            | Usaré la versión precompilada si es posible, reduciré la resolución y entrenaré solo la escena de ejemplo. Si la compilación sigue bloqueando el proyecto, mantendré esta propuesta como alternativa y no la convertiré en la principal. |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 3 (alternativa)

> Tercera opción: es la alternativa más segura dentro de la temática BCI.

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | *EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces* |
| **Autores y año**          | Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung y Brent J. Lance (2018) |
| **Enlace**                 | https://arxiv.org/abs/1611.08024 |
| **Vía**                    | B (propuesta propia) |
| **Por qué este y no otro** | Es una alternativa más sencilla y cercana a las interfaces cerebro-computador. El modelo es compacto, los datos son públicos y el aplicativo se puede probar reproduciendo EEG grabado e incluso igual se podría probar en directo con el casco EEG que disponen en el lab de BioTech. |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace: https://bnci-horizon-2020.eu/database/data-sets
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | BNCI 2014-001 / BCI Competition IV 2a: imaginación motora. |
| **Enlace de descarga**                | https://bnci-horizon-2020.eu/database/data-sets |
| **Tamaño**                            | 9 sujetos, 22 canales EEG + 3 EOG, 4 clases, 2 sesiones; aproximadamente 744 MB en la distribución BNCI. |
| **Licencia**                          | Creative Commons Attribution-NoDerivatives 4.0 (CC BY-ND 4.0). |
| **¿Hace falta registro o solicitud?** | No; el dataset es de acceso abierto y se descarga desde la página de BNCI. |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | Sí: https://github.com/vlawhern/arl-eegmodels |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | No aplica: hay código de los autores y el paper describe las convoluciones temporal, espacial y separable. |
| **Qué NO especifica el paper**                           | El paper y el repositorio dejan bastante definida la parte del modelo: arquitectura EEGNet, convoluciones, Adam, función de pérdida, número de épocas y parada por validación. No tendría que inventar la red. Lo que no ofrecen es una receta completamente cerrada para mi ejecución actual: el paper usa TensorFlow/Keras con una Quadro M6000 y CUDA 9, mientras que el entorno de la asignatura será más moderno; además, no describe las semillas necesarias para repetir literalmente sus particiones aleatorias. El aplicativo y el recorte del análisis a una ejecución manejable sobre BNCI son decisiones mías. Documentaré las versiones, el preprocesado, la partición por sujetos, los hiperparámetros y los tiempos. |

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | Evalúa EEGNet en cuatro paradigmas BCI: P300, ERN, MRCP y ritmos sensoriomotores, con varios datasets y configuraciones. |
| **Escala que voy a hacer**                    | Un solo dataset, BNCI 2014-001, con las 4 clases de imaginación motora; EEGNet compacto y entre 50 y 100 épocas, con división por sujeto. |
| **Tiempo estimado del entrenamiento completo** | Objetivo: menos de 5 minutos en una T4 de Colab. Aún no medido; lo comprobaré con una ejecución reproducible y guardaré la configuración. |
| **Qué se pierde al recortar**                  | No estudiaré los otros tres paradigmas del paper ni podré validar el sistema con EEG en directo. La generalización entre sujetos puede ser peor que en los resultados originales. |
| **Hardware que voy a usar**                   | Google Colab con T4; pruebas ligeras en local. |

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construyo encima**       | Panel web que reproduce segmentos EEG y traduce las cuatro clases de imaginación motora a comandos visuales, mostrando la probabilidad de cada clase. |
| **Quién lo usaría y para qué** | Estudiantes o investigadores que quieran probar un clasificador BCI sin disponer de un casco EEG. |
| **Qué necesita del modelo**    | Entrada: ventana multicanal de EEG. Salida: una de cuatro clases y sus probabilidades. La predicción debe tardar menos que la ventana reproducida. |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | Fugas entre ventanas del mismo sujeto, diferencias fuertes entre sujetos y sensibilidad a la normalización o al filtrado de EEG. |
| **Qué haría si pasa**                            | Separaré los sujetos correctamente, reduciré el problema a dos clases y usaré una baseline clásica como CSP + LDA para comprobar que el fallo no está en los datos. |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |
