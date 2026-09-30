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
| **Rol profesional al que apuntáis** | ML Engenieer |

---

## Propuesta 1

### Identificación

| | |
|---|---|
| **Paper o referencia** | *A high-performance speech neuroprosthesis* |
| **Autores y año** | Francis R. Willett y colaboradores (2023). |
| **Enlace** | https://doi.org/10.1038/s41586-023-06377-x |
| **Vía** | B (propuesta propia). |
| **Por qué este y no otro** | Al principio había elegido el trabajo sobre escritura imaginada. Pero luego he visto esta alternativa que es muy parecida y me ha parecido más interesante aún. Me interesa cómo se puede aprovechar la actividad neuronal para recuperar una forma de comunicación, y aquí el objetivo es obtener texto a partir de los intentos de hablar, en vez de los intentos de escribir. |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace: https://doi.org/10.5061/dryad.x69p8czpq
- [x] La licencia permite el uso académico.

| | |
|---|---|
| **Dataset** | Data for: *A high-performance speech neuroprosthesis*. Empezaré con `competitionData`, las grabaciones de T12 preparadas para entrenar y evaluar decodificadores. Usaré la versión de septiembre de 2023. |
| **Enlace de descarga** | https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq |
| **Tamaño** | `competitionData`: 3,67 GB comprimidos y 6,95 GB extraídos. El inventario de los archivos descargados contiene 8.800 ensayos de train, 880 de test y 1.200 de competitionHoldOut, repartidos en 24, 24 y 15 archivos. Toda la publicación de Dryad ocupa unos 80,44 GB comprimidos, incluyendo otras tareas, intermedios y modelos de lenguaje. |
| **Licencia** | CC0 1.0, según los metadatos de Dryad. |
| **¿Hace falta registro o solicitud?** | No hace falta solicitar acceso. Ya he podido descargar `competitionData` desde el navegador y comprobar su SHA-256. La descarga desde terminal dio problemas, pero los archivos están disponibles en la web. |

Ya tengo un notebook que permite abrir una sesión, elegir una frase y ver sus señales. Cada ensayo contiene características neuronales en intervalos de 20 ms y el texto que T12 intentaba decir. Las transcripciones de `competitionHoldOut` están ocultas, así que no podré calcular su error localmente. Todavía no hay un modelo de habla entrenado ni una métrica propia.

Para empezar usaré `spikePow` y `tx1` del área 6v, siguiendo la recomendación del README oficial. Combinar las primeras 128 columnas de cada variable da 256 características por intervalo. Los datos ya vienen procesados; este paquete no contiene los voltajes originales para repetir la extracción desde cero.

También estoy descargando `sentences`, `diagnosticBlocks` y `tuningTasks` para explorar más contexto y otras tareas. `competitionData` se preparó a partir de las grabaciones de `sentences`, así que no puedo sumarlos como si todas sus frases fueran nuevas. Los otros participantes y tareas que quiero investigar están en el [catálogo de datos](datos.md).

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como para implementarla sin adivinar.

| | |
|---|---|
| **¿Hay código de referencia?** | Sí: [speechBCI](https://github.com/fwillett/speechBCI), con preparación de datos, red recurrente y decodificador de lenguaje. El repositorio también enlaza una [implementación en PyTorch](https://github.com/cffan/neural_seq_decoder). |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | No aplica: hay código de referencia. Tomaré como base una red recurrente con GRU que predice fonemas y un decodificador que los convierte en palabras. |
| **Qué NO especifica el paper** | El paper y el código explican la parte principal del sistema. Lo que no está cerrado para mi versión es qué configuración reducida aprenderá algo útil y cuánto tardará en una T4. Tengo que decidir el tamaño de la red, la validación, el tratamiento de las diferencias entre días y el modelo de lenguaje que pueda utilizar. Revisaré también las diferencias de la adaptación a PyTorch y dejaré las decisiones en los archivos de configuración. |

La red se entrena con CTC, que permite aprender una secuencia de fonemas sin indicar el instante exacto de cada uno en la grabación. Mi primera medida será el **PER**, el error en fonemas. Cuando tenga la conversión a texto, mediré también **WER**, el error en palabras. Un PER no sustituye al WER del paper; para comparar resultados tendré que explicar qué datos, modelo y decodificación he usado.

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

| | |
|---|---|
| **Escala del paper original** | El depósito del estudio describe unas 12.100 frases registradas de T12 y un sistema que combina una red recurrente con modelos de lenguaje. El paquete preparado para aprendizaje automático tiene 8.800 frases de entrenamiento. La réplica offline completa incluye también el decodificador de palabras y los recursos de lenguaje; no tengo un tiempo medido para ejecutarla en una T4. |
| **Escala que voy a hacer** | Para clase, empezaré con unas 500 frases de train, repartidas entre varios bloques y días, una GRU más pequeña y un decodificador de palabras con un diccionario y un modelo de lenguaje más ligero. Dejaré fuera los modelos de lenguaje más pesados, el rescoring con OPT, el trabajo entre participantes y la generación de voz. Las 500 frases son un punto de partida; cerraré el recorte después de medir tiempo y error. |
| **Tiempo estimado del entrenamiento completo** | Mi objetivo es menos de 20 minutos en una T4, con una prueba completa de menos de 5 minutos en CPU. Todavía no lo he medido, así que no marco la casilla de viabilidad. Primero cronometraré una ejecución corta y después el entrenamiento completo de la configuración elegida. Anotaré también el tiempo de preparación y evaluación. |
| **Qué se pierde al recortar** | Entrenar con menos frases, una red más pequeña y menos entrenamiento puede empeorar la predicción de fonemas. Un modelo de lenguaje reducido también puede limitar el vocabulario y la corrección de errores. No podré presentar ese resultado como una reproducción de las cifras del sistema completo ni como una prueba de que funciona con otra persona. |
| **Hardware que voy a usar** | Google Colab gratuito con una T4 cuando esté disponible. Haré también pruebas en local con RTX 2060 o RTX 4070 (lab que nos dejan en deusto) |


#### Lo que quiero hacer sin ese recorte por mi cuenta

Quiero usar el mismo código con dos configuraciones: una que pueda entregar y reproducir dentro de los límites de clase, y otra para seguir experimentando. En esa segunda usaré todas las frases de entrenamiento de T12, probaré un modelo de tamaño comparable al de referencia y le dedicaré más tiempo de entrenamiento. También quiero incorporar modelos de lenguaje más completos y ver si consigo bajar el error de palabras sin cambiar la red neuronal.

Después me gustaría probar T15. Empezaría con su propia preparación de datos y un modelo separado, porque sus entradas no son intercambiables con las de T12. La adaptación entre personas vendría después de comprobar cada caso por separado. El habla interna y la síntesis de voz me interesan como ampliaciones, pero quedan fuera de la primera entrega y requieren otra preparación y evaluación.

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

| | |
|---|---|
| **Qué construís encima** | Quiero hacer una web donde pueda elegir una sesión y una frase, ver la actividad neuronal y ejecutar el decodificador. Mostraría la referencia, la predicción y sus errores. También me interesa poder inspeccionar los fonemas mientras desarrollo la conversión a palabras. La aplicación está por implementar; ahora tengo el notebook de exploración. |
| **Quién lo usaría y para qué** | La usaría para entender y probar el modelo. También podría servir a estudiantes o investigadores que quieran acercarse a estos datos sin disponer de un implante. |
| **Qué necesita del modelo** | Entrada: una secuencia de características neuronales preprocesadas. Salida: probabilidades de fonemas y, cuando esté integrado el decodificador de lenguaje, texto. La aplicación cargará un checkpoint ya entrenado. Mediré cuánto tarda en procesar una grabación antes de fijar un objetivo de latencia. |

Con los archivos públicos puedo trabajar en la parte offline: cargar una grabación, obtener una predicción y compararla con la referencia. Lo que no puedo reproducir es el experimento con un implante y una participante usándolo en directo, porque no tengo ese equipo ni grabaciones propias. Si reproduzco los archivos por ventanas para simular una entrada continua, lo explicaré como una reproducción de datos grabados. También tendré que comprobar si el modelo utiliza información posterior de la frase antes de hablar de funcionamiento en directo.

### Riesgo principal

| | |
|---|---|
| **Qué es lo que más probablemente va a salir mal** | Que el recorte necesario para entrar en los veinte minutos deje un modelo que no aprenda lo suficiente, o que integrar el decodificador de palabras resulte más costoso que entrenar la red pequeña. También tengo que cuidar las diferencias entre sesiones y evitar usar el test para elegir el modelo. |
| **Qué haría si pasa** | Primero comprobaré que el modelo puede aprender unas pocas frases y mediré el coste de cada parte. Ajustaré datos, tamaño de red y entrenamiento y volveré a evaluar. Si no consigo integrar la salida de palabras dentro del presupuesto, propondré al profesor una entrega centrada en fonemas y su visualización. Si el modelo reducido no aprende, tendré que revisar el alcance antes de dar la propuesta por viable. La versión completa seguirá como trabajo personal con su propia configuración. |

### Visto bueno del profesor (H2) · no rellenar

| | |
|---|---|
| **Decisión** | |
| **Condiciones** | |
