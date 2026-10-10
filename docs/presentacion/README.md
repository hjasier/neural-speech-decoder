# Presentación del paper

Diez diapositivas, incluida la despedida «Mila esker!», para la exposición de tres minutos de Asier Aurre. La cantidad de diapositivas no es un límite: ensayar el recorrido completo dentro de los tres minutos acordados. El recorrido sigue el [guion revisado](../guion_presentacion.md): cursor y escritura, método del paper, datos y resultados, aplicación e hipótesis de mejora.

## Abrirla

Se puede abrir `index.html` directamente en un navegador. Reveal.js, la tipografía y todas las imágenes están guardados en esta carpeta: la presentación funciona sin internet.

Para usar también la ventana de notas del presentador, ejecutar desde la raíz del repositorio:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs/presentacion
```

Después abrir [la presentación local](http://127.0.0.1:8765/). Permitir la ventana emergente si el navegador bloquea las notas. Las fuentes enlazadas requieren conexión si se quiere consultar los artículos.

## Durante la exposición

| Tecla | Acción |
|---|---|
| → o espacio | Avanzar; algunos elementos aparecen en pasos separados |
| ← | Volver |
| F | Pantalla completa |
| S | Abrir las notas con el guion y los tiempos orientativos |
| Esc | Vista general de las diapositivas |
| R | Repetir las animaciones de la diapositiva actual |
| M | Pausar o activar el movimiento |
| I | Consultar fuentes y controles |

Las figuras originales se amplían al pulsarlas. La diapositiva del proyecto muestra una grabación real de T12 y una maqueta animada de la salida por fonemas y palabras; esas salidas son ilustrativas, no predicciones del modelo. El decodificador todavía está pendiente de integrar.

Conviene ensayar los avances con el guion: las animaciones se ejecutan al entrar y no obligan a esperar para pasar de diapositiva. La preferencia del sistema de reducir movimiento se respeta automáticamente.

## Material

- Una ilustración conceptual y tres ejemplos de señales reales de T12.
- Figuras originales del paper, ampliables.
- Animaciones del cursor, handwriting, recorrido de la señal, CTC y enmascaramiento temporal.
- Notas con el discurso y aclaraciones para preguntas.
- Visor de una muestra neuronal con barrido temporal, fonemas secuenciales y texto ilustrativo.

Los resultados de 62 palabras/minuto y 23,8 % de WER son del artículo original, no del proyecto. Haré dos versiones de decodificación offline con el mismo código y configuraciones distintas: una de entrega con los recortes necesarios para entrenar en menos de veinte minutos, y otra completa con las 8.800 frases de entrenamiento de T12 y el tiempo que necesite. Ambas mantienen las demás restricciones del curso, incluida la prueba corta de menos de cinco minutos en CPU. El tiempo, el consumo y la calidad de cada versión están pendientes de medir. Comparar el Transformer causal con la GRU es una hipótesis experimental: aún no hay resultados propios ni una mejora garantizada.

Los [créditos, fuentes y prompts](CREDITOS.md) detallan la procedencia de cada recurso. Los ejemplos usados están documentados en `assets/data-source.json`. Las visualizaciones aplican `log1p` y escalas para mostrar la actividad; ese procesamiento no es una normalización de entrenamiento ni un experimento controlado de variación entre días.

## Editar

`index.html` contiene las diapositivas y las notas, `styles.css` el diseño y las animaciones, y `deck.js` las interacciones. `assets/data.js` carga los tres registros públicos que necesita la demostración. No hace falta instalar paquetes ni compilar.
