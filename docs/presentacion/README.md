# Presentación del paper

Once diapositivas, incluida la despedida «Mila esker!», para la exposición de tres minutos de Asier Aurre. El recorrido sigue el [guion aprobado](../guion_presentacion.md): perder la posibilidad de comunicarse, los intentos de escritura, el salto al habla intentada, el método y el proyecto que quiero desarrollar.

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

Las figuras originales se amplían al pulsarlas. En la diapositiva del proyecto se puede elegir entre tres grabaciones reales; la frase mostrada es su referencia publicada. El decodificador todavía está pendiente de integrar.

Conviene ensayar los avances con el guion: las animaciones se ejecutan al entrar y no obligan a esperar para pasar de diapositiva. La preferencia del sistema de reducir movimiento se respeta automáticamente.

## Material

- Dos ilustraciones conceptuales y tres ejemplos de señales reales de T12.
- Figuras originales del paper, ampliables.
- Animaciones del cursor, handwriting, recorrido de la señal, CTC y enmascaramiento temporal.
- Notas con el discurso y aclaraciones para preguntas.
- Selector de grabaciones con mapas y frases de referencia reales.

Los resultados de 62 palabras/minuto y 23,8 % de WER son del artículo original, no del proyecto. La comparación con un Transformer causal y las otras mejoras se presentan como experimentos propuestos.

Los [créditos, fuentes y prompts](CREDITOS.md) detallan la procedencia de cada recurso. Los ejemplos usados están documentados en `assets/data-source.json`. Las visualizaciones aplican `log1p` y escalas para mostrar la actividad; ese procesamiento no es una normalización de entrenamiento ni un experimento controlado de variación entre días.

## Editar

`index.html` contiene las diapositivas y las notas, `styles.css` el diseño y las animaciones, y `deck.js` las interacciones. `assets/data.js` carga los tres registros públicos que necesita la demostración. No hace falta instalar paquetes ni compilar.
