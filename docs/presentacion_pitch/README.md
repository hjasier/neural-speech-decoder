# Pitch del proyecto

Presentación nueva con fondo blanco: **3 diapositivas, 4 minutos de explicación y 1 minuto para preguntas**. La presentación anterior sigue en `docs/presentacion/`.

Abre [index.html](index.html) en un navegador. Incluye la fuente, Reveal.js y las curvas; las animaciones no necesitan Internet. Mantén toda esta carpeta junta.

Para usar las notas del presentador, abre desde un servidor local. Desde la raíz del repo:

```bash
python -m http.server 8767 --bind 127.0.0.1 --directory docs
```

Después abre `http://127.0.0.1:8767/presentacion_pitch/index.html`.

- **→ o espacio:** siguiente paso de la animación.
- **←:** volver.
- **R:** repetir la animación de la diapositiva.
- **F:** pantalla completa.
- **S:** notas del presentador y reloj.
- **M:** activar o desactivar movimiento.
- **I:** fuentes y controles.

Las diapositivas están numeradas del 01 al 03. Las tres escenas son: señal → fonemas → texto; recorte de datos y anchura del modelo; maqueta de recepción por bloques y texto durante la recepción. Avanzas cuando llegas a esa parte de la explicación. El [guion](guion.md) indica las pulsaciones y el reparto de tiempo.

La tercera emite una grabación al ritmo original, en bloques de 20 ms. No muestra los datos futuros. La primera pulsación reinicia con un modelo lento: se acumula señal pendiente y las palabras aparecen tarde. La segunda reinicia al ritmo normal con palabras equivocadas y muestra la referencia. Son dos animaciones separadas. «Reiniciar» o R repiten el escenario actual; la flecha izquierda vuelve al anterior. Al desactivar el movimiento, cada fallo conserva una representación estática.

Los ≈14 minutos son una proyección del piloto en RTX 2060, con 20 épocas y un margen del 25 %. Todavía faltan la ejecución completa y la medida en T4. La maqueta usa una señal real, pero su decodificación y el retraso de la animación están simulados. No presenta una métrica obtenida por nuestro modelo.

La comparación de calidad usa 76,2 % (`100 − WER`) para el resultado publicado con vocabulario amplio. La cifra esperada de ≈70 % se apoya en la curva de Extended Data Fig. 5i: unos 25 % WER con 5 días frente a 18 % con 15, una pérdida de ≈7 puntos. Unas 400 frases por día aproximan las 2.000 a 5 días equivalentes. Trasladar esa pérdida al 23,8 % publicado da ≈31 % WER, o ≈70 % en la diapositiva. Es una aproximación condicionada a un modelo de lenguaje comparable y entrenamiento suficiente, pendiente de evaluar; no cuantifica la pérdida por limitarse a 20 épocas. Las 1.024 unidades mostradas antes del recorte son una variante del piloto; el paper ya usa 512.

Fuentes y procedencia: [CREDITOS.md](CREDITOS.md).
