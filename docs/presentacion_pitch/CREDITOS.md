# Fuentes y procedencia

- Paper: Willett et al. (2023), [A high-performance speech neuroprosthesis](https://doi.org/10.1038/s41586-023-06377-x).
- Datos T12: [publicación de Dryad](https://doi.org/10.5061/dryad.x69p8czpq), `competitionData`. Las 8.800 son ensayos de train, no frases distintas garantizadas.
- Tiempo: [prueba del proyecto](../prueba_tiempo_entrenamiento.md) y [medidas originales de 5 × 512](../pruebas/tiempo_gru_2026-10-11/grande.json). Los 13,58 min se redondean a ≈14. Son una proyección, con margen del 25 %, para 2.000 ensayos, lote 8 y 20 épocas; no incluyen la ejecución completa del pipeline.
- Estructura del pitch: documento docente aportado por el usuario, «Tu proyecto en 5 minutos», consultado el 12/10/2026. Tres diapositivas y cuatro minutos de explicación, más uno de preguntas.

## Comparación de calidad

La tabla 1 y el apartado «Decoding attempted speech» del paper reportan un **23,8 % de WER** para habla vocalizada con vocabulario de **125.000 palabras**. La diapositiva muestra **76,2 % = 100 − 23,8 %**, identificado como `100 − WER`. No representa accuracy de clasificación ni el porcentaje exacto de palabras correctamente reconocidas, porque WER incluye inserciones además de sustituciones y omisiones. No se usa el 9,1 % de WER del vocabulario restringido de 50 palabras.

Para el recorte se plantea **≈70 % esperado en 100 − WER**, equivalente a **≈31 % WER** antes de redondear la cifra positiva. La cuenta usa una referencia del paper, no las pruebas de clasificación de fonemas del proyecto:

1. En «Decoding attempted speech» los autores indican 260–480 frases recogidas al inicio de cada día de evaluación. Tomar unas 400 frases/día aproxima 2.000 frases a 5 días equivalentes de datos. No identifica nuestras muestras con cinco sesiones concretas.
2. En [Extended Data Fig. 5i](../s41586-023-06377-x.pdf) se leen aproximadamente 25 % WER con 5 días previos y 18 % con 15: unos 7 puntos de pérdida. Son valores aproximados leídos de la curva, no cifras exactas de una tabla. La curva promedia 10 semillas y 5 días de evaluación; las barras muestran desviaciones estándar entre semillas.
3. Usamos esa diferencia como penalización orientativa por reducir datos sobre el resultado principal: `23,8 + (≈25 − ≈18) ≈31 % WER`; `100 − ≈31 ≈69 %`, redondeado a **≈70 %** para el pitch.

Trasladar una diferencia del barrido offline al resultado en tiempo real es una inferencia nuestra. La figura reduce días previos con el resto del sistema de los autores; no prueba un subconjunto de 2.000 frases repartido entre sesiones ni un entrenamiento de 20 épocas. La expectativa supone un modelo de lenguaje, adaptación entre días y aprendizaje suficientes y comparables. No se ha medido la penalización de limitar las épocas. Si esas condiciones fallan, el error podría ser mayor. No es una métrica medida por nuestro modelo ni una extrapolación calibrada con intervalo de confianza; habrá que evaluarla.

Fig. 4c reduce ejemplos del día de evaluación manteniendo los anteriores; no se usa para esta cuenta. Fig. 5h compara anchuras, pero mantener 512 unidades no es un recorte de anchura respecto al paper.

Extended Data Fig. 5h señala 512 unidades como configuración usada en la evaluación en tiempo real. Las 1.024 de la animación son la variante grande del piloto del proyecto, no la anchura original del paper.

## Señal y ejemplos

Las curvas se reutilizan del extracto de la presentación anterior: `../presentacion/assets/data.js` y su `data-source.json`. Registro de T12 del 25/08/2022, ensayo 0, bloque 1, referencia «Is that pollution?», duración 4,14 s. Se muestran cuatro canales de `spikePow` de 6v (9, 31, 64 y 100).

Para visualizar las curvas se había aplicado `log1p`, reescalado por los percentiles 2 y 98 y reducido los puntos. Es una transformación gráfica del extracto, no la normalización del entrenamiento. La procedencia está en [assets/procedencia.json](assets/procedencia.json).

El cerebro y el implante son dibujos SVG esquemáticos: no indican una posición anatómica exacta. Los fonemas de «hello», «Is that pollution?» e «Is this solution?», las palabras predichas y su sincronización con la señal son ilustrativos. No salen de un modelo entrenado ni de una alineación del dataset. Las sustituciones this/that y solution/pollution tampoco constituyen una medición de calidad.

La maqueta del aplicativo emite la grabación en bloques de 20 ms, al ritmo original, y oculta los valores futuros. Al entrar en la diapositiva muestra el recorrido previsto con la salida ilustrativa correcta. Hay dos escenarios de fallo, cada uno reinicia la animación:

- Primera pulsación: el procesamiento avanza tres veces más despacio que la entrada. Dos cursores distinguen señal recibida y procesada, una banda naranja muestra la diferencia y la fila «En espera» representa la cola. Los segmentos son una escala gráfica del volumen pendiente; el número muestra su duración en segundos de señal, no el tiempo total que tardará en vaciarse. Las palabras salen tarde y la animación continúa hasta vaciar la cola.
- Segunda pulsación: procesa al ritmo de la entrada, pero genera «Is this solution?» frente a la referencia «Is that pollution?». Los dos errores se resaltan al aparecer las palabras. La referencia sirve para comparar, no como entrada al decodificador.

La salida normal y el escenario con errores usan un desfase ilustrativo de 350 ms. El factor de lentitud, los tiempos, la cola y las predicciones se eligen para explicar los fallos; no son medidas de latencia ni promesas de rendimiento. «Reiniciar» y R repiten el escenario seleccionado; volver atrás cambia al anterior. Con movimiento desactivado, el caso lento conserva una instantánea de la cola y el caso de errores muestra la comparación completa.

## Recursos locales

- Reveal.js 6.0.2: biblioteca y plugin de notas reutilizados de la presentación anterior. [Licencia MIT incluida](vendor/reveal/LICENSE).
- Manrope: fuente local reutilizada de la presentación anterior. [Licencia SIL Open Font License incluida](assets/fonts/OFL.txt).
- Diseño, diagramas, animaciones y maqueta nuevos para este pitch. No requieren servicios externos ni cargan un modelo.
