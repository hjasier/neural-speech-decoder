# Medidas disponibles para preparar la tutoría

Notas factuales a 20 de octubre de 2026. La explicación personal, interpretación
y discusión de la entrega corresponden al autor.

- **Base T12:** arquitectura offline portada de TensorFlow a PyTorch y conectada
  a datos, CTC, PER, W&B y checkpoints. El autor completó 2.001 pasos en T4:
  PER 32,22 % sobre 680 frases en unos 45 minutos. Falta comprobar la recarga
  de ese archivo concreto, el perfil docente T12 y el entrenamiento original
  de 10.001 pasos. No confundir el perfil acortado con la réplica completa.
- **Adaptación propia a Alice:** misma clase SpeechGRU y mismo CTC/conversor
  de fonemas/lotes/métricas; entrada EEG de escucha, 60 canales y modelo menor.
  Train/val/test separados por tramos del audiolibro; estadísticos solo train.
  Referencias por transcripción, sin tiempos de fonemas para CTC.
- **Primera ejecución Alice:** 1.000 pasos, 97,80 s en CPU con dos hilos.
  Test: 36 fragmentos y 1.102 fonemas incluyendo SIL; PER 75,05 %. El prior
  constante ajustado con train obtiene 77,31 %. Sin SIL, el PER del modelo
  es 90,89 %. No es una demostración de lectura de pensamiento ni de
  decodificación de habla funcional. No hay WER ni modelo de lenguaje.
- **Reproducibilidad Alice:** checkpoint elegido por CTC de validación y
  recargado en otro proceso; cero actualizaciones y mismas CTC/PER.
  Fuentes, configuración y hashes documentados; entrenamiento sincronizado
  en [W&B](https://wandb.ai/aaurre-deusto/tiiac-2026/runs/gk69kbxx).
  El lanzador Colab nuevo está preparado, pendiente de ejecución remota.
- **Posibles próximos bloques para decidir:** controles temporales más fuertes
  y revisión de las hipótesis de fonemas; después ampliar participantes con un
  protocolo que reserve texto de test en todos ellos. La validación entre
  personas requerirá además decidir cómo adaptar las entradas por participante.
- **T15 Brain-to-voice:** esquema y propuesta revisados, sin entrenamiento.
  Dryad respondió HTTP 403 al archivo de control. No se descargaron los 26 GB.

Detalle: `docs/experimentos/alice_ctc.md` y resultados JSON en `docs/resultados/`.
El piloto previo de regresión `sound_power` se conserva como exploración
histórica con un problema de referencia en auditoría; no interviene en CTC.
