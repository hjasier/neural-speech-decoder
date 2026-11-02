# Runtime CPU del trigrama

El smoke y la demo usan `lm_decoder`, el decodificador de palabras del proyecto
original. Se instala en otro entorno con Python 3.10 y PyTorch 1.13.1 CPU;
`requirements.txt` prepara el entorno principal con Python 3.12.

Esta instalación se hace una vez, antes del smoke. No descarga ni carga el
trigrama completo. El smoke genera su propio grafo de tres palabras.

## Compilar en Linux

En Debian/Ubuntu, instala las herramientas de compilación:

```bash
sudo apt-get update
sudo apt-get install git build-essential autoconf automake libtool zlib1g-dev
```

Desde la raíz del repositorio, con el entorno principal activado:

```bash
python -m pip install uv==0.9.4
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/prepare_speech_trigram.py --config configs/entrenos_online/speech_trigrama_compacto.yaml --build-runtime
```

El script prepara Python 3.10.15, instala las versiones fijadas y compila con
dos hilos. Es el mismo procedimiento que usa GitHub Actions. Si ya hay un
runtime válido, lo comprueba y lo reutiliza, también si procede de un respaldo.

## Restaurar un respaldo ya compilado

Si estás en Linux x86_64 puedes usar la versión que yo ya he compilado en colab:

https://drive.google.com/drive/u/0/folders/1UN2NWClYwF7Vn27xIvWUJrTyo_ybNOCm

Coloca estos tres archivos en `data/respaldo_runtime_e8d287b5/`:

- `runtime_trigrama_cpu.tar.gz`
- `runtime_trigrama_cpu.tar.gz.sha256`
- `respaldo_verificado.json`

```bash
python scripts/restore_speech_trigram_runtime.py --backup data/respaldo_runtime_e8d287b5 --sha256 1c91b2797c7a2837c6f237501e6ef95c1c1f8de603d8f84a3eaf0f59c9da51b9
```

El restaurador verifica los archivos y deja el entorno en
`data/language_models/runtime`, sin cambiar el entorno principal.

Tras cualquiera de las dos opciones:

```bash
python run.py --smoke
```
