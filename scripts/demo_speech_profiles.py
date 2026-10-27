"""Demo local T12: actividad y fonemas con salidas elegidas desde pesos guardados."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from textwrap import fill
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import build_smoother
from src.data.speech import AREA_6V_CHANNELS, BIN_SECONDS
from src.data.speech_dataset import SpeechDataset, collate_speech, file_sha256
from src.models.factory import construir_modelo
from src.training.checkpoints import load_checkpoint
from src.training.metrics import decode_ctc, edit_distance
from src.utils.seed import fijar_seed


class SpeechProfileDemo:
    """Mantiene pesos e input de una frase para predecir por distintos prefijos."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Recarga v1/v2 con datos compatibles; prepara una entrada común sin ruido."""
        self.cfg = cfg
        self._closed = False
        for key in ("calentamiento", "repeticiones"):
            if type(cfg["medicion"][key]) is not int or cfg["medicion"][key] <= 0:
                raise ValueError("Las medidas requieren calentamientos y repeticiones positivos.")
        if type(cfg["ejecucion"]["hilos_cpu"]) is not int or cfg["ejecucion"]["hilos_cpu"] <= 0:
            raise ValueError("hilos_cpu debe ser un entero positivo.")
        torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
        self.device = torch.device(cfg["ejecucion"]["dispositivo"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. Selecciona el kernel .venv con PyTorch CUDA.")
        self.checkpoint = Path(cfg["checkpoint"])
        digest = file_sha256(self.checkpoint)
        header = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if header.get("format_version") not in (1, 2) or header["config"]["modelo"]["nombre"] != "speech_gru":
            raise ValueError("La demo requiere un checkpoint T12 del proyecto.")
        self._model_cfg = header["config"]
        if self._model_cfg["modelo"]["entrada"] != 2 * AREA_6V_CHANNELS:
            raise ValueError("La actividad T12 requiere las 256 características tx1/spikePow de 6v.")
        # La cabecera se usa para elegir los datos; liberar sus pesos evita
        # mantener dos copias CPU al llamar al cargador estricto compartido.
        del header
        split = cfg["datos"]["split"]
        if split == "auto":
            split = "val" if self._model_cfg.get("protocolo") else "test"
        if split not in ("val", "test"):
            raise ValueError("La demo admite val/test, o auto para usar el protocolo del checkpoint.")
        dataset = SpeechDataset(self._model_cfg, split)
        name = cfg["datos"]["sesion"]
        if name not in dataset.sessions:
            raise ValueError("La sesión no está disponible en el checkpoint.")
        position = cfg["datos"]["frase"]
        indices = dataset.session_indices[dataset.sessions.index(name)]
        if type(position) is not int or not 0 <= position < len(indices):
            raise ValueError("frase debe ser un índice válido dentro de la sesión y split elegidos.")
        self._sample = dataset[indices[position]]
        fijar_seed(self._model_cfg["experimento"]["seed"])
        self._model = construir_modelo(self._model_cfg)
        state = load_checkpoint(self.checkpoint, self._model, self._model_cfg, dataset)
        self._model.to(self.device).eval()
        self.available_exits = self._model.available_exits
        batch = collate_speech([self._sample]).to(self.device)
        with torch.inference_mode():
            self._features = build_smoother(self._model_cfg).to(self.device)(batch.features)
        self._sessions = batch.session_ids
        self._lengths = self._model.output_lengths(batch.input_lengths)
        self._input_cpu = self._features.squeeze(0).cpu().numpy()
        self._phones, self._blank = state["phones"], state["blank_id"]
        self.output = Path(cfg["salida"]["directorio"]) / f"{digest[:12]}_{uuid4().hex[:8]}"
        self.output.mkdir(parents=True, exist_ok=False)
        np.save(self.output / "entrada_modelo.npy", self._input_cpu, allow_pickle=False)
        (self.output / "config_demo.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (self.output / "config_checkpoint.yaml").write_text(yaml.safe_dump(self._model_cfg, sort_keys=False), encoding="utf-8")
        if file_sha256(self.checkpoint) != digest:
            raise ValueError("El checkpoint cambió durante la carga. Repite con un archivo estable.")
        self.metadata = {
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": digest,
            "checkpoint_step": state["step"], "format_version": state["format_version"],
            "protocol": self._model_cfg.get("protocolo"), "split": split,
            "source_split": "train" if split == "val" else "test",
            "session": name, "position_in_session": position,
            "source_trial_index": self._sample.trial_index, "block": self._sample.block,
            "reference_text": self._sample.reference, "input_shape": list(self._input_cpu.shape),
            "input_sha256": file_sha256(self.output / "entrada_modelo.npy"),
            "duration_seconds": len(self._input_cpu) * BIN_SECONDS,
            "available_exits": list(self.available_exits),
            "hardware": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "device": str(self.device), "torch_version": str(torch.__version__),
            "training_updates": 0, "tracking_run_opened": False,
        }
        (self.output / "ejemplo.json").write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.last_prediction: dict[str, Any] | None = None

    def close(self) -> None:
        """Libera el modelo y tensores GPU aunque un widget antiguo conserve la demo."""
        self._closed = True
        self._model = None
        self._features = None
        self._sessions = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def draw_activity(self) -> Path:
        """Guarda los dos mapas de la misma entrada que recibirán todos los perfiles."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        spec = self.cfg["figura"]
        if not np.isfinite(spec["limite_z"]) or spec["limite_z"] <= 0:
            raise ValueError("El límite de color debe ser finito y positivo.")
        meta = self.metadata
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
        fig.suptitle(f"T12 · {meta['session']} · {meta['split']} · ensayo original {meta['source_trial_index']}\n"
                     + fill(meta["reference_text"], 100), fontsize=13)
        for axis, offset, label in zip(axes, (0, AREA_6V_CHANNELS),
                                       ("Cruces de umbral · tx1", "Potencia de spikes · spikePow"), strict=True):
            plotted = axis.imshow(self._input_cpu[:, offset:offset + AREA_6V_CHANNELS].T,
                                  aspect="auto", origin="lower", interpolation="nearest",
                                  extent=(0, meta["duration_seconds"], 0.5, AREA_6V_CHANNELS + 0.5),
                                  cmap=spec["cmap"], vmin=-spec["limite_z"], vmax=spec["limite_z"])
            axis.set(title=label, ylabel="Canal de 6v")
            fig.colorbar(plotted, ax=axis, label="Entrada normalizada y suavizada")
        axes[-1].set_xlabel(f"Tiempo desde el inicio del ensayo (s) · bins de {BIN_SECONDS * 1000:g} ms")
        path = self.output / "actividad_neuronal.png"
        fig.savefig(path, dpi=spec["dpi"])
        plt.close(fig)
        return path

    @torch.inference_mode()
    def predict(self, depth: int) -> dict[str, Any]:
        """Predice con un prefijo real; devuelve fonemas, PER y tiempos de la red."""
        if self._closed:
            raise RuntimeError("La demo está cerrada. Vuelve a ejecutar la celda de carga.")
        if type(depth) is not int or depth not in self.available_exits:
            raise ValueError(f"El checkpoint solo dispone de las salidas {self.available_exits}.")
        if (file_sha256(self.checkpoint) != self.metadata["checkpoint_sha256"]
                or file_sha256(self.output / "entrada_modelo.npy") != self.metadata["input_sha256"]):
            raise ValueError("Cambió el checkpoint o la entrada guardada. Vuelve a cargar la demo.")
        self._model.eval()
        calls = []
        handles = [layer.register_forward_pre_hook(lambda _m, _a, n=i: calls.append(n))
                   for i, layer in enumerate(self._model.gru_layers, start=1)]
        try:
            probe = self._model(self._features, self._sessions, exit_layer=depth)
            expected = decode_ctc(probe, self._lengths, self._blank)[0]
            del probe
        finally:
            for handle in handles:
                handle.remove()
        if calls != list(range(1, depth + 1)):
            raise ValueError("El perfil ejecutó capas distintas a las solicitadas.")
        spec = self.cfg["medicion"]
        for _ in range(spec["calentamiento"]):
            value = self._model(self._features, self._sessions, exit_layer=depth)
            del value
        times = []
        for _ in range(spec["repeticiones"]):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = perf_counter()
            logits = self._model(self._features, self._sessions, exit_layer=depth)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            times.append((perf_counter() - start) * 1000)
            hypothesis = decode_ctc(logits, self._lengths, self._blank)[0]
            del logits
            if hypothesis != expected:
                raise ValueError("Las repeticiones del perfil no produjeron los mismos fonemas.")
        reference = self._sample.targets.tolist()
        errors = edit_distance(reference, hypothesis)
        result = {**self.metadata, "exit": depth, "executed_layers": calls,
                  "hypothesis_ids": hypothesis, "hypothesis_phones": [self._phones[i] for i in hypothesis],
                  "reference_ids": reference, "reference_phones": [self._phones[i] for i in reference],
                  "errors": errors, "reference_phone_count": len(reference), "per": errors / len(reference),
                  "forward_ms_samples": times, "forward_ms_p50": float(np.median(times)),
                  "forward_ms_p95": float(np.percentile(times, 95)),
                  "timing_scope": "red con entrada ya preparada en dispositivo; warmup previo; sin greedy ni carga/preprocesado",
                  "metric_scope": "una frase fija; incluye SIL; CTC greedy"}
        (self.output / f"prediccion_capa_{depth}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        self.last_prediction = result
        return result


def load_demo(cfg: dict[str, Any]) -> SpeechProfileDemo:
    """Carga la demo sin abrir W&B ni ejecutar entrenamiento."""
    return SpeechProfileDemo(cfg)


def create_controls(demo: SpeechProfileDemo) -> Any:
    """Crea selector de salidas existentes, botón y resultado para el notebook local."""
    try:
        import ipywidgets as widgets
    except ImportError as error:
        raise RuntimeError("Instala requirements-demo.txt en el kernel local para usar el selector.") from error
    selector = widgets.Dropdown(options=[(f"{depth} capas", depth) for depth in demo.available_exits],
                                value=demo.available_exits[-1], description="Perfil:")
    button = widgets.Button(description="Predecir fonemas", button_style="primary")
    output = widgets.Output()

    def _predict(_button: Any) -> None:
        button.disabled = True
        try:
            with output:
                output.clear_output(wait=True)
                result = demo.predict(selector.value)
                print(f"Salida {result['exit']} · GRU ejecutadas: {result['executed_layers']}")
                print("Fonemas predichos:\n" + (" ".join(result["hypothesis_phones"]) or "[secuencia vacía]"))
                print("\nReferencia:\n" + " ".join(result["reference_phones"]))
                print(f"\nPER de esta frase: {result['per']:.2%} ({result['errors']}/{result['reference_phone_count']})")
                print(f"Mediana de la red: {result['forward_ms_p50']:.3f} ms; entrada ya preparada.")
                print("Resultado:", demo.output / f"prediccion_capa_{result['exit']}.json")
        finally:
            button.disabled = False

    button.on_click(_predict)
    return widgets.VBox([widgets.HBox([selector, button]), output])


def main() -> int:
    """Ejecuta también la demo por terminal, sin necesitar un frontend de widgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/speech_demo_salidas.yaml"))
    parser.add_argument("--exit-layer", type=int, help="por defecto, salida final disponible")
    args = parser.parse_args()
    demo = load_demo(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    print(f"Checkpoint paso {demo.metadata['checkpoint_step']}; salidas {demo.available_exits}; split {demo.metadata['split']}.")
    print("Actividad:", demo.draw_activity())
    depth = args.exit_layer if args.exit_layer is not None else demo.available_exits[-1]
    result = demo.predict(depth)
    print("Referencia:", result["reference_text"])
    print("Fonemas predichos:", " ".join(result["hypothesis_phones"]) or "[secuencia vacía]")
    print(f"PER de una frase: {result['per']:.2%}; red: {result['forward_ms_p50']:.3f} ms; cero actualizaciones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
