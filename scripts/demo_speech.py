"""Inferencia AHPSN de una frase: checkpoint real, actividad y fonemas, sin entrenar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from textwrap import fill
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phonemes import PhonemeEncoder
from src.data.preprocessing import TrainingNormalization, build_smoother
from src.data.speech import AREA_6V_CHANNELS, BIN_SECONDS, load_session
from src.data.speech_dataset import file_sha256
from src.models.factory import construir_modelo
from src.training.metrics import decode_ctc, edit_distance


def _checkpoint(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("format_version") != 1 or "model_state" not in state
            or state["config"]["modelo"]["nombre"] != "speech_gru"
            or state["config"]["modelo"]["entrada"] != 2 * AREA_6V_CHANNELS
            or state["sessions"] != state["config"]["datos"]["sesiones"]
            or state["phones"] != state["config"]["datos"]["fonemas"]
            or state["blank_id"] != len(state["phones"])):
        raise ValueError("Se necesita un checkpoint T12/AHPSN del proyecto, con configuración y normalización.")
    return state


def inspect_checkpoint(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Lee pesos y configuración reales; exporta la configuración para recursos g2p."""
    state = _checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_config.yaml").write_text(yaml.safe_dump(state["config"], sort_keys=False))
    info = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "step": state["step"], "model": state["config"]["modelo"],
            "sessions": state["sessions"], "phones": state["phones"], "torch_training": state["torch_version"]}
    (output / "checkpoint_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Checkpoint leído: {checkpoint}\nPaso guardado: {state['step']}", flush=True)
    print(f"Modelo: {info['model']}\nSesiones disponibles: {', '.join(state['sessions'])}", flush=True)
    return info


def prepare_example(checkpoint: Path, corpus: Path, session: str, trial_index: int,
                    output: Path, demo_cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepara y dibuja el ensayo test usando solo estadísticas de train guardadas."""
    state = _checkpoint(checkpoint)
    if session not in state["sessions"] or Path(session).name != session:
        raise ValueError("La sesión debe estar entre las usadas por este checkpoint.")
    source = corpus / "test" / f"{session}.mat"
    manifest = json.loads(Path(demo_cfg["datos"]["manifiesto"]).read_text())
    expected = manifest["files"][source.name]
    source_hash = file_sha256(source)
    if source.stat().st_size != expected["size_bytes"] or source_hash != expected["sha256"]:
        raise ValueError("La sesión test no coincide con el archivo T12 verificado del catálogo.")
    recording = load_session(source)
    if not 0 <= trial_index < len(recording.trials):
        raise ValueError(f"El índice de frase debe estar entre 0 y {len(recording.trials) - 1}.")
    trial = recording.trials[trial_index]
    if trial.text is None:
        raise ValueError("La demo requiere una frase de test con referencia pública.")
    stats = TrainingNormalization.from_dict(state["normalization"][session])
    features = torch.from_numpy(stats.transform(trial, session)).unsqueeze(0)
    with torch.inference_mode():
        model_input = build_smoother(state["config"])(features).squeeze(0).numpy()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "entrada_modelo.npy", model_input, allow_pickle=False)
    metadata = {"checkpoint_sha256": file_sha256(checkpoint), "checkpoint_step": state["step"],
                "source": str(source), "source_sha256": source_hash, "split": "test",
                "session": session, "session_id": state["sessions"].index(session),
                "trial_index": trial_index, "block": trial.block, "reference_text": trial.text,
                "input_shape": list(model_input.shape), "duration_seconds": trial.duration_seconds,
                "input_sha256": file_sha256(output / "entrada_modelo.npy"),
                "preprocessing": "tx1 6v seguido de spikePow 6v; normalización train del checkpoint y suavizado; sin ruido"}
    (output / "ejemplo.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, layout="constrained")
    fig.suptitle(f"AHPSN · {session} · test[{trial_index}]\n" + fill(trial.text, 100), fontsize=13)
    limit = demo_cfg["figura"]["limite_z"]
    if limit <= 0:
        raise ValueError("figura.limite_z debe ser positivo.")
    for axis, offset, label in zip(axes, (0, AREA_6V_CHANNELS),
                                   ("Cruces de umbral · tx1", "Potencia de spikes · spikePow"), strict=True):
        plotted = axis.imshow(model_input[:, offset:offset + AREA_6V_CHANNELS].T, aspect="auto", origin="lower",
                              extent=(0, trial.duration_seconds, 0.5, AREA_6V_CHANNELS + 0.5),
                              cmap=demo_cfg["figura"]["cmap"], vmin=-limit, vmax=limit, interpolation="nearest")
        axis.set(title=label, ylabel="Canal de 6v")
        fig.colorbar(plotted, ax=axis, label="Entrada normalizada y suavizada (z)")
    axes[-1].set_xlabel(f"Tiempo desde el inicio del ensayo (s) · bins de {BIN_SECONDS * 1000:g} ms")
    fig.savefig(output / "actividad_neuronal.png", dpi=demo_cfg["figura"]["dpi"])
    plt.close(fig)
    print(f"Frase de referencia: {trial.text}\nSesión: {session} · test[{trial_index}] · bloque {trial.block}", flush=True)
    print(f"Entrada real del modelo: {list(model_input.shape)} · duración {trial.duration_seconds:.2f} s", flush=True)
    return metadata


def predict_example(checkpoint: Path, output: Path, demo_cfg: dict[str, Any]) -> dict[str, Any]:
    """Carga el modelo entrenado y predice desde la señal neural; compara con g2p."""
    state = _checkpoint(checkpoint)
    metadata = json.loads((output / "ejemplo.json").read_text())
    if (metadata["checkpoint_sha256"] != file_sha256(checkpoint)
            or metadata["input_sha256"] != file_sha256(output / "entrada_modelo.npy")
            or state["sessions"][metadata["session_id"]] != metadata["session"]):
        raise ValueError("Cambió el checkpoint o la entrada. Vuelve a ejecutar la celda de actividad.")
    device = torch.device(demo_cfg["ejecucion"]["dispositivo"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible en este Python. Selecciona el kernel .venv del proyecto con PyTorch CUDA.")
    model = construir_modelo(state["config"])
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    features = np.load(output / "entrada_modelo.npy", allow_pickle=False)
    if list(features.shape) != metadata["input_shape"] or not np.isfinite(features).all():
        raise ValueError("Forma o valores no válidos en la entrada preparada.")
    signal = torch.from_numpy(features).unsqueeze(0).to(device)
    session_ids = torch.tensor([metadata["session_id"]], device=device)
    start = perf_counter()
    with torch.inference_mode():
        logits = model(signal, session_ids)
        lengths = model.output_lengths(torch.tensor([len(features)]))
        predicted = decode_ctc(logits, lengths, state["blank_id"])[0]
    seconds = perf_counter() - start
    # La transcripción solo se convierte después de inferir; no entra en model(...).
    target = PhonemeEncoder(state["config"]).encode(metadata["reference_text"])
    errors = edit_distance(target.ids, predicted)
    phones = [state["phones"][i] for i in predicted]
    result = {**metadata, "prediction_ids": predicted, "prediction_phones": phones,
              "reference_ids": list(target.ids), "reference_phones": list(target.phones),
              "errors": errors, "reference_phone_count": len(target.ids), "per": errors / len(target.ids),
              "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "inference_seconds": seconds, "torch_inference": str(torch.__version__),
              "training_updates": 0, "decoding": "CTC greedy; repeticiones consecutivas colapsadas y blank retirado",
              "note": "PER de una frase, incluyendo SIL; no WER ni resultado agregado del test."}
    (output / "prediccion.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("FONEMAS PREDICHOS:\n" + (" ".join(phones) or "[secuencia vacía]"), flush=True)
    print("\nFONEMAS DE REFERENCIA (g2p-en):\n" + " ".join(target.phones), flush=True)
    print(f"\nPER de esta frase: {result['per']:.2%} ({errors}/{len(target.ids)}) · incluye SIL", flush=True)
    print(f"Inferencia: {seconds:.3f} s · {device} · cero actualizaciones", flush=True)
    return result


def main() -> int:
    """Expone las tres acciones de la demo para ejecutarlas en celdas separadas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=("inspeccionar", "preparar", "predecir"))
    parser.add_argument("--config", type=Path, default=Path("configs/speech_demo.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--sesion")
    parser.add_argument("--frase", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    if args.accion == "inspeccionar":
        inspect_checkpoint(args.checkpoint, args.salida)
    elif args.accion == "preparar":
        if args.corpus is None:
            parser.error("preparar requiere --corpus")
        prepare_example(args.checkpoint, args.corpus, args.sesion or cfg["datos"]["sesion"],
                        cfg["datos"]["frase"] if args.frase is None else args.frase, args.salida, cfg)
    else:
        predict_example(args.checkpoint, args.salida, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
