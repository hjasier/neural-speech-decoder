"""Figuras descriptivas T12 a partir de resultados compatibles y trazables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import isclose, isfinite
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import yaml


def _read(path: str, wrapper: str) -> tuple[dict[str, Any], dict[str, str]]:
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    return value.get(wrapper, value), {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}


def _valid_metrics(metrics: dict[str, Any]) -> None:
    if (metrics["completo"] != 1 or any(type(metrics[key]) is not int or metrics[key] < 0
                                       for key in ("errores", "fonemas", "frases"))
            or not metrics["fonemas"] or not metrics["frases"]
            or not isfinite(metrics["ctc_loss"]) or metrics["ctc_loss"] < 0
            or not isfinite(metrics["per"]) or metrics["per"] < 0
            or not isclose(metrics["per"], metrics["errores"] / metrics["fonemas"], abs_tol=1e-12)):
        raise ValueError("Las métricas requieren recuentos íntegros, PER consistente y evaluación completa.")


def collect_plot_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """Une PER y latencia por hash, split y salida; rechaza condiciones mezcladas.

    Cada punto conserva su semilla de entrenamiento. Las repeticiones del
    benchmark no se convierten en semillas ni se eligen mejores pasos por cabeza.
    """
    if cfg["etapa_latencia"] not in ("forward", "raw_to_phones") or not cfg["registros"]:
        raise ValueError("Se necesitan registros y una etapa de latencia válida.")
    points, sessions, histories, sources = [], [], [], []
    common = None
    seen = set()
    hashes = set()
    for entry in cfg["registros"]:
        evaluation, source_eval = _read(entry["evaluacion"], "evaluation")
        benchmark, source_bench = _read(entry["benchmark"], "benchmark")
        sources.extend([source_eval, source_bench])
        digest = evaluation["checkpoint_sha256"]
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or digest != benchmark["checkpoint_sha256"]
                or evaluation["checkpoint_step"] != benchmark["checkpoint_step"]
                or evaluation["training_seed"] != benchmark["training_seed"]
                or evaluation["split"] != benchmark["split"]
                or evaluation["data_signature"] != benchmark["inputs"]["data_signature"]
                or evaluation["optimizer_steps"] != 0 or benchmark["optimizer_steps"] != 0
                or benchmark["decoded_profiles_match"] is not True):
            raise ValueError("PER y latencia deben proceder del mismo checkpoint, semilla, datos y split.")
        duplicate = (entry["etiqueta"], evaluation["training_seed"])
        if duplicate in seen or digest in hashes:
            raise ValueError("Solo se admite un checkpoint por variante/semilla; hay un duplicado.")
        seen.add(duplicate)
        hashes.add(digest)
        profiles = benchmark["profiles"]
        if set(profiles) != set(evaluation["global"]["original"]):
            raise ValueError("Las salidas de calidad y benchmark deben coincidir.")
        first = next(iter(profiles))
        cohort = {name: [metrics["frases"], metrics["fonemas"]]
                  for name, metrics in evaluation["by_session"]["original"][first].items()}
        identity = {"input_signature": benchmark["inputs"]["input_signature"],
                    "data_signature": evaluation["data_signature"], "split": evaluation["split"],
                    "protocol_sha256": evaluation["protocol_sha256"], "cohort": cohort,
                    "runtime": benchmark["runtime"], "spec": benchmark["spec"],
                    "phones": evaluation["phones"], "blank_id": evaluation["blank_id"],
                    "control_seed": evaluation["control_seed"], "controls": evaluation["controls"],
                    "comparison_plan_sha256": (evaluation.get("comparison") or {}).get("plan_sha256")}
        if common is not None and identity != common:
            raise ValueError("No se pueden mezclar muestras, protocolos, controles, hardware o configuración de medidas.")
        common = identity
        for depth, profile in profiles.items():
            if profile["executed_layers"] != list(range(1, int(depth) + 1)):
                raise ValueError("El benchmark no confirma la parada del perfil.")
            latency = profile[cfg["etapa_latencia"]]
            if any(not isfinite(latency[key]) or latency[key] <= 0 for key in ("p50_ms", "p95_ms")):
                raise ValueError("Las latencias deben ser finitas y positivas.")
            expected_observations = len(benchmark["inputs"]["samples"]) / benchmark["spec"]["batch_size"] * benchmark["spec"]["repeticiones"]
            if latency["p95_ms"] < latency["p50_ms"] or latency["observations"] != expected_observations:
                raise ValueError("Los percentiles o el número de observaciones no corresponden al benchmark.")
            point = {"label": entry["etiqueta"], "exit": int(depth), "seed": evaluation["training_seed"],
                     "checkpoint_sha256": digest, "checkpoint_step": evaluation["checkpoint_step"],
                     "split": evaluation["split"], "latency_ms_p50": latency["p50_ms"],
                     "latency_ms_p95": latency["p95_ms"], "benchmark_observations": latency["observations"],
                     "per_by_control": {}}
            for control in evaluation["controls"]:
                metrics = evaluation["global"][control][depth]
                _valid_metrics(metrics)
                by_session = evaluation["by_session"][control][depth]
                if (set(by_session) != set(cohort)
                        or any(sum(value[key] for value in by_session.values()) != metrics[key]
                               for key in ("errores", "fonemas", "frases"))):
                    raise ValueError("Las sesiones no suman los recuentos globales.")
                for name, values in by_session.items():
                    _valid_metrics(values)
                    if [values["frases"], values["fonemas"]] != cohort[name]:
                        raise ValueError("Las condiciones no conservan la misma cohorte de frases/fonemas.")
                    sessions.append({"label": entry["etiqueta"], "exit": int(depth), "seed": point["seed"],
                                     "control": control, "session": name, **values})
                point["per_by_control"][control] = metrics["per"]
            points.append(point)
        if entry.get("historia"):
            history, source_history = _read(entry["historia"], "history")
            sources.append(source_history)
            if "evaluaciones_consola" in history:
                if (history["entrenamiento"]["pasos"] != evaluation["checkpoint_step"]
                        or history["entrenamiento"]["semilla"] != evaluation["training_seed"]
                        or evaluation["protocol_sha256"] is not None
                        or evaluation["split"] != "test"
                        or any(not row["completo"] or row["frases"] != evaluation["global"]["original"][first]["frases"]
                               for row in history["evaluaciones_consola"])):
                    raise ValueError("La curva histórica no corresponde al protocolo/paso de estos pesos.")
                rows = [{"step": row["paso"], "per": row["per_porcentaje_redondeado"] / 100,
                         "ctc_loss": row["ctc_redondeada"]} for row in history["evaluaciones_consola"]]
                if (not rows or rows[-1]["step"] != evaluation["checkpoint_step"]
                        or round(rows[-1]["per"] * 100, 2) != round(evaluation["global"]["original"][first]["per"] * 100, 2)):
                    raise ValueError("La última evaluación histórica no coincide con los pesos representados.")
                histories.append({"label": entry["etiqueta"], "split": "test monitorizado (consola redondeada)",
                                  "exit": int(first), "rows": rows})
            elif "validation_history" in history:
                if (history["best_val_step"] != evaluation["checkpoint_step"]
                        or history["config"]["experimento"]["seed"] != evaluation["training_seed"]
                        or history["config"]["protocolo"]["sha256"] != evaluation["protocol_sha256"]):
                    raise ValueError("La historia debe corresponder al checkpoint, semilla y protocolo seleccionados.")
                if evaluation["split"] == "val":
                    selected = (history["best_val_by_exit"] if history.get("best_val_by_exit") else {first: history["best_val"]})
                    for depth in profiles:
                        previous, actual = selected[depth], evaluation["global"]["original"][depth]
                        if (any(previous[key] != actual[key] for key in ("errores", "fonemas", "frases", "per"))
                                or not isclose(previous["ctc_loss"], actual["ctc_loss"], rel_tol=1e-6, abs_tol=1e-5)):
                            raise ValueError("La historia y la reevaluación de val no corresponden al mismo resultado.")
                if history.get("validation_history_by_exit"):
                    for depth in profiles:
                        histories.append({"label": f"{entry['etiqueta']} · semilla {evaluation['training_seed']}",
                                          "split": "val", "exit": int(depth),
                                          "rows": [{"step": row["step"], **row["metrics"][depth]}
                                                   for row in history["validation_history_by_exit"]]})
                else:
                    histories.append({"label": f"{entry['etiqueta']} · semilla {evaluation['training_seed']}", "split": "val", "exit": int(first),
                                      "rows": history["validation_history"]})
            else:
                raise ValueError("La historia no contiene curvas reconocibles.")
    for history in histories:
        steps = [row["step"] for row in history["rows"]]
        if not steps or steps != sorted(set(steps)):
            raise ValueError("Las curvas requieren pasos únicos en orden creciente.")
    return {"points": points, "sessions": sessions, "histories": histories,
            "conditions": common, "sources": sources, "interpretation": "pending_author_review"}


def render_figures(cfg: dict[str, Any], data: dict[str, Any], output: Path) -> list[str]:
    """Exporta figuras PNG/PDF y tablas; presenta medidas sin interpretar diferencias."""
    if not cfg["formatos"] or any(kind not in ("png", "pdf", "svg") for kind in cfg["formatos"]):
        raise ValueError("Los formatos admitidos son png/pdf/svg.")
    output.mkdir(parents=True, exist_ok=False)
    written = []

    def save(fig: Any, name: str) -> None:
        fig.suptitle(cfg["titulo"], fontsize=13)
        caption = cfg["alcance"]
        if name == "per_latencia":
            caption += f"\nLatencia: {data['conditions']['runtime']['hardware']} · lote {data['conditions']['spec']['batch_size']}"
        fig.supxlabel(caption, fontsize=8)
        for kind in cfg["formatos"]:
            filename = f"{name}.{kind}"
            fig.savefig(output / filename, dpi=cfg["dpi"], bbox_inches="tight", pad_inches=0.12)
            written.append(filename)
        plt.close(fig)

    points = data["points"]
    controls = data["conditions"]["controls"]
    colors = ["#245d80", "#c26b43", "#755b8e"]
    fig, ax = plt.subplots(figsize=(max(8, len(points) * 1.05), 5), layout="constrained")
    x = np.arange(len(points))
    width = 0.8 / len(controls)
    for index, control in enumerate(controls):
        values = [100 * point["per_by_control"][control] for point in points]
        bars = ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=control, color=colors[index % len(colors)])
        if len(points) <= 6:
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x, [f"{point['label']}\ncapa {point['exit']} · s{point['seed']}" for point in points], fontsize=8)
    ymax = max(100, max(100 * value for point in points for value in point["per_by_control"].values()) * 1.15)
    ax.set(ylabel="PER (%) · errores / fonemas", ylim=(0, ymax), title="Calidad y controles de señal")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    save(fig, "per_controles")

    names = list(data["conditions"]["cohort"])
    matrix, labels = [], []
    for point in points:
        for control in controls:
            rows = {row["session"]: row["per"] for row in data["sessions"]
                    if (row["label"], row["exit"], row["seed"], row["control"]) == (point["label"], point["exit"], point["seed"], control)}
            matrix.append([100 * rows[name] for name in names])
            labels.append(f"{point['label']} · c{point['exit']} · s{point['seed']} · {control}")
    fig, ax = plt.subplots(figsize=(12, max(3.8, 0.3 * len(matrix) + 2)), layout="constrained")
    shown = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(100, float(np.max(matrix))))
    ax.set_xticks(range(len(names)), [name.removeprefix("t12.2022.") for name in names], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set(title="PER por sesión y condición", xlabel="Sesión T12")
    fig.colorbar(shown, ax=ax, label="PER (%)")
    save(fig, "per_sesion")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    groups = list(dict.fromkeys((point["label"], point["exit"]) for point in points))
    palette = plt.get_cmap("tab10")
    for index, (label, depth) in enumerate(groups):
        rows = [point for point in points if (point["label"], point["exit"]) == (label, depth)]
        ax.scatter([point["latency_ms_p50"] for point in rows], [100 * point["per_by_control"]["original"] for point in rows],
                   label=f"{label} · capa {depth}", color=palette(index % 10), s=65)
        for point in rows:
            ax.annotate(f"s{point['seed']}", (point["latency_ms_p50"], 100 * point["per_by_control"]["original"]),
                        xytext=(5, 6), textcoords="offset points", fontsize=8)
    stage = "red preparada" if cfg["etapa_latencia"] == "forward" else "señal cruda en RAM → fonemas"
    batch_size = data["conditions"]["spec"]["batch_size"]
    ax.set(xlabel=f"Mediana de {stage} (ms/lote de {batch_size})", ylabel="PER global original (%)",
           title="Cada punto conserva su semilla de entrenamiento", xlim=(0, max(p["latency_ms_p50"] for p in points) * 1.25),
           ylim=(0, max(100, max(100 * p["per_by_control"]["original"] for p in points) * 1.15)))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "per_latencia")

    if data["histories"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
        for history in data["histories"]:
            rows = history["rows"]
            label = f"{history['label']} · c{history['exit']} · {history['split']}"
            axes[0].plot([row["step"] for row in rows], [100 * row["per"] for row in rows], marker="o", ms=3, label=label)
            axes[1].plot([row["step"] for row in rows], [row["ctc_loss"] for row in rows], marker="o", ms=3, label=label)
        axes[0].set(ylabel="PER (%)", title="Evaluación durante el entrenamiento")
        axes[1].set(ylabel="CTC media por frase", title="Pérdida de evaluación")
        for ax in axes:
            ax.set_xlabel("Actualizaciones")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        save(fig, "curvas_evaluacion")

    (output / "datos_figuras.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    rows = [{**{key: point[key] for key in ("label", "exit", "seed", "checkpoint_sha256", "checkpoint_step", "split", "latency_ms_p50", "latency_ms_p95")},
             **{f"per_{control}": value for control, value in point["per_by_control"].items()}} for point in points]
    with (output / "comparacion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return written


def main() -> int:
    """Lee el manifiesto explícito de figuras y exporta solo registros compatibles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/speech_figuras_referencia.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = collect_plot_data(cfg)
    files = render_figures(cfg, data, args.output)
    print(f"Figuras: {len(files)} archivos; puntos: {len(data['points'])}; salida: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
