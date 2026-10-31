"""Presentación descriptiva de historias PER verificadas, sin cargar modelos."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml


def collect_training_round(cfg: dict[str, Any]) -> dict[str, Any]:
    """Vincula cada PER e historia al registro de su entrenamiento y selección.

    Solo presenta calidad: el benchmark completo conserva su validador
    independiente en plot_speech_results.py, que exige los JSON originales.
    """
    source = Path(cfg["resumen"])
    payload = source.read_bytes()
    report = json.loads(payload)
    conditions = report["conditions"]
    records, sources = {}, {str(source): hashlib.sha256(payload).hexdigest()}
    for name, expected in report["sources_sha256"].items():
        payload = Path(name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f"Fuente modificada desde el resumen: {name}")
        record = json.loads(payload)
        training = record["runs"]["training"]
        summary = record["notebook_training_summary"]
        variant = summary["comparison"]["variante"]
        if (variant in records or training["state"] != "finished"
                or summary["seed"] != conditions["seed"]
                or summary["protocol_sha256"] != conditions["protocol_sha256"]
                or summary["completed_steps"] != conditions["training_steps"]
                or summary["test_evaluated"] is not False):
            raise ValueError("La ronda requiere entrenamientos únicos y completos del mismo protocolo.")
        final_prefix = f"val/capa_{summary['model_layers']}/" if len(summary["available_exits"]) > 1 else "val/"
        history = training["validation_history"]
        steps = [row["_step"] for row in history]
        if not steps or steps != sorted(set(steps)) or steps[-1] != summary["completed_steps"]:
            raise ValueError("El historial requiere pasos completos, únicos y ordenados.")
        selected = min(history, key=lambda row: row[final_prefix + "per"])
        if (selected["_step"] != summary["best_val_step"]
                or selected != training["selected_validation"]):
            raise ValueError("La selección debe ser el primer mínimo de PER de la salida final.")
        records[variant] = record
        sources[name] = digest

    points = {(point["variant"], point["exit"]): point for point in report["points"]}
    requested = [(p["variante"], p["salida"]) for p in cfg["perfiles"]]
    if (len(points) != len(report["points"]) or len(set(requested)) != len(requested)
            or set(requested) != set(points) or len(records) != report["training_runs"]):
        raise ValueError("Las figuras deben conservar todos los perfiles y sus entrenamientos.")
    series = []
    for profile in cfg["perfiles"]:
        variant, depth = profile["variante"], profile["salida"]
        point = points[variant, depth]
        record = records[variant]
        summary = record["notebook_training_summary"]
        training = record["runs"]["training"]
        prefix = f"val/capa_{depth}/" if len(summary["available_exits"]) > 1 else "val/"
        if (point["seed"] != summary["seed"] or point["training_run_id"] != training["id"]
                or summary["run_id"] != training["id"] or depth not in summary["available_exits"]
                or point["checkpoint_step"] != summary["best_val_step"]
                or point["checkpoint_sha256"] != summary["checkpoint_sha256"]):
            raise ValueError("El perfil debe corresponder a los pesos, paso y run del resumen.")
        rows = []
        for row in training["validation_history"]:
            per, errors = row[prefix + "per"], row[prefix + "errores"]
            phones, trials = row[prefix + "fonemas"], row[prefix + "frases"]
            if (row[prefix + "completo"] != 1 or not isfinite(per) or per < 0
                    or type(errors) is not int or errors < 0
                    or phones != conditions["validation_phonemes"]
                    or trials != conditions["validation_trials"]
                    or not isclose(per, errors / phones, abs_tol=1e-12)):
                raise ValueError("PER requiere recuentos consistentes y toda la validación.")
            rows.append({"step": row["_step"], "per": per, "errors": errors,
                         "phonemes": phones, "trials": trials})
        selected = next(row for row in rows if row["step"] == point["checkpoint_step"])
        if (selected["errors"] != point["errors"] or selected["phonemes"] != point["phonemes"]
                or selected["trials"] != point["trials"]
                or not isclose(selected["per"], point["per"], abs_tol=1e-12)):
            raise ValueError("La observación seleccionada no coincide con su historia.")
        series.append({"profile": profile, "rows": rows, "selected": selected,
                       "checkpoint_sha256": point["checkpoint_sha256"], "run_id": training["id"]})
    return {"sources_sha256": sources, "conditions": conditions, "series": series,
            "scope": "validation PER only; no latency join or scientific interpretation"}


def render_training_round(cfg: dict[str, Any], data: dict[str, Any], output: Path) -> list[str]:
    """Exporta curvas sin suavizar y PER de los pesos seleccionados, con fuentes."""
    if not cfg["formatos"] or set(cfg["formatos"]) - {"png", "svg"}:
        raise ValueError("Formatos admitidos: png y svg.")
    output.mkdir(parents=True, exist_ok=False)
    files = []

    def save(fig: Any, stem: str) -> None:
        for kind in cfg["formatos"]:
            name = f"{stem}.{kind}"
            target = output / name
            fig.savefig(target, dpi=cfg["dpi"], bbox_inches="tight")
            if kind == "svg":
                target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())
                                  + "\n", encoding="utf-8")
            files.append(name)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.suptitle(cfg["titulo"] + " · evolución de PER", fontsize=14)
    for item in data["series"]:
        profile, rows = item["profile"], item["rows"]
        for ax in axes:
            ax.plot([r["step"] for r in rows], [100 * r["per"] for r in rows],
                    label=profile["etiqueta"], color=profile["color"],
                    linestyle=profile["estilo"], linewidth=1.5)
            ax.scatter(item["selected"]["step"], 100 * item["selected"]["per"],
                       color=profile["color"], marker="o", s=25, zorder=3)
    for ax, limits, title in zip(axes, (cfg["limites_per"], cfg["limites_detalle"]),
                                ("Escala general", "Detalle: eje vertical ampliado"), strict=True):
        ax.set(xlabel="Actualizaciones del optimizador", ylabel="PER de validación (%)",
               title=title, ylim=limits)
        ax.grid(alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.08), frameon=False, fontsize=9)
    fig.text(0.5, 0.055, cfg["alcance"], ha="center", fontsize=9)
    fig.text(0.5, 0.025, "Sin suavizado. Los puntos indican los pesos seleccionados; no el mínimo de cada cabeza.",
             ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.28, top=0.85, wspace=0.28)
    save(fig, "curvas_per")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.suptitle(cfg["titulo"] + " · pesos seleccionados", fontsize=14)
    for index, item in enumerate(data["series"]):
        value = item["selected"]
        ax.barh(index, 100 * value["per"], color=item["profile"]["color"], height=0.6)
        ax.text(100 * value["per"] + 0.6, index,
                f"{100 * value['per']:.2f} % · {value['errors']:,}/{value['phonemes']:,}",
                va="center", fontsize=9)
    ax.set_yticks(range(len(data["series"])), [s["profile"]["etiqueta"] for s in data["series"]])
    ax.invert_yaxis()
    ax.set(xlabel="PER de validación (%) · eje desde cero", xlim=(0, cfg["limites_detalle"][1]))
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.065, cfg["alcance"], ha="center", fontsize=9)
    fig.text(0.5, 0.025, cfg["nota"], ha="center", fontsize=8)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.19, top=0.86)
    save(fig, "per_seleccionado")
    (output / "datos_figuras.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return files


def main() -> int:
    """Genera la presentación PER de una ronda desde registros ya verificados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/speech_figuras_semilla0.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = collect_training_round(cfg)
    files = render_training_round(cfg, data, args.output)
    receipt = {"config": str(args.config), "script": str(Path(__file__).relative_to(Path.cwd())),
               "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.config, Path(__file__), *(args.output / name for name in files),
                                    args.output / "datos_figuras.json")}}
    (args.output / "procedencia.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Presentación PER: {len(files)} figuras, {len(data['series'])} perfiles; {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
