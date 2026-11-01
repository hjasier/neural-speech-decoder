"""Bucle T12: actualización, validación, checkpoints y reanudación."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.preprocessing import GaussianSmoothing
from src.data.speech_dataset import SpeechBatch
from src.models.speech_gru import SpeechGRU
from src.training import speech
from src.training.checkpoints import save_checkpoint
from src.training.optim import TensorFlowAdam
from src.training.speech_resume import (
    flush_metrics,
    restore_training,
    runtime_state,
    save_training,
    validate_resume_config,
)
from src.utils.seed import restaurar_azar
from src.utils.tracking import registrar


class SpeechTrainer:
    """Conserva el estado de una ejecución y separa las tareas del entrenamiento."""

    def __init__(
        self,
        model: SpeechGRU,
        train_loader: DataLoader,
        evaluation_loader: DataLoader,
        cfg: dict[str, Any],
        run: Any,
        smoother: GaussianSmoothing,
        device: torch.device,
        output: Path,
        resume_state: dict[str, Any] | None,
    ) -> None:
        """Prepara el optimizador y recupera el estado cuando se solicita."""
        self.model, self.cfg, self.run = model, cfg, run
        self.train_loader, self.evaluation_loader = train_loader, evaluation_loader
        self.smoother, self.device, self.output = smoother, device, output
        self.resume_state = resume_state
        self._configure()
        self._restore()
        self.start = perf_counter()
        self.training_seconds = (
            self.progress["training_seconds"] if self.progress is not None else 0.0
        )
        self.time_limit = self.training.get("max_segundos")
        self.stopped_for_time = False
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.stopped_for_session = False
        self.metrics: dict[str, float | int] = (
            self.progress["metrics"] if self.progress is not None else {}
        )

    def _configure(self) -> None:
        self.training = self.cfg["entrenamiento"]
        self.weights = speech._ctc_weights(self.cfg)
        self.best_exit_metrics, self.exit_history = None, []
        if self.weights is not None:
            self.run.summary.update(
                {
                    "seleccion/capa": self.cfg["modelo"]["capas"],
                    **{
                        f"ctc/peso_capa_{depth}": value
                        for depth, value in self.weights.items()
                    },
                }
            )
        self.split = "val" if self.cfg.get("protocolo") else "test"
        self.best_metrics, self.best_step = None, 0
        self.history = []
        self.criterion = speech._criterion(self.cfg)
        self.optimizer = TensorFlowAdam(
            self.model.parameters(),
            lr=speech._learning_rate(self.cfg, 0),
            betas=tuple(self.training["betas"]),
            eps=self.training["epsilon"],
        )
        self.resumable = bool(self.cfg.get("reanudacion"))
        self.initial_step, self.previous_seconds = 0, 0.0
        self.metric_history: list[dict[str, Any]] = []
        self.logical_id = (
            self.resume_state["tracking"]["logical_id"]
            if self.resume_state is not None
            else self.run.id
        )
        self.progress = None
        self.segments: list[dict[str, Any]] = []

    def _restore(self) -> None:
        if self.resumable:
            validate_resume_config(self.cfg)
            if self.resume_state is not None:
                if int(getattr(self.run, "step", 0)) > self.resume_state["step"] + 1:
                    raise ValueError(
                        "W&B está adelantado al checkpoint; utiliza el estado más reciente."
                    )
                self.progress = restore_training(
                    self.resume_state,
                    self.model,
                    self.optimizer,
                    self.train_loader,
                    self.evaluation_loader,
                    self.device,
                    self.output,
                )
                self.initial_step = self.resume_state["step"]
                self.best_metrics, self.best_step = (
                    self.progress["best_metrics"],
                    self.progress["best_step"],
                )
                self.best_exit_metrics, self.exit_history = (
                    self.progress["best_exit_metrics"],
                    self.progress["exit_history"],
                )
                self.history, self.metric_history = (
                    self.progress["history"],
                    self.progress["metric_history"],
                )
                self.previous_seconds = self.progress["loop_seconds"]
                self.segments = self.progress["segments"]
                try:
                    flush_metrics(self.run, self.metric_history, self.initial_step)
                finally:
                    restaurar_azar(self.resume_state["rng"], self.device)
            self.segments.append(
                {
                    "from_step": self.initial_step,
                    "tracking_id": self.run.id,
                    "runtime": runtime_state(self.device),
                }
            )

    def _emit(self, values: dict[str, Any], step: int) -> None:
        if not self.resumable:
            registrar(self.run, values, step)
        elif self.metric_history and self.metric_history[-1]["step"] == step:
            self.metric_history[-1]["values"].update(values)
        else:
            self.metric_history.append({"step": step, "values": dict(values)})

    def _persist(self) -> None:
        save_training(
            self.output / "entrenamiento.pt",
            self.model,
            self.optimizer,
            self.cfg,
            self.train_loader,
            self.evaluation_loader,
            self.device,
            {
                "best_metrics": self.best_metrics,
                "best_step": self.best_step,
                "best_exit_metrics": self.best_exit_metrics,
                "exit_history": self.exit_history,
                "history": self.history,
                "metric_history": self.metric_history,
                "metrics": self.metrics,
                "training_seconds": self.training_seconds,
                "loop_seconds": self.previous_seconds + perf_counter() - self.start,
                "segments": self.segments,
            },
            self.run,
            self.logical_id,
        )

    def _iterator(self) -> Iterator[SpeechBatch]:
        if self.resumable and self.resume_state is None:
            self._persist()  # También se puede recuperar un corte antes de la primera actualización.
        iterator = (
            iter(self.train_loader)
            if self.initial_step < self.training["pasos"]
            else iter(())
        )
        if self.resumable and self.resume_state is not None and self.initial_step:
            # DataLoader consume una semilla al crear el iterador, incluso con workers=0.
            # Ese consumo ya ocurrió en el proceso original; no debe repetirse.
            self.train_loader.generator.set_state(self.resume_state["loaders"]["train"])
            restaurar_azar(self.resume_state["rng"], self.device)
        self.step = self.initial_step
        return iterator

    def _train_step(self, batch: SpeechBatch) -> dict[str, Any]:
        step_start = perf_counter()
        self.model.train()
        learning_rate = speech._learning_rate(self.cfg, self.step - 1)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        components: dict[int, float] = {}
        backward_options = (
            {"component_losses": components} if self.weights is not None else {}
        )
        ctc_loss = sum(
            speech._backward_microbatch(
                self.model,
                microbatch,
                self.cfg,
                self.smoother,
                self.criterion,
                self.device,
                len(batch.references),
                **backward_options,
            )
            for microbatch in batch.microbatches(self.training["micro_batch_size"])
        )
        regularization = speech._regularization(
            self.model, batch.session_ids, self.training["l2"]
        )
        if not bool(torch.isfinite(regularization)):
            raise FloatingPointError("La pérdida total no es finita.")
        # L2 y clipping se aplican una vez por actualización de 64 frases.
        regularization.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.training["clip_grad_norm"],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        # item sincroniza también el trabajo CUDA antes de medir el tiempo.
        logged = {
            "train/ctc_loss": ctc_loss,
            "train/l2": regularization.item(),
            "train/loss": ctc_loss + regularization.item(),
            "train/grad_norm": gradient_norm.item(),
            "train/learning_rate": learning_rate,
            **speech._memory_metrics(self.device),
            **{
                f"train/capa_{depth}/ctc_loss": value
                for depth, value in components.items()
            },
        }
        self.training_seconds += perf_counter() - step_start
        logged["train/segundos_acumulados"] = self.training_seconds
        self._emit(logged, self.step)
        return logged

    def _print_step(self, logged: dict[str, Any]) -> None:
        if (
            self.step == 1
            or self.step % self.training["imprimir_cada"] == 0
            or self.step == self.training["pasos"]
        ):
            memory = (
                f"  pico CUDA asignado={logged['cuda/pico_asignado_mib']:.0f} MiB"
                f"  reservado={logged['cuda/pico_reservado_mib']:.0f} MiB"
                if self.device.type == "cuda"
                else ""
            )
            print(
                f"paso {self.step}/{self.training['pasos']}  ctc={logged['train/ctc_loss']:.4f}  "
                f"l2={logged['train/l2']:.4f}  lr={logged['train/learning_rate']:.6f}{memory}",
                flush=True,
            )

    def _evaluate(self) -> None:
        exit_metrics = None
        per_exit_logged = {}
        if self.weights is None:
            self.metrics = speech.evaluar_habla(
                self.model, self.evaluation_loader, self.cfg, self.smoother, self.device
            )
        else:
            exit_metrics, seconds = speech.evaluar_salidas(
                self.model, self.evaluation_loader, self.cfg, self.smoother, self.device
            )
            self.metrics = {
                **exit_metrics[self.cfg["modelo"]["capas"]],
                "segundos": seconds,
            }
            per_exit_logged = {
                f"{self.split}/capa_{depth}/{key}": value
                for depth, values in exit_metrics.items()
                for key, value in values.items()
            }
            per_exit_logged[f"{self.split}/segundos_todas_salidas"] = seconds
            self.exit_history.append(
                {
                    "step": self.step,
                    "metrics": exit_metrics,
                    "segundos_conjuntos": seconds,
                }
            )
            print(
                "salidas: "
                + "  ".join(
                    f"{depth}: PER={values['per']:.2%}"
                    for depth, values in exit_metrics.items()
                ),
                flush=True,
            )
        if self.split == "val" and not self.metrics["completo"]:
            raise ValueError("La selección de checkpoint requiere toda la validación.")
        self._emit(
            {
                **{f"{self.split}/{key}": value for key, value in self.metrics.items()},
                **per_exit_logged,
            },
            self.step,
        )
        print(
            f"{self.split}: PER={self.metrics['per']:.2%}  CTC={self.metrics['ctc_loss']:.4f}  "
            f"frases={self.metrics['frases']}  completo={bool(self.metrics['completo'])}",
            flush=True,
        )
        checkpoint_options = (
            {"exit_metrics": exit_metrics} if self.weights is not None else {}
        )
        save_checkpoint(
            self.output / "ultimo.pt",
            self.model,
            self.cfg,
            self.train_loader.dataset,
            self.step,
            self.metrics,
            split=self.split,
            **checkpoint_options,
        )
        if self.split == "val":
            self.history.append({"step": self.step, **self.metrics})
            if (
                self.best_metrics is None
                or self.metrics["per"] < self.best_metrics["per"]
            ):
                self.best_metrics, self.best_step = dict(self.metrics), self.step
                self.best_exit_metrics = exit_metrics
                save_checkpoint(
                    self.output / "mejor_val.pt",
                    self.model,
                    self.cfg,
                    self.train_loader.dataset,
                    self.step,
                    self.metrics,
                    split="val",
                    **checkpoint_options,
                )

    def _write_result(self, elapsed: float) -> None:
        (self.output / "resultado.json").write_text(
            json.dumps(
                {
                    "config": self.cfg,
                    "completed_steps": self.step,
                    "best_val_step": self.best_step,
                    "best_val": self.best_metrics,
                    "last_val": self.metrics,
                    "validation_history": self.history,
                    "checkpoint": str(self.output / "mejor_val.pt"),
                    "test_evaluated": False,
                    "stopped_for_time": self.stopped_for_time,
                    "hardware": torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda"
                    else "CPU",
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "training_seconds": self.training_seconds,
                    "loop_seconds": elapsed,
                    "cuda_memory": speech._memory_metrics(self.device),
                    "model_parameters": sum(
                        parameter.numel() for parameter in self.model.parameters()
                    ),
                    "run_id": self.logical_id,
                    **(
                        {
                            "resumable": True,
                            "resume_checkpoint": str(self.output / "entrenamiento.pt"),
                            "resumed_from_step": self.initial_step,
                            "stopped_for_session_steps": self.stopped_for_session,
                            "training_complete": self.step == self.training["pasos"],
                            "tracking_segment_id": self.run.id,
                            "last_val_step": self.history[-1]["step"],
                            "segments": self.segments,
                        }
                        if self.resumable
                        else {}
                    ),
                    **(
                        {
                            "ctc_weights": self.weights,
                            "selection_exit": self.cfg["modelo"]["capas"],
                            "best_val_by_exit": self.best_exit_metrics,
                            "validation_history_by_exit": self.exit_history,
                        }
                        if self.weights is not None
                        else {}
                    ),
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _finish(self) -> dict[str, float | int]:
        if not self.metrics:
            raise ValueError("No se completó ningún paso con evaluación.")
        self.run.summary.update(
            {
                "entrenamiento/pasos": self.step,
                "entrenamiento/segundos": self.training_seconds,
                "entrenamiento/pasos_previstos": self.training["pasos"],
                "entrenamiento/parada_por_tiempo": self.stopped_for_time,
                "ejecucion/segundos": self.previous_seconds
                + perf_counter()
                - self.start,
                "checkpoint": str(
                    self.output
                    / ("mejor_val.pt" if self.split == "val" else "ultimo.pt")
                ),
                **speech._memory_metrics(self.device),
                **{
                    f"{self.split}_final/{key}": value
                    for key, value in self.metrics.items()
                },
            }
        )
        if self.split == "val":
            elapsed = self.previous_seconds + perf_counter() - self.start
            if self.best_exit_metrics is not None:
                self.run.summary.update(
                    {
                        f"val_mejor/capa_{depth}/{key}": value
                        for depth, values in self.best_exit_metrics.items()
                        for key, value in values.items()
                    }
                )
            self.run.summary.update(
                {
                    "seleccion/metrica": "val/per",
                    "seleccion/paso": self.best_step,
                    "seleccion/empates": "primer checkpoint",
                    "protocolo/sha256": self.cfg["protocolo"]["sha256"],
                    **{
                        f"val_mejor/{key}": value
                        for key, value in self.best_metrics.items()
                    },
                }
            )
            self._write_result(elapsed)
            print(
                f"checkpoint seleccionado: {self.output / 'mejor_val.pt'} (val PER; paso {self.best_step})",
                flush=True,
            )
            return self.best_metrics
        print(f"checkpoint: {self.output / 'ultimo.pt'}", flush=True)
        return self.metrics

    def fit(self) -> dict[str, float | int]:
        """Entrena hasta completar el presupuesto o guardar una pausa de sesión."""
        for self.step, batch in enumerate(
            self._iterator(), start=self.initial_step + 1
        ):
            logged = self._train_step(batch)
            self._print_step(logged)
            time_reached = (
                self.time_limit is not None
                and perf_counter() - self.start >= self.time_limit
            )
            evaluation_due = (
                (self.step - 1) % self.cfg["evaluacion"]["cada_pasos"] == 0
                or self.step == self.training["pasos"]
                or time_reached
            )
            if evaluation_due:
                self._evaluate()
            self.stopped_for_time = (
                evaluation_due
                and self.step < self.training["pasos"]
                and self.time_limit is not None
                and perf_counter() - self.start >= self.time_limit
            )
            if self.resumable:
                session_limit = self.cfg["reanudacion"].get("max_pasos_sesion")
                self.stopped_for_session = (
                    self.step < self.training["pasos"]
                    and session_limit is not None
                    and self.step - self.initial_step >= session_limit
                )
                if (
                    (self.step - 1) % self.cfg["reanudacion"]["cada_pasos"] == 0
                    or self.step == self.training["pasos"]
                    or self.stopped_for_time
                    or self.stopped_for_session
                ):
                    self._persist()
            if self.stopped_for_time:
                print(
                    f"Límite de {self.time_limit:g} s alcanzado; parada tras el paso {self.step}, "
                    "con evaluación y checkpoint guardados.",
                    flush=True,
                )
                break
            if self.stopped_for_session:
                print(
                    f"Pausa de sesión tras el paso {self.step}; estado completo en {self.output / 'entrenamiento.pt'}.",
                    flush=True,
                )
                break
        return self._finish()
