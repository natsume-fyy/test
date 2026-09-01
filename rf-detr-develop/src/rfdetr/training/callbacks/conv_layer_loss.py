# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Plot the loss sensitivity of every convolutional layer during training."""

from __future__ import annotations

import csv
import math
import re
import warnings
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class ConvLayerLossCallback(Callback):
    """Track and plot the gradient RMS of each trainable ``Conv2d`` layer.

    A network has one scalar objective rather than an independent loss after
    every convolution.  The RMS of ``d(loss) / d(weight)`` is therefore used
    as a well-defined per-layer loss-sensitivity signal.  Values are averaged
    over optimizer steps in each epoch and written below ``output_dir/loss``.

    Args:
        output_dir: Root training output directory.
        overview_layers: Maximum number of curves in the compact overview.
    """

    def __init__(self, output_dir: str | Path, overview_layers: int = 12) -> None:
        super().__init__()
        self.loss_dir = Path(output_dir) / "loss"
        self.overview_layers = overview_layers
        self._layers: dict[str, nn.Conv2d] = {}
        self._epoch_sums: dict[str, float] = {}
        self._epoch_counts: dict[str, int] = {}
        self._history: dict[str, list[tuple[int, float]]] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Discover trainable convolutional layers before the first batch."""
        if not trainer.is_global_zero:
            return
        root_module = getattr(pl_module, "model", pl_module)
        self._layers = {
            name: module
            for name, module in root_module.named_modules()
            if isinstance(module, nn.Conv2d) and module.weight.requires_grad
        }
        self._history = {name: self._history.get(name, []) for name in self._layers}
        if not self._layers:
            warnings.warn(
                "ConvLayerLossCallback found no trainable Conv2d layers; no layer-loss plots will be generated.",
                UserWarning,
                stacklevel=2,
            )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Clear accumulators for the new epoch."""
        del pl_module
        if trainer.is_global_zero:
            self._epoch_sums.clear()
            self._epoch_counts.clear()

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Collect unscaled convolution-weight gradient RMS values."""
        del pl_module, optimizer
        if not trainer.is_global_zero or not self._layers:
            return

        names: list[str] = []
        values: list[torch.Tensor] = []
        for name, layer in self._layers.items():
            gradient = layer.weight.grad
            if gradient is None:
                continue
            names.append(name)
            values.append(gradient.detach().float().square().mean().sqrt())

        if not values:
            return
        host_values = torch.stack(values).cpu().tolist()
        for name, value in zip(names, host_values):
            if not math.isfinite(value):
                continue
            self._epoch_sums[name] = self._epoch_sums.get(name, 0.0) + value
            self._epoch_counts[name] = self._epoch_counts.get(name, 0) + 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Average the epoch observations and refresh aggregate artifacts."""
        del pl_module
        if not trainer.is_global_zero or not self._layers:
            return
        epoch = int(trainer.current_epoch) + 1
        for name in self._layers:
            count = self._epoch_counts.get(name, 0)
            if count:
                self._history.setdefault(name, []).append((epoch, self._epoch_sums[name] / count))
        self._export(include_layer_plots=False)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write final aggregate and per-layer plots."""
        del pl_module
        if trainer.is_global_zero and any(self._history.values()):
            self._export(include_layer_plots=True)

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        """Preserve all observations collected before an interrupted run."""
        del pl_module, exception
        if trainer.is_global_zero and any(self._history.values()):
            self._export(include_layer_plots=True)

    def state_dict(self) -> dict[str, Any]:
        """Return callback history for Lightning checkpoints."""
        return {"history": self._history}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore callback history from a Lightning checkpoint."""
        raw_history = state_dict.get("history", {})
        self._history = {
            str(name): [(int(epoch), float(value)) for epoch, value in values]
            for name, values in raw_history.items()
        }

    def _export(self, *, include_layer_plots: bool) -> None:
        """Write CSV and plot artifacts for the accumulated history."""
        self.loss_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv()
        (self.loss_dir / "metric_definition.txt").write_text(
            "Each value is the epoch mean of RMS(d(total_loss)/d(conv_weight)) "
            "measured immediately before an optimizer step. It is a per-layer "
            "loss-sensitivity signal, not an independent auxiliary loss.\n",
            encoding="utf-8",
        )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            warnings.warn(
                f"Convolution-layer plots were skipped because a plotting dependency is unavailable: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return

        layer_names = [name for name in self._layers if self._history.get(name)]
        if not layer_names:
            return
        epochs = sorted({epoch for name in layer_names for epoch, _ in self._history[name]})
        values_by_layer = {name: dict(self._history[name]) for name in layer_names}
        matrix = np.asarray(
            [[values_by_layer[name].get(epoch, np.nan) for epoch in epochs] for name in layer_names],
            dtype=float,
        )

        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        floor = float(positive.min()) if positive.size else 1e-30
        log_matrix = np.log10(np.where(matrix > 0, matrix, floor))
        figure_height = min(30.0, max(5.0, 0.28 * len(layer_names)))
        fig, ax = plt.subplots(figsize=(12, figure_height))
        image = ax.imshow(log_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title("Convolution-layer loss sensitivity")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Conv2d layer")
        ax.set_xticks(range(len(epochs)), epochs)
        ax.set_yticks(range(len(layer_names)), layer_names, fontsize=7)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("log10 gradient RMS")
        fig.tight_layout()
        fig.savefig(self.loss_dir / "conv_layer_loss_heatmap.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        ranking = sorted(
            layer_names,
            key=lambda name: self._history[name][-1][1],
            reverse=True,
        )[: self.overview_layers]
        fig, ax = plt.subplots(figsize=(12, 7))
        for name in ranking:
            layer_epochs, layer_values = zip(*self._history[name])
            ax.plot(layer_epochs, layer_values, marker="o", linewidth=1.5, markersize=3, label=name)
        ax.set_title(f"Top {len(ranking)} convolution-layer loss sensitivities")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient RMS")
        if any(value > 0 for name in ranking for _, value in self._history[name]):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        fig.savefig(self.loss_dir / "conv_layer_loss_overview.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        if include_layer_plots:
            layer_dir = self.loss_dir / "layers"
            layer_dir.mkdir(parents=True, exist_ok=True)
            for index, name in enumerate(layer_names):
                layer_epochs, layer_values = zip(*self._history[name])
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(layer_epochs, layer_values, marker="o", linewidth=1.8, markersize=4)
                ax.set_title(name)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Gradient RMS (loss sensitivity)")
                if any(value > 0 for value in layer_values):
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:100] or "conv"
                fig.savefig(layer_dir / f"{index:03d}_{safe_name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

        logger.info("Convolution-layer loss plots saved to %s", self.loss_dir)

    def _write_csv(self) -> None:
        """Write one epoch row and one column per convolutional layer."""
        layer_names = [name for name in self._layers if self._history.get(name)]
        epochs = sorted({epoch for name in layer_names for epoch, _ in self._history[name]})
        values_by_layer = {name: dict(self._history[name]) for name in layer_names}
        with (self.loss_dir / "conv_layer_loss.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", *layer_names])
            for epoch in epochs:
                writer.writerow([epoch, *(values_by_layer[name].get(epoch, "") for name in layer_names)])
