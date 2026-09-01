# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for convolution-layer loss-sensitivity plotting."""

from types import SimpleNamespace

import torch
from torch import nn

from rfdetr.training.callbacks.conv_layer_loss import ConvLayerLossCallback


class _TinyConvModel(nn.Module):
    """Small model with two convolutional layers for callback tests."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.head = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the scalar training objective."""
        return self.head(torch.relu(self.encoder(inputs))).square().mean()


def test_callback_collects_conv_gradient_rms_and_writes_outputs(tmp_path) -> None:
    """Each trainable Conv2d is recorded and exported below ``output/loss``."""
    callback = ConvLayerLossCallback(tmp_path / "output")
    model = _TinyConvModel()
    trainer = SimpleNamespace(is_global_zero=True, current_epoch=0)

    callback.on_fit_start(trainer, model)
    loss = model(torch.randn(2, 3, 8, 8))
    loss.backward()
    callback.on_before_optimizer_step(trainer, model, torch.optim.SGD(model.parameters(), lr=0.1))
    callback.on_train_epoch_end(trainer, model)
    callback.on_fit_end(trainer, model)

    loss_dir = tmp_path / "output" / "loss"
    assert (loss_dir / "conv_layer_loss.csv").is_file()
    assert (loss_dir / "conv_layer_loss_heatmap.png").is_file()
    assert (loss_dir / "conv_layer_loss_overview.png").is_file()
    assert (loss_dir / "metric_definition.txt").is_file()
    layer_plots = sorted((loss_dir / "layers").glob("*.png"))
    assert len(layer_plots) == 2
    csv_text = (loss_dir / "conv_layer_loss.csv").read_text(encoding="utf-8")
    assert "encoder" in csv_text
    assert "head" in csv_text


def test_callback_ignores_non_global_process(tmp_path) -> None:
    """Only rank zero collects and writes plotting data."""
    callback = ConvLayerLossCallback(tmp_path / "output")
    model = _TinyConvModel()
    trainer = SimpleNamespace(is_global_zero=False, current_epoch=0)

    callback.on_fit_start(trainer, model)
    model(torch.randn(1, 3, 4, 4)).backward()
    callback.on_before_optimizer_step(trainer, model, torch.optim.SGD(model.parameters(), lr=0.1))
    callback.on_train_epoch_end(trainer, model)
    callback.on_fit_end(trainer, model)

    assert not (tmp_path / "output" / "loss").exists()
