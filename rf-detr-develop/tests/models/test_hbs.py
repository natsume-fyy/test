# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for HBS background smoothing with foreground high-frequency compensation."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRMediumConfig, RFDETRSegMediumConfig, TrainConfig
from rfdetr.models.hbs import HBS, BackgroundSmoothingBlock, SpatialForegroundAttention
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


class _ZeroSmoother(nn.Module):
    """Return a deterministic fully smoothed tensor for branch-composition tests."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return zeros with the input shape."""
        return torch.zeros_like(features)


def test_background_smoothing_block_preserves_shape_and_gradients() -> None:
    """The SET denoising block must preserve shape and remain trainable."""
    block = BackgroundSmoothingBlock(channels=8, reduction=4, kernel_size=3)
    features = torch.randn(2, 8, 7, 9, requires_grad=True)

    block(features).sum().backward()

    assert block(features).shape == features.shape
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in block.parameters())


def test_spatial_attention_is_lightweight_and_bounded() -> None:
    """The foreground predictor must emit one bounded spatial gate per feature map."""
    attention = SpatialForegroundAttention(kernel_size=3)
    residual = torch.randn(2, 8, 5, 7)

    gate = attention(residual)

    assert gate.shape == (2, 1, 5, 7)
    assert torch.all((gate >= 0) & (gate <= 1))
    assert sum(parameter.numel() for parameter in attention.parameters()) == 19


def test_hbs_reinjects_residual_only_inside_foreground() -> None:
    """Background stays smoothed while attention restores part of F-HBS(F) in a box."""
    hbs = HBS(
        channels=1,
        kernel_sizes=[3],
        reduction=1,
        foreground_scale=0.4,
        attention_kernel_size=3,
    )
    hbs.denoisers[0] = _ZeroSmoother()
    with torch.no_grad():
        hbs.foreground_attentions[0].spatial.weight.zero_()
        hbs.foreground_attentions[0].spatial.bias.zero_()

    features = [torch.ones(1, 1, 6, 6)]
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 1 / 3, 1 / 3]])}]
    output = hbs(features, targets)[0]

    # sigmoid(0) == 0.5, hence 0 + 0.4 * 0.5 * (1 - 0) == 0.2 in the 2x2 foreground box.
    torch.testing.assert_close(output[:, :, 2:4, 2:4], torch.full((1, 1, 2, 2), 0.2))
    assert torch.count_nonzero(output[:, :, :2, :]) == 0
    assert torch.count_nonzero(output[:, :, 4:, :]) == 0


def test_hbs_preserves_padding() -> None:
    """Padded feature positions must bypass both smoothing and compensation."""
    hbs = HBS(channels=1, kernel_sizes=[3], reduction=1)
    hbs.denoisers[0] = _ZeroSmoother()
    features = [torch.ones(1, 1, 6, 6)]
    padding_mask = torch.zeros(1, 6, 6, dtype=torch.bool)
    padding_mask[:, 4:, :] = True

    output = hbs(features, [{"boxes": torch.empty(0, 4)}], [padding_mask])[0]

    assert torch.count_nonzero(output[:, :, :4, :]) == 0
    torch.testing.assert_close(output[:, :, 4:, :], features[0][:, :, 4:, :])


def test_hbs_config_is_opt_in_and_detection_only() -> None:
    """HBS configuration must be forwarded without changing the default architecture."""
    default_config = RFDETRMediumConfig()
    hbs_config = RFDETRMediumConfig(
        hbs_enabled=True,
        hbs_reduction=8,
        hbs_foreground_scale=0.25,
        hbs_attention_kernel_size=5,
    )

    assert default_config.hbs_enabled is False
    namespace = _namespace_from_configs(hbs_config, TrainConfig(dataset_dir="."))
    assert namespace.hbs_enabled is True
    assert namespace.hbs_reduction == 8
    assert namespace.hbs_foreground_scale == 0.25
    assert namespace.hbs_attention_kernel_size == 5
    with pytest.raises(ValueError, match="HBS currently supports detection models only"):
        RFDETRSegMediumConfig(hbs_enabled=True)


def test_hbs_config_validates_attention_and_loss_ranges() -> None:
    """Invalid spatial kernels, restoration fractions, and loss weights must fail early."""
    with pytest.raises(ValueError, match="hbs_attention_kernel_size must be odd"):
        RFDETRMediumConfig(hbs_attention_kernel_size=4)
    with pytest.raises(ValueError):
        RFDETRMediumConfig(hbs_foreground_scale=1.1)
    with pytest.raises(ValueError):
        TrainConfig(dataset_dir=".", hbs_loss_coef=-0.1)


def test_lwdetr_emits_hbs_outputs_only_during_training() -> None:
    """LWDETR must run the shared DETR head twice only for HBS training batches."""
    batch_size = 1
    num_queries = 2
    hidden_dim = 4
    features = [
        NestedTensor(
            torch.ones(batch_size, hidden_dim, 4, 4),
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
        )
    ]
    positions = [torch.zeros(batch_size, hidden_dim, 4, 4)]
    backbone = MagicMock(return_value=(features, positions, None))
    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = (
        torch.zeros(1, batch_size, num_queries, hidden_dim),
        torch.zeros(1, batch_size, num_queries, 4),
        torch.zeros(batch_size, num_queries, hidden_dim),
        torch.zeros(batch_size, num_queries, 4),
    )
    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=3,
        num_queries=num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        hbs_enabled=True,
        hbs_reduction=2,
        hbs_kernel_sizes=[3],
        hbs_foreground_scale=0.2,
        hbs_attention_kernel_size=5,
    )
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]])}]

    model.train()
    train_outputs = model(torch.ones(batch_size, 3, 8, 8), targets)
    model.eval()
    eval_outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert "hbs_outputs" in train_outputs
    assert "hbs_outputs" not in eval_outputs
    assert transformer.call_count == 3
