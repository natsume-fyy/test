# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for SET HBS background-only smoothing."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRMediumConfig, RFDETRSegMediumConfig, TrainConfig
from rfdetr.models.hbs import HBS, BackgroundSmoothingBlock
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


class _ConstantSmoother(nn.Module):
    """Return a deterministic tensor for branch-composition tests."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a constant while retaining a zero-valued input gradient path."""
        return features * 0 + 3


def test_background_smoothing_block_preserves_shape_and_gradients() -> None:
    """The SET denoising block must preserve shape and remain trainable."""
    block = BackgroundSmoothingBlock(channels=8, reduction=4, kernel_size=3)
    features = torch.randn(2, 8, 7, 9, requires_grad=True)

    block(features).sum().backward()

    assert block(features).shape == features.shape
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in block.parameters())


def test_hbs_smooths_only_background_and_copies_foreground_exactly() -> None:
    """The official SET composition must never alter features inside GT boxes."""
    hbs = HBS(channels=1, kernel_sizes=[3], reduction=1)
    hbs.denoisers[0] = _ConstantSmoother()

    features = [torch.ones(1, 1, 6, 6)]
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}]
    output = hbs(features, targets)[0]

    torch.testing.assert_close(output[:, :, 2:4, 2:4], features[0][:, :, 2:4, 2:4])
    torch.testing.assert_close(output[:, :, :2, :], torch.full((1, 1, 2, 6), 3.0))
    torch.testing.assert_close(output[:, :, 4:, :], torch.full((1, 1, 2, 6), 3.0))


def test_hbs_masks_foreground_before_denoising() -> None:
    """Foreground activations must not leak into background through HBS convolutions."""
    hbs = HBS(channels=1, kernel_sizes=[3], reduction=1)
    recorded_input: list[torch.Tensor] = []

    class _Recorder(nn.Module):
        """Record the denoiser input and pass it through."""

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """Store a detached input snapshot."""
            recorded_input.append(features.detach().clone())
            return features

    hbs.denoisers[0] = _Recorder()
    features = [torch.ones(1, 1, 6, 6)]
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}]

    hbs(features, targets)

    assert torch.count_nonzero(recorded_input[0][:, :, 2:4, 2:4]) == 0
    assert torch.all(recorded_input[0][:, :, :2, :] == 1)


def test_hbs_preserves_padding() -> None:
    """Padded feature positions must bypass background smoothing."""
    hbs = HBS(channels=1, kernel_sizes=[3], reduction=1)
    hbs.denoisers[0] = _ConstantSmoother()
    features = [torch.ones(1, 1, 6, 6)]
    padding_mask = torch.zeros(1, 6, 6, dtype=torch.bool)
    padding_mask[:, 4:, :] = True

    output = hbs(features, [{"boxes": torch.empty(0, 4)}], [padding_mask])[0]

    torch.testing.assert_close(output[:, :, :4, :], torch.full((1, 1, 4, 6), 3.0))
    torch.testing.assert_close(output[:, :, 4:, :], features[0][:, :, 4:, :])


def test_hbs_config_is_opt_in_and_detection_only() -> None:
    """HBS configuration must be forwarded without changing the default architecture."""
    default_config = RFDETRMediumConfig()
    hbs_config = RFDETRMediumConfig(
        hbs_enabled=True,
        hbs_reduction=8,
    )

    assert default_config.hbs_enabled is False
    namespace = _namespace_from_configs(hbs_config, TrainConfig(dataset_dir="."))
    assert namespace.hbs_enabled is True
    assert namespace.hbs_reduction == 8
    with pytest.raises(ValueError, match="HBS currently supports detection models only"):
        RFDETRSegMediumConfig(hbs_enabled=True)


def test_hbs_config_validates_loss_range() -> None:
    """Invalid auxiliary loss weights must fail early."""
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
    )
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]])}]

    model.train()
    train_outputs = model(torch.ones(batch_size, 3, 8, 8), targets)
    model.eval()
    eval_outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert "hbs_outputs" in train_outputs
    assert "hbs_outputs" not in eval_outputs
    assert transformer.call_count == 3
