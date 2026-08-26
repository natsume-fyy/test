# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for training-only HBS background suppression and foreground restoration."""

from unittest.mock import MagicMock

import pytest
import torch

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRMediumConfig, RFDETRSegMediumConfig, TrainConfig
from rfdetr.models.hbs import BackgroundSmoothingBlock, ForegroundDetailEnhancement, HBS
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


class TestBackgroundSmoothingBlock:
    """Unit tests for the residual denoising block used by HBS."""

    def test_preserves_shape_and_backpropagates(self) -> None:
        """The residual block must preserve feature shape and receive gradients."""
        block = BackgroundSmoothingBlock(channels=8, reduction=4, kernel_size=3)
        features = torch.randn(2, 8, 7, 9, requires_grad=True)

        output = block(features)
        output.sum().backward()

        assert output.shape == features.shape
        assert features.grad is not None
        assert all(parameter.grad is not None for parameter in block.parameters())


class TestForegroundDetailEnhancement:
    """Unit tests for foreground high-frequency restoration."""

    def test_enhances_foreground_detail_without_changing_background(self) -> None:
        """The high-pass residual must be added only at foreground locations."""
        enhancer = ForegroundDetailEnhancement(kernel_size=3, scale=0.5)
        features = torch.full((1, 1, 5, 5), 2.0)
        features[:, :, 1:4, 1:4] = 0
        features[:, :, 2, 2] = 1
        foreground_mask = torch.zeros(1, 1, 5, 5)
        foreground_mask[:, :, 1:4, 1:4] = 1

        enhanced = enhancer(features, foreground_mask)

        torch.testing.assert_close(enhanced[0, 0, 2, 2], torch.tensor(13 / 9))
        assert torch.count_nonzero(enhanced * (1 - foreground_mask)) == 0

    def test_learns_an_independent_enhancement_scale(self) -> None:
        """The per-level detail scale must receive gradients during HBS training."""
        enhancer = ForegroundDetailEnhancement(kernel_size=3, scale=0.1)
        features = torch.randn(1, 2, 5, 5, requires_grad=True)
        foreground_mask = torch.ones(1, 1, 5, 5)

        enhancer(features, foreground_mask).sum().backward()

        assert enhancer.scale.grad is not None


class TestHBS:
    """Behavioural tests for foreground masking and feature-level smoothing."""

    def test_preserves_foreground_and_only_changes_background(self) -> None:
        """A zero restoration scale must preserve the original background-only HBS behaviour."""
        hbs = HBS(channels=4, kernel_sizes=[3], reduction=2, foreground_scale=0.0)
        with torch.no_grad():
            for parameter in hbs.denoisers.parameters():
                parameter.fill_(0.1)

        features = [torch.ones(1, 4, 8, 8)]
        targets = [
            {
                # Normalized cx, cy, width, height: central 4x4 feature region.
                "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            }
        ]

        smoothed = hbs(features, targets)[0]

        assert torch.equal(smoothed[:, :, 2:6, 2:6], features[0][:, :, 2:6, 2:6])
        assert not torch.equal(smoothed[:, :, :2, :2], features[0][:, :, :2, :2])

    def test_preserves_padded_feature_locations(self) -> None:
        """Padding must not be smoothed or interpreted as real image background."""
        hbs = HBS(channels=4, kernel_sizes=[3], reduction=2)
        with torch.no_grad():
            for parameter in hbs.parameters():
                parameter.fill_(0.1)
        features = [torch.ones(1, 4, 8, 8)]
        padding_mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        padding_mask[:, 6:, :] = True
        padding_mask[:, :, 6:] = True

        smoothed = hbs(
            features,
            [{"boxes": torch.empty(0, 4)}],
            [padding_mask],
        )[0]

        assert torch.equal(smoothed[:, :, 6:, :], features[0][:, :, 6:, :])
        assert torch.equal(smoothed[:, :, :, 6:], features[0][:, :, :, 6:])

    def test_restores_detail_inside_foreground_boxes(self) -> None:
        """HBS must enhance non-uniform foreground features before recombination."""
        hbs = HBS(channels=1, kernel_sizes=[3], reduction=1, foreground_scale=0.5)
        features = [torch.zeros(1, 1, 5, 5)]
        features[0][:, :, 2, 2] = 1
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.6, 0.6]])}]

        output = hbs(features, targets)[0]

        torch.testing.assert_close(output[0, 0, 2, 2], torch.tensor(13 / 9))


class TestHBSConfig:
    """Configuration tests for the opt-in HBS architecture."""

    def test_detection_config_accepts_hbs(self) -> None:
        """Detection variants should expose HBS without changing the default architecture."""
        default_config = RFDETRMediumConfig()
        hbs_config = RFDETRMediumConfig(
            hbs_enabled=True,
            hbs_reduction=8,
            hbs_foreground_scale=0.2,
            hbs_foreground_kernel_size=5,
        )

        assert default_config.hbs_enabled is False
        assert hbs_config.hbs_enabled is True
        assert hbs_config.hbs_reduction == 8
        assert hbs_config.hbs_foreground_scale == 0.2
        assert hbs_config.hbs_foreground_kernel_size == 5

        namespace = _namespace_from_configs(hbs_config, TrainConfig(dataset_dir="."))
        assert namespace.hbs_foreground_scale == 0.2
        assert namespace.hbs_foreground_kernel_size == 5

    def test_detection_config_rejects_even_foreground_kernel(self) -> None:
        """The centered foreground low-pass filter requires an odd kernel size."""
        with pytest.raises(ValueError, match="hbs_foreground_kernel_size must be odd"):
            RFDETRMediumConfig(hbs_foreground_kernel_size=4)

    def test_segmentation_config_rejects_unvalidated_hbs_branch(self) -> None:
        """HBS must fail explicitly for task heads whose auxiliary objective is not implemented."""
        with pytest.raises(ValueError, match="HBS currently supports detection models only"):
            RFDETRSegMediumConfig(hbs_enabled=True)

    def test_training_config_accepts_hbs_loss_weight(self) -> None:
        """The training config should expose the auxiliary loss coefficient."""
        assert TrainConfig(dataset_dir=".", hbs_loss_coef=0.25).hbs_loss_coef == 0.25

    def test_training_config_rejects_negative_hbs_loss_weight(self) -> None:
        """The auxiliary loss coefficient must be non-negative."""
        with pytest.raises(ValueError):
            TrainConfig(dataset_dir=".", hbs_loss_coef=-0.1)


def test_lwdetr_emits_hbs_outputs_only_during_training() -> None:
    """LWDETR should add an HBS auxiliary prediction branch only for training batches with targets."""
    batch_size = 1
    num_queries = 2
    hidden_dim = 4
    num_classes = 3
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
        num_classes=num_classes,
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
        hbs_foreground_kernel_size=5,
    )
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]])}]

    model.train()
    train_outputs = model(torch.ones(batch_size, 3, 8, 8), targets)
    model.eval()
    eval_outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert "hbs_outputs" in train_outputs
    assert "hbs_outputs" not in eval_outputs
    assert model.hbs is not None
    assert model.hbs.foreground_enhancers[0].kernel_size == 5
    torch.testing.assert_close(model.hbs.foreground_enhancers[0].scale, torch.tensor(0.2))
    assert transformer.call_count == 3
