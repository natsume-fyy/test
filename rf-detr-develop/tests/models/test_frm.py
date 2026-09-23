# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for frequency refinement in projected feature maps."""

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from rfdetr.config import RFDETRSmallConfig
from rfdetr.models.backbone.frm import FrequencyRefinementModule
from rfdetr.models.backbone.projector import MultiScaleProjector


def test_frm_preserves_shape_and_backpropagates() -> None:
    """FRM should preserve a feature map's shape and support gradients."""
    module = FrequencyRefinementModule(channels=8, layer_norm=True)
    feature = torch.randn(2, 8, 7, 9, requires_grad=True)

    refined = module(feature)
    refined.mean().backward()

    assert refined.shape == feature.shape
    assert torch.isfinite(refined).all()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_multiscale_projector_refines_selected_output_level() -> None:
    """The projector should apply FRM after projecting a selected level."""
    projector = MultiScaleProjector(
        in_channels=[8],
        out_channels=8,
        scale_factors=[1.0],
        num_blocks=1,
        layer_norm=True,
        frm_enabled=[True],
    )

    outputs = projector([torch.randn(2, 8, 8, 8)])

    assert len(outputs) == 1
    assert outputs[0].shape == (2, 8, 8, 8)
    assert isinstance(projector.frequency_refinement[0], FrequencyRefinementModule)


def test_multiscale_projector_leaves_unselected_level_unchanged() -> None:
    """An unselected projector level should use an identity refinement."""
    projector = MultiScaleProjector(
        in_channels=[8],
        out_channels=8,
        scale_factors=[1.0],
        num_blocks=1,
        layer_norm=True,
        frm_enabled=[False],
    )

    assert isinstance(projector.frequency_refinement[0], nn.Identity)


def test_model_config_rejects_frm_level_without_projector_output() -> None:
    """FRM levels must be a subset of the configured projector outputs."""
    with pytest.raises(ValidationError, match="not present in projector_scale"):
        RFDETRSmallConfig(frm_levels=["P3"])
