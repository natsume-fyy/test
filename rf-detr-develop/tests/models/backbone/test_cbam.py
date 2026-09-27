# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for residual CBAM refinement of projected pyramid features."""

import torch

from rfdetr.models.backbone.projector import PyramidCBAM, ResidualCBAM


def test_residual_cbam_preserves_shape_and_adds_attention_branch() -> None:
    """Residual CBAM should add its channel-and-spatial attention branch to the input."""
    module = ResidualCBAM(channels=8, reduction=4)
    for parameter in module.parameters():
        parameter.data.zero_()

    inputs = torch.ones(2, 8, 6, 5)
    outputs = module(inputs)

    assert outputs.shape == inputs.shape
    # Zero logits give 0.5 channel attention followed by 0.5 spatial attention.
    torch.testing.assert_close(outputs, inputs * 1.25)


def test_pyramid_cbam_refines_only_p3_and_p4() -> None:
    """P3/P4 should be refined while other projector levels pass through unchanged."""
    module = PyramidCBAM(levels=["P3", "P4", "P5"], channels=8)
    for parameter in module.parameters():
        parameter.data.zero_()

    features = [torch.ones(1, 8, 8, 8), torch.ones(1, 8, 4, 4), torch.ones(1, 8, 2, 2)]
    outputs = module(features)

    torch.testing.assert_close(outputs[0], features[0] * 1.25)
    torch.testing.assert_close(outputs[1], features[1] * 1.25)
    assert outputs[2] is features[2]


def test_pyramid_cbam_supports_single_projector_level() -> None:
    """A model exposing only P4 should still apply CBAM to that feature."""
    module = PyramidCBAM(levels=["P4"], channels=8)

    outputs = module([torch.randn(2, 8, 5, 5)])

    assert len(outputs) == 1
    assert outputs[0].shape == (2, 8, 5, 5)
