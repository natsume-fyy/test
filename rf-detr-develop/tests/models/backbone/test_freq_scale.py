# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the FDAM frequency-dynamic scaling module."""

import pytest
import torch

from rfdetr.models.backbone.freq_scale import GroupDynamicScale


def test_group_dynamic_scale_preserves_shape_and_dtype() -> None:
    """FreqScale should preserve the NCHW feature contract."""
    module = GroupDynamicScale(dim=64, group=8, num_filters=4, size=8)
    features = torch.randn(2, 64, 11, 13)

    output = module(features)

    assert output.shape == features.shape
    assert output.dtype == features.dtype


def test_group_dynamic_scale_zero_weights_produce_zero_modulation() -> None:
    """The module output is a modulation branch; its residual is added by the backbone."""
    module = GroupDynamicScale(dim=32, group=8, num_filters=2, size=8)
    with torch.no_grad():
        module.complex_weights.zero_()

    output = module(torch.randn(1, 32, 8, 8))

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_group_dynamic_scale_is_differentiable() -> None:
    """Both feature inputs and learned frequency filters should receive gradients."""
    module = GroupDynamicScale(dim=32, group=8, num_filters=2, size=8)
    features = torch.randn(2, 32, 8, 8, requires_grad=True)

    module(features).square().mean().backward()

    assert features.grad is not None
    assert module.complex_weights.grad is not None


def test_group_dynamic_scale_rejects_invalid_grouping() -> None:
    """Channel groups must partition the feature channels exactly."""
    with pytest.raises(ValueError, match="divisible"):
        GroupDynamicScale(dim=48, group=32)
