# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for frequency-band modulation."""

import pytest
import torch

from rfdetr._namespace import build_namespace
from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.backbone.fbm import FrequencyBandModulation, MultiScaleFrequencyBandModulation


@pytest.mark.parametrize("shape", [(2, 16, 15, 17), (1, 8, 16, 16)])
def test_fbm_preserves_shape_dtype_and_gradient(shape: tuple[int, ...]) -> None:
    """FBM should support odd sizes and remain differentiable."""
    feature = torch.randn(shape, requires_grad=True)
    output = FrequencyBandModulation(shape[1], spatial_group=2)(feature)

    assert output.shape == feature.shape
    assert output.dtype == feature.dtype
    output.mean().backward()
    assert feature.grad is not None


def test_fbm_is_near_identity_at_initialization() -> None:
    """Zero-centred attention initialization should preserve pretrained features."""
    feature = torch.randn(2, 16, 16, 16)
    output = FrequencyBandModulation(16)(feature)

    torch.testing.assert_close(output, feature, rtol=1e-4, atol=1e-4)


def test_multiscale_fbm_preserves_pyramid() -> None:
    """Every projected pyramid level should be modulated independently."""
    features = [torch.randn(1, 8, 16, 16), torch.randn(1, 8, 8, 8)]
    outputs = MultiScaleFrequencyBandModulation(8, 2)(features)

    assert [output.shape for output in outputs] == [feature.shape for feature in features]


def test_fbm_rejects_invalid_spatial_group() -> None:
    """Attention groups must evenly partition the channel dimension."""
    with pytest.raises(ValueError, match="divide in_channels"):
        FrequencyBandModulation(10, spatial_group=3)


def test_fbm_config_reaches_builder_namespace() -> None:
    """Model FBM settings should be forwarded to the legacy model builder."""
    model_config = RFDETRSmallConfig(
        use_fbm=True,
        fbm_k_list=[2, 4],
        fbm_lowfreq_att=True,
        fbm_spatial_group=4,
    )
    namespace = build_namespace(model_config, TrainConfig(dataset_dir="/tmp"))

    assert namespace.use_fbm is True
    assert namespace.fbm_k_list == [2, 4]
    assert namespace.fbm_lowfreq_att is True
    assert namespace.fbm_spatial_group == 4
