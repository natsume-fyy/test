# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF-DETR's spectral perturbation augmentation."""

import random

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from rfdetr.config import TrainConfig
from rfdetr.datasets.coco import make_coco_transforms
from rfdetr.datasets.transforms import SpectralPerturbation


def _gradient_image(width: int = 32, height: int = 24) -> Image.Image:
    """Create a non-trivial RGB image for frequency-domain tests."""
    values = np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3) % 256
    return Image.fromarray(values.astype(np.uint8), mode="RGB")


def test_spectral_perturbation_probability_zero_is_identity() -> None:
    """A disabled SP transform must preserve both pixels and target data."""
    image = _gradient_image()
    target = {"boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]])}

    output, output_target = SpectralPerturbation(probability=0.0)(image, target)

    np.testing.assert_array_equal(np.asarray(output), np.asarray(image))
    assert output_target is target


def test_spectral_perturbation_changes_pixels_without_changing_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """SP must alter image frequencies while preserving size and annotations."""
    image = _gradient_image(width=31, height=25)
    target = {"boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]])}
    monkeypatch.setattr(random, "randint", lambda _low, _high: 0)

    output, output_target = SpectralPerturbation(probability=1.0)(image, target)

    assert output.size == image.size
    assert output_target is target
    assert not np.array_equal(np.asarray(output), np.asarray(image))


def test_train_config_rejects_reversed_sp_frequency_scales() -> None:
    """The low-frequency cutoff must precede the high-frequency cutoff."""
    with pytest.raises(ValidationError, match="sp_v1_scale must be smaller than sp_v2_scale"):
        TrainConfig(dataset_dir=".", sp_v1_scale=0.8, sp_v2_scale=0.7)


def test_coco_train_pipeline_includes_sp_when_enabled() -> None:
    """COCO training transforms must insert SP before tensor conversion."""
    pipeline = make_coco_transforms(
        "train",
        resolution=64,
        aug_config={},
        sp_prob=0.5,
        sp_v1_scale=0.005,
        sp_v2_scale=0.7,
    )

    sp_indexes = [
        index for index, transform in enumerate(pipeline.transforms) if isinstance(transform, SpectralPerturbation)
    ]
    assert len(sp_indexes) == 1
    assert sp_indexes[0] < len(pipeline.transforms) - 2
