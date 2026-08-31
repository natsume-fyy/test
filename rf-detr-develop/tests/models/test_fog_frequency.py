# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the fog-aware dynamic frequency regulator."""

import torch

from rfdetr.config import RFDETRSmallConfig
from rfdetr.models.fog_frequency import FogAwareFrequencyRegulator


def _clear_detail_image(size: int = 64) -> torch.Tensor:
    """Create a high-contrast RGB image with abundant small-scale detail."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    checker = ((xx + yy) % 2).float()
    return torch.stack((checker, 1.0 - checker, checker), dim=0).unsqueeze(0)


def test_small_config_accepts_fog_frequency_options() -> None:
    """RF-DETR constructors should expose the regulator through model config."""
    config = RFDETRSmallConfig(
        pretrain_weights=None,
        use_fog_frequency_regulator=True,
        fog_frequency_probability=0.8,
    )

    assert config.use_fog_frequency_regulator is True
    assert config.fog_frequency_probability == 0.8


def test_dense_fog_narrows_protected_mid_band_and_increases_strength() -> None:
    """Dense fog should perturb wider low/high bands than a clear detailed input."""
    regulator = FogAwareFrequencyRegulator(probability=1.0)
    regulator.set_training_progress(1.0)
    clear = _clear_detail_image()
    dense_fog = torch.full_like(clear, 0.82)

    clear_params = regulator.estimate_parameters(clear)
    fog_params = regulator.estimate_parameters(dense_fog)

    assert fog_params["fog_score"].item() > clear_params["fog_score"].item()
    assert fog_params["tau1"].item() > clear_params["tau1"].item()
    assert fog_params["tau2"].item() < clear_params["tau2"].item()
    assert fog_params["low_strength"].item() > clear_params["low_strength"].item()
    assert fog_params["high_strength"].item() > clear_params["high_strength"].item()


def test_training_progress_ramps_strength_without_moving_image_driven_boundaries() -> None:
    """The curriculum should ramp perturbation while keeping image-derived bands stable."""
    regulator = FogAwareFrequencyRegulator(probability=1.0)
    dense_fog = torch.full((1, 3, 64, 64), 0.82)
    regulator.set_training_progress(0.0)
    early = regulator.estimate_parameters(dense_fog)
    regulator.set_training_progress(1.0)
    late = regulator.estimate_parameters(dense_fog)

    assert torch.equal(early["tau1"], late["tau1"])
    assert torch.equal(early["tau2"], late["tau2"])
    assert late["low_strength"].item() > early["low_strength"].item()
    assert late["high_strength"].item() > early["high_strength"].item()


def test_regulator_only_changes_valid_training_pixels() -> None:
    """The regulator must preserve shape, dtype, padding, and evaluation inputs."""
    regulator = FogAwareFrequencyRegulator(probability=1.0)
    regulator.set_training_progress(1.0)
    regulator.train()
    image = torch.full((1, 3, 64, 64), 0.82)
    padding_mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    padding_mask[:, :, 56:] = True

    torch.manual_seed(3)
    output = regulator(image, padding_mask)

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert torch.isfinite(output).all()
    assert torch.equal(output[..., 56:], image[..., 56:])
    assert not torch.equal(output[..., :56], image[..., :56])

    regulator.eval()
    assert torch.equal(regulator(image, padding_mask), image)
