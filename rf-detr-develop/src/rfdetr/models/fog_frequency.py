# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fog-aware dynamic frequency regulation for RF-DETR training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class FogAwareFrequencyRegulator(nn.Module):
    """Protect mid-frequency structure while perturbing fog-sensitive bands.

    The controller replaces DGCR's teacher/student discrepancy signal with
    three signals available in ordinary supervised RF-DETR training:

    * a per-image fog score from dark channel, contrast, and saturation;
    * radial spectrum statistics from the current image;
    * the current normalized training progress supplied by the trainer.

    ``tau1`` grows and ``tau2`` shrinks as fog becomes denser, narrowing the
    protected middle band.  High-frequency perturbation is deliberately close
    to zero for clear/light-fog inputs so UAV small-object edges are retained.

    Args:
        probability: Probability of applying regulation to each training image.
        tau1_range: Minimum and maximum low/middle boundary.
        tau2_range: Minimum and maximum middle/high boundary.
        max_low_strength: Maximum low-frequency perturbation strength.
        max_high_strength: Maximum high-frequency perturbation strength.
        warmup_fraction: Fraction of training used to ramp perturbations in.
        transition_width: Width of the soft frequency-band boundaries.
        normalized_input: Whether inputs use ImageNet normalization.
        image_mean: Channel means used to undo input normalization.
        image_std: Channel standard deviations used to undo normalization.
    """

    def __init__(
        self,
        probability: float = 0.8,
        tau1_range: tuple[float, float] = (0.03, 0.18),
        tau2_range: tuple[float, float] = (0.60, 0.92),
        max_low_strength: float = 0.35,
        max_high_strength: float = 0.28,
        warmup_fraction: float = 0.1,
        transition_width: float = 0.02,
        normalized_input: bool = False,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.probability = probability
        self.tau1_range = tau1_range
        self.tau2_range = tau2_range
        self.max_low_strength = max_low_strength
        self.max_high_strength = max_high_strength
        self.warmup_fraction = warmup_fraction
        self.transition_width = transition_width
        self.normalized_input = normalized_input
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("training_progress", torch.tensor(0.0), persistent=True)
        self.last_statistics: dict[str, torch.Tensor] = {}

    def set_training_progress(self, progress: float) -> None:
        """Set normalized RF-DETR training progress in ``[0, 1]``.

        Args:
            progress: Completed fraction of the current training run.
        """
        self.training_progress.fill_(min(max(float(progress), 0.0), 1.0))

    @staticmethod
    def _radial_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return an FFT-shifted radial coordinate grid normalized to ``[0, 1]``."""
        yy = torch.linspace(-1.0, 1.0, height, device=device)
        xx = torch.linspace(-1.0, 1.0, width, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.sqrt(grid_x.square() + grid_y.square()).clamp_max(1.0)

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Compute a per-image mean over channels and valid spatial positions."""
        expanded_valid = valid.expand(-1, values.shape[1], -1, -1)
        count = expanded_valid.sum(dim=(1, 2, 3)).clamp_min(1)
        return (values * expanded_valid).sum(dim=(1, 2, 3)) / count

    def _fill_padding(self, images: torch.Tensor, padding_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill padded pixels with the valid channel mean before spectral analysis."""
        if padding_mask is None:
            valid = torch.ones(
                (images.shape[0], 1, images.shape[-2], images.shape[-1]),
                dtype=torch.bool,
                device=images.device,
            )
            return images, valid
        valid = (~padding_mask).unsqueeze(1)
        count = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1)
        channel_mean = (images * valid).sum(dim=(2, 3), keepdim=True) / count
        return torch.where(valid, images, channel_mean), valid

    def estimate_parameters(
        self,
        images: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Estimate fog, frequency boundaries, and perturbation strengths.

        Args:
            images: RGB images in ``[0, 1]`` with shape ``[B, 3, H, W]``.
            padding_mask: Optional mask with ``True`` on padded pixels.

        Returns:
            Per-image tensors for fog score, ``tau1``, ``tau2``, low strength,
            high strength, spectral centroid, and high-frequency energy ratio.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB BCHW input, got shape {tuple(images.shape)}")

        images = images.float().clamp(0.0, 1.0)
        images, valid = self._fill_padding(images, padding_mask)
        dark = images.amin(dim=1, keepdim=True)
        kernel = max(3, min(15, images.shape[-2], images.shape[-1]))
        if kernel % 2 == 0:
            kernel -= 1
        local_dark = -F.max_pool2d(-dark, kernel_size=kernel, stride=1, padding=kernel // 2)
        dark_score = self._masked_mean(local_dark, valid)

        grayscale = 0.299 * images[:, :1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        gray_mean = self._masked_mean(grayscale, valid)
        gray_variance = self._masked_mean((grayscale - gray_mean[:, None, None, None]).square(), valid)
        low_contrast = 1.0 - (gray_variance.sqrt() / 0.25).clamp(0.0, 1.0)
        channel_max = images.amax(dim=1, keepdim=True)
        channel_min = images.amin(dim=1, keepdim=True)
        saturation = (channel_max - channel_min) / channel_max.clamp_min(1e-4)
        low_saturation = 1.0 - self._masked_mean(saturation, valid).clamp(0.0, 1.0)
        fog_score = (0.5 * dark_score + 0.3 * low_contrast + 0.2 * low_saturation).clamp(0.0, 1.0)

        spectrum = torch.fft.fftshift(torch.fft.fft2(images, norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(spectrum.abs()).mean(dim=1)
        radius = self._radial_grid(images.shape[-2], images.shape[-1], images.device)
        total_energy = magnitude.sum(dim=(1, 2)).clamp_min(1e-6)
        spectral_centroid = (magnitude * radius).sum(dim=(1, 2)) / total_energy
        high_frequency_ratio = (magnitude * (radius >= 0.65)).sum(dim=(1, 2)) / total_energy
        low_frequency_bias = (1.0 - spectral_centroid / 0.5).clamp(0.0, 1.0)
        high_frequency_activity = (4.0 * high_frequency_ratio).clamp(0.0, 1.0)

        tau1_min, tau1_max = self.tau1_range
        tau2_min, tau2_max = self.tau2_range
        tau1_control = fog_score * (0.85 + 0.15 * low_frequency_bias)
        tau2_control = fog_score * (0.82 + 0.18 * high_frequency_activity)
        tau1 = tau1_min + (tau1_max - tau1_min) * tau1_control
        tau2 = tau2_max - (tau2_max - tau2_min) * tau2_control
        tau2 = torch.maximum(tau2, tau1 + 2.0 * self.transition_width)

        progress = self.training_progress.to(images.device)
        ramp_position = ((progress - self.warmup_fraction) / max(1.0 - self.warmup_fraction, 1e-6)).clamp(0, 1)
        smooth_ramp = ramp_position.square() * (3.0 - 2.0 * ramp_position)
        curriculum = 0.15 + 0.85 * smooth_ramp
        low_strength = self.max_low_strength * curriculum * fog_score * (0.85 + 0.15 * low_frequency_bias)
        high_strength = self.max_high_strength * curriculum * fog_score.square() * (
            0.85 + 0.15 * high_frequency_activity
        )

        return {
            "fog_score": fog_score,
            "tau1": tau1,
            "tau2": tau2,
            "low_strength": low_strength,
            "high_strength": high_strength,
            "spectral_centroid": spectral_centroid,
            "high_frequency_ratio": high_frequency_ratio,
        }

    def forward(self, images: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply fog-aware frequency regulation during training only.

        Args:
            images: Input tensor in configured normalization space.
            padding_mask: Optional mask with ``True`` on padded pixels.

        Returns:
            Regulated images with the same shape, dtype, and normalization.
        """
        if not self.training or self.probability <= 0.0:
            return images

        original_dtype = images.dtype
        working = images.float()
        if self.normalized_input:
            working = working * self.image_std + self.image_mean
        working = working.clamp(0.0, 1.0)
        spectral_input, _ = self._fill_padding(working, padding_mask)
        with torch.no_grad():
            parameters = {key: value.detach() for key, value in self.estimate_parameters(working, padding_mask).items()}

        radius = self._radial_grid(working.shape[-2], working.shape[-1], working.device)[None, None]
        tau1 = parameters["tau1"][:, None, None, None]
        tau2 = parameters["tau2"][:, None, None, None]
        low_mask = torch.sigmoid((tau1 - radius) / self.transition_width)
        high_mask = torch.sigmoid((radius - tau2) / self.transition_width)
        low_strength = parameters["low_strength"][:, None, None, None]
        high_strength = parameters["high_strength"][:, None, None, None]
        stochastic_scale = torch.randn(
            (working.shape[0], working.shape[1], 1, 1),
            device=working.device,
            dtype=working.dtype,
        ).clamp(-2.0, 2.0)
        band_strength = low_strength * low_mask + high_strength * high_mask
        gain = (1.0 - band_strength + 0.12 * stochastic_scale * band_strength).clamp(0.1, 1.15)
        apply_mask = (torch.rand((working.shape[0], 1, 1, 1), device=working.device) < self.probability).float()
        gain = 1.0 + apply_mask * (gain - 1.0)

        spectrum = torch.fft.fftshift(torch.fft.fft2(spectral_input, norm="ortho"), dim=(-2, -1))
        regulated_spectrum = spectrum * gain
        regulated = torch.fft.ifft2(
            torch.fft.ifftshift(regulated_spectrum, dim=(-2, -1)),
            norm="ortho",
        ).real.clamp(0.0, 1.0)
        if padding_mask is not None:
            regulated = torch.where((~padding_mask).unsqueeze(1), regulated, working)
        if self.normalized_input:
            regulated = (regulated - self.image_mean) / self.image_std

        self.last_statistics = {key: value.mean().detach() for key, value in parameters.items()}
        return regulated.to(original_dtype)
