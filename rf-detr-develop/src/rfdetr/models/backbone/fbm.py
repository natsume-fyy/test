# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from FDConv (https://github.com/Linwei-Chen/FDConv), MIT License.
# Copyright (c) 2025 Linwei Chen.
# ------------------------------------------------------------------------
"""Frequency-band modulation for projected detector features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FrequencyBandModulation(nn.Module):
    """Adaptively reweight disjoint spatial-frequency bands.

    This is the FBM component from FDConv, adapted to avoid MMCV dependencies and
    to preserve the input dtype under mixed-precision training.

    Args:
        in_channels: Number of channels in the input feature map.
        k_list: Frequency divisors defining nested low-pass cutoffs.
        lowfreq_att: Whether to learn an attention map for the final low band.
        spatial_group: Number of channel groups sharing each attention map.
        spatial_kernel: Kernel size used to predict spatial attention.
    """

    def __init__(
        self,
        in_channels: int,
        k_list: Sequence[int] = (2, 4, 8),
        lowfreq_att: bool = False,
        spatial_group: int = 1,
        spatial_kernel: int = 3,
    ) -> None:
        super().__init__()
        if not k_list or any(k <= 0 for k in k_list):
            raise ValueError("k_list must contain at least one positive integer")
        if spatial_group <= 0 or in_channels % spatial_group != 0:
            raise ValueError("spatial_group must be positive and divide in_channels")
        if spatial_kernel <= 0 or spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be a positive odd integer")

        self.k_list = tuple(k_list)
        self.lowfreq_att = lowfreq_att
        self.spatial_group = spatial_group
        attention_count = len(self.k_list) + int(lowfreq_att)
        self.freq_weight_convs = nn.ModuleList(
            nn.Conv2d(
                in_channels,
                spatial_group,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                groups=spatial_group,
            )
            for _ in range(attention_count)
        )
        for conv in self.freq_weight_convs:
            nn.init.normal_(conv.weight, std=1e-6)
            nn.init.zeros_(conv.bias)

    @staticmethod
    def _frequency_radius(height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return the Chebyshev frequency radius for an RFFT2 grid."""
        freq_h = torch.fft.fftfreq(height, device=device)
        freq_w = torch.fft.rfftfreq(width, device=device)
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
        return torch.maximum(grid_h.abs(), grid_w.abs())

    @staticmethod
    def _attention(logits: torch.Tensor) -> torch.Tensor:
        """Map zero-centred logits to identity-centred modulation weights."""
        return logits.sigmoid() * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate frequency bands while preserving input shape and dtype."""
        input_dtype = x.dtype
        batch_size, _, height, width = x.shape
        x_float = x.float()
        spectrum = torch.fft.rfft2(x_float, norm="ortho")
        radius = self._frequency_radius(height, width, x.device)

        remaining = x_float
        bands: list[torch.Tensor] = []
        for index, divisor in enumerate(self.k_list):
            low_mask = radius < (0.5 / divisor + 1e-8)
            low_part = torch.fft.irfft2(spectrum * low_mask, s=(height, width), norm="ortho")
            high_part = remaining - low_part
            remaining = low_part

            weight = self._attention(self.freq_weight_convs[index](x))
            weighted = weight.reshape(batch_size, self.spatial_group, -1, height, width) * high_part.reshape(
                batch_size, self.spatial_group, -1, height, width
            )
            bands.append(weighted.flatten(1, 2))

        if self.lowfreq_att:
            weight = self._attention(self.freq_weight_convs[-1](x))
            remaining = (
                weight.reshape(batch_size, self.spatial_group, -1, height, width)
                * remaining.reshape(batch_size, self.spatial_group, -1, height, width)
            ).flatten(1, 2)
        bands.append(remaining)
        return sum(bands[1:], start=bands[0]).to(input_dtype)


class MultiScaleFrequencyBandModulation(nn.Module):
    """Apply an independent FBM block to every projected feature level."""

    def __init__(
        self,
        channels: int,
        num_levels: int,
        k_list: Sequence[int] = (2, 4, 8),
        lowfreq_att: bool = False,
        spatial_group: int = 1,
    ) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            FrequencyBandModulation(
                channels,
                k_list=k_list,
                lowfreq_att=lowfreq_att,
                spatial_group=spatial_group,
            )
            for _ in range(num_levels)
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Return FBM-modulated features in the original pyramid order."""
        if len(features) != len(self.levels):
            raise ValueError(f"Expected {len(self.levels)} feature levels, got {len(features)}")
        return [fbm(feature) for fbm, feature in zip(self.levels, features)]
