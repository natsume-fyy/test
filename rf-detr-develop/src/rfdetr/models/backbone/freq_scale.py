# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Frequency-dynamic scaling adapted from the FDAM reference implementation.

Reference: https://github.com/Linwei-Chen/FDAM/blob/main/FDAM_mmseg/mmcv_custom/deit_fdam.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class StarReLU(nn.Module):
    """Squared ReLU with learnable scale and bias."""

    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        """Initialize the activation.

        Args:
            scale: Initial output scale.
            bias: Initial output bias.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the scaled squared-ReLU activation."""
        return self.scale * F.relu(inputs).square() + self.bias


class RoutingMLP(nn.Module):
    """Small routing network used to mix learned frequency filters."""

    def __init__(self, dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize the routing MLP."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.activation = StarReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict filter-routing coefficients from pooled features."""
        return self.fc2(self.activation(self.fc1(inputs)))


class GroupDynamicScale(nn.Module):
    """Dynamically scale grouped feature frequencies with learned filter bases.

    This is the ``FreqScale`` component of FDAM. It returns only the frequency
    modulation branch; callers should add it to the original features.
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 1.0,
        reweight_expansion_ratio: float = 0.125,
        num_filters: int = 4,
        size: int = 14,
        group: int = 32,
        init_scale: float = 1e-5,
    ) -> None:
        """Initialize learned frequency filters and their routing network.

        Args:
            dim: Number of input feature channels.
            expansion_ratio: Retained for compatibility with the FDAM API.
            reweight_expansion_ratio: Hidden-width ratio of the routing MLP.
            num_filters: Number of learned frequency-filter bases per group.
            size: Initial spatial size of each learned filter.
            group: Number of channel groups.
            init_scale: Standard deviation used to initialize filter weights.

        Raises:
            ValueError: If dimensions or grouping values are invalid.
        """
        super().__init__()
        if dim < 1 or group < 1 or num_filters < 1 or size < 1:
            raise ValueError("dim, group, num_filters, and size must be positive")
        if dim % group != 0:
            raise ValueError(f"dim ({dim}) must be divisible by group ({group})")

        self.dim = dim
        self.group = group
        self.num_filters = num_filters
        self.med_channels = int(expansion_ratio * dim)
        hidden_dim = max(1, int(reweight_expansion_ratio * dim))
        self.reweight = RoutingMLP(dim, hidden_dim, group * num_filters)
        self.complex_weights = nn.Parameter(
            torch.empty(num_filters, dim // group, size, size // 2 + 1, dtype=torch.float32)
        )
        nn.init.trunc_normal_(self.complex_weights, std=init_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply dynamically routed frequency scaling to an NCHW tensor.

        Args:
            inputs: Feature tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Frequency-modulated features with the same shape and dtype.

        Raises:
            ValueError: If the input does not match the configured channel count.
        """
        if inputs.ndim != 4 or inputs.shape[1] != self.dim:
            raise ValueError(f"Expected NCHW input with {self.dim} channels, got {tuple(inputs.shape)}")

        batch, channels, height, width = inputs.shape
        spectrum = torch.fft.rfft2(inputs.float(), dim=(2, 3), norm="ortho")
        spectrum_height, spectrum_width = spectrum.shape[-2:]

        pooled = inputs.mean(dim=(2, 3))
        routing = self.reweight(pooled).view(batch, self.group, self.num_filters).tanh()

        weights = self.complex_weights
        if weights.shape[-2:] != (spectrum_height, spectrum_width):
            weights = F.interpolate(
                weights,
                size=(spectrum_height, spectrum_width),
                mode="bicubic",
                align_corners=True,
            )
        weights = torch.einsum("bgf,fchw->bgchw", routing.float(), weights)
        weights = weights.reshape(batch, channels, spectrum_height, spectrum_width)
        scaled_spectrum = torch.complex(spectrum.real * weights, spectrum.imag * weights)
        output = torch.fft.irfft2(scaled_spectrum, s=(height, width), dim=(2, 3), norm="ortho")
        return output.to(inputs.dtype)


FrequencyDynamicScale = GroupDynamicScale
FreqScale = GroupDynamicScale

__all__ = ["FreqScale", "FrequencyDynamicScale", "GroupDynamicScale", "StarReLU"]
