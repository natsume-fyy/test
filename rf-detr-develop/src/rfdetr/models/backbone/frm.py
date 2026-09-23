# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Frequency-domain refinement for weather-degraded feature maps."""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _dct_1d(inputs: Tensor) -> Tensor:
    """Compute a type-II DCT along the final dimension."""
    input_shape = inputs.shape
    length = input_shape[-1]
    flattened = inputs.contiguous().reshape(-1, length)
    reordered = torch.cat([flattened[:, ::2], flattened[:, 1::2].flip(1)], dim=1)
    spectrum = torch.fft.fft(reordered, dim=1)
    angles = -torch.arange(length, dtype=inputs.dtype, device=inputs.device) * math.pi / (2 * length)
    transformed = spectrum.real * torch.cos(angles) - spectrum.imag * torch.sin(angles)
    return (2 * transformed).reshape(input_shape)


def _idct_1d(inputs: Tensor) -> Tensor:
    """Invert :func:`_dct_1d` along the final dimension."""
    input_shape = inputs.shape
    length = input_shape[-1]
    coefficients = inputs.contiguous().reshape(-1, length) / 2
    angles = torch.arange(length, dtype=inputs.dtype, device=inputs.device) * math.pi / (2 * length)
    imaginary = torch.cat([coefficients[:, :1] * 0, -coefficients.flip(1)[:, :-1]], dim=1)
    real_part = coefficients * torch.cos(angles) - imaginary * torch.sin(angles)
    imag_part = coefficients * torch.sin(angles) + imaginary * torch.cos(angles)
    reordered = torch.fft.irfft(torch.complex(real_part, imag_part), n=length, dim=1)
    restored = reordered.new_zeros(reordered.shape)
    restored[:, ::2] = reordered[:, : length - (length // 2)]
    restored[:, 1::2] = reordered.flip(1)[:, : length // 2]
    return restored.reshape(input_shape)


def dct_2d(inputs: Tensor) -> Tensor:
    """Compute a two-dimensional type-II DCT over a feature map."""
    transformed = _dct_1d(inputs)
    return _dct_1d(transformed.transpose(-1, -2)).transpose(-1, -2)


def idct_2d(inputs: Tensor) -> Tensor:
    """Invert :func:`dct_2d` over a feature map."""
    restored = _idct_1d(inputs)
    return _idct_1d(restored.transpose(-1, -2)).transpose(-1, -2)


class _ChannelLayerNorm(nn.Module):
    """Apply LayerNorm over channels of a BCHW tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize channel values independently at each spatial position."""
        return self.norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _ChannelAttention(nn.Module):
    """Generate channel weights from a low-frequency response."""

    def __init__(self, channels: int, pooled_size: int = 16) -> None:
        super().__init__()
        self.pooled_size = pooled_size
        self.avg_projection = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.max_projection = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.output_projection = nn.Conv2d(2 * channels, channels, kernel_size=1, groups=channels)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one attention weight per channel."""
        pooled_size = min(self.pooled_size, inputs.shape[-2], inputs.shape[-1])
        avg_descriptor = F.adaptive_avg_pool2d(inputs, pooled_size).relu().sum(dim=(2, 3), keepdim=True)
        max_descriptor = F.adaptive_max_pool2d(inputs, pooled_size).relu().sum(dim=(2, 3), keepdim=True)
        descriptor = torch.cat(
            [self.avg_projection(avg_descriptor), self.max_projection(max_descriptor)],
            dim=1,
        )
        return self.output_projection(descriptor).sigmoid()


class _SpatialAttention(nn.Module):
    """Generate spatial weights from a high-frequency response."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one attention weight per spatial position."""
        descriptors = torch.cat(
            [inputs.amax(dim=1, keepdim=True), inputs.mean(dim=1, keepdim=True)],
            dim=1,
        )
        return self.norm(self.projection(descriptors)).sigmoid()


class FrequencyRefinementModule(nn.Module):
    """Refine a feature map using adaptive high- and low-frequency cues.

    The module predicts sample-dependent DCT cutoffs. High-frequency responses
    produce spatial attention, while low-frequency responses produce channel
    attention. The attention maps reweight the original feature so FRM shapes
    detection features without reconstructing an image.
    """

    def __init__(self, channels: int, layer_norm: bool = True, mask_sharpness: float = 20.0) -> None:
        super().__init__()
        hidden_channels = max(1, channels // 8)
        self.mask_sharpness = mask_sharpness
        self.threshold_predictor = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.Mish(),
            nn.Linear(hidden_channels, 4),
            nn.Sigmoid(),
        )
        self.spatial_attention = _SpatialAttention()
        self.channel_attention = _ChannelAttention(channels)
        output_norm: nn.Module = _ChannelLayerNorm(channels) if layer_norm else nn.BatchNorm2d(channels)
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            output_norm,
            nn.Mish(),
        )

    def _high_pass_mask(self, height: int, width: int, cutoffs: Tensor) -> Tensor:
        """Build a hard high-pass mask with a differentiable STE gradient."""
        rows, columns = self._frequency_coordinates(height, width, cutoffs)
        row_cutoff = cutoffs[:, 0, None, None] * height
        column_cutoff = cutoffs[:, 1, None, None] * width
        soft_row = torch.sigmoid(self.mask_sharpness * (rows - row_cutoff))
        soft_column = torch.sigmoid(self.mask_sharpness * (columns - column_cutoff))
        soft_mask = 1 - (1 - soft_row) * (1 - soft_column)
        hard_mask = ((rows >= row_cutoff) | (columns >= column_cutoff)).to(cutoffs.dtype)
        return hard_mask + soft_mask - soft_mask.detach()

    def _low_pass_mask(self, height: int, width: int, cutoffs: Tensor) -> Tensor:
        """Build a hard low-pass mask with a differentiable STE gradient."""
        rows, columns = self._frequency_coordinates(height, width, cutoffs)
        row_cutoff = cutoffs[:, 0, None, None] * height
        column_cutoff = cutoffs[:, 1, None, None] * width
        soft_row = torch.sigmoid(self.mask_sharpness * (rows - row_cutoff))
        soft_column = torch.sigmoid(self.mask_sharpness * (columns - column_cutoff))
        soft_mask = 1 - soft_row * soft_column
        hard_mask = ((rows < row_cutoff) | (columns < column_cutoff)).to(cutoffs.dtype)
        return hard_mask + soft_mask - soft_mask.detach()

    @staticmethod
    def _frequency_coordinates(height: int, width: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Return broadcastable DCT row and column coordinates."""
        rows = torch.arange(height, dtype=reference.dtype, device=reference.device)[None, :, None]
        columns = torch.arange(width, dtype=reference.dtype, device=reference.device)[None, None, :]
        return rows, columns

    def forward(self, inputs: Tensor) -> Tensor:
        """Refine one BCHW feature map."""
        _, _, height, width = inputs.shape
        cutoffs = self.threshold_predictor(F.adaptive_avg_pool2d(inputs, 1).flatten(1))

        # FFT kernels are more reliable in float32 under mixed precision. Cast
        # the reconstructed responses back before applying attention.
        frequency = dct_2d(inputs.float())
        mask_cutoffs = cutoffs.float()
        high_mask = self._high_pass_mask(height, width, mask_cutoffs).unsqueeze(1)
        low_mask = self._low_pass_mask(height, width, mask_cutoffs[:, 2:]).unsqueeze(1)
        high_response = idct_2d(frequency * high_mask).to(inputs.dtype)
        low_response = idct_2d(frequency * low_mask).to(inputs.dtype)

        spatial_weights = self.spatial_attention(high_response)
        channel_weights = self.channel_attention(low_response)
        return self.output_projection(inputs * (spatial_weights + channel_weights))

