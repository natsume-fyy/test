# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Official SET-style hierarchical background smoothing (HBS).

HBS masks ground-truth foreground before denoising the background and then
copies the original foreground features back exactly. It is used only by
RF-DETR's training-time auxiliary branch and adds no inference latency.
"""

from __future__ import annotations

import torch
from torch import nn


class BackgroundSmoothingBlock(nn.Module):
    """Scale-adaptive residual denoiser from SET's HBS implementation.

    Args:
        channels: Number of input and output feature channels.
        reduction: Bottleneck channel reduction factor.
        kernel_size: Positive odd spatial kernel size.
    """

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 3) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0 or channels // reduction <= 0:
            raise ValueError(
                f"reduction must produce at least one bottleneck channel, got channels={channels}, "
                f"reduction={reduction}."
            )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

        padding = (kernel_size - 1) // 2
        bottleneck_channels = channels // reduction
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size, stride=1, padding=padding, bias=True),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, channels, kernel_size, stride=1, padding=padding, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the residual-smoothed feature map.

        Args:
            features: Feature tensor shaped ``(B, C, H, W)``.

        Returns:
            Smoothed tensor with the same shape as ``features``.
        """
        return features + self.conv_block(features)


class HBS(nn.Module):
    """Smooth background while preserving ground-truth foreground exactly.

    For each level, HBS computes
    ``D(F * M_bg) * M_bg + F * M_fg``. Masking before the denoiser prevents
    foreground activations from bleeding into nearby background through the
    convolution. Padded locations bypass the module unchanged.

    Args:
        channels: Channel count shared by projected feature levels.
        kernel_sizes: One SET denoising kernel size per feature level.
        reduction: Denoiser bottleneck reduction factor.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: list[int],
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one feature level.")
        self.denoisers = nn.ModuleList(
            [
                BackgroundSmoothingBlock(channels=channels, reduction=reduction, kernel_size=kernel_size)
                for kernel_size in kernel_sizes
            ]
        )

    @staticmethod
    def _foreground_mask(
        boxes: torch.Tensor,
        height: int,
        width: int,
        *,
        valid_height: int,
        valid_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Rasterize normalized RF-DETR ``cxcywh`` boxes at one feature level.

        Args:
            boxes: Normalized boxes shaped ``(N, 4)``.
            height: Padded feature-map height.
            width: Padded feature-map width.
            valid_height: Height of the unpadded feature area.
            valid_width: Width of the unpadded feature area.
            dtype: Output mask dtype.
            device: Output mask device.

        Returns:
            Foreground mask shaped ``(1, H, W)``.
        """
        mask = torch.zeros((1, height, width), dtype=dtype, device=device)
        if boxes.numel() == 0:
            return mask

        normalized_boxes = boxes.detach().to(device=device, dtype=torch.float32)
        xyxy = torch.empty_like(normalized_boxes)
        xyxy[:, 0] = normalized_boxes[:, 0] - normalized_boxes[:, 2] / 2
        xyxy[:, 1] = normalized_boxes[:, 1] - normalized_boxes[:, 3] / 2
        xyxy[:, 2] = normalized_boxes[:, 0] + normalized_boxes[:, 2] / 2
        xyxy[:, 3] = normalized_boxes[:, 1] + normalized_boxes[:, 3] / 2
        xyxy.clamp_(0, 1)

        for box in xyxy:
            x1 = max(0, min(valid_width, int(torch.floor(box[0] * valid_width).item())))
            y1 = max(0, min(valid_height, int(torch.floor(box[1] * valid_height).item())))
            x2 = max(0, min(valid_width, int(torch.ceil(box[2] * valid_width).item())))
            y2 = max(0, min(valid_height, int(torch.ceil(box[3] * valid_height).item())))
            if x2 > x1 and y2 > y1:
                mask[:, y1:y2, x1:x2] = 1
        return mask

    def forward(
        self,
        features: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padding_masks: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Transform projected RF-DETR features for the auxiliary training pass.

        Args:
            features: Projected feature maps shaped ``(B, C, H, W)``.
            targets: Per-image targets containing normalized ``boxes``.
            padding_masks: Optional masks shaped ``(B, H, W)`` where ``True`` denotes padding.

        Returns:
            HBS-transformed tensors in the same order and shapes as ``features``.
        """
        if len(features) != len(self.denoisers):
            raise ValueError(f"Expected {len(self.denoisers)} feature levels, received {len(features)}.")
        if not features:
            return []
        if len(targets) != features[0].shape[0]:
            raise ValueError(
                f"Target batch size {len(targets)} does not match feature batch size {features[0].shape[0]}."
            )
        if padding_masks is None:
            padding_masks = [None] * len(features)
        if len(padding_masks) != len(features):
            raise ValueError(f"Expected {len(features)} padding masks, received {len(padding_masks)}.")

        outputs: list[torch.Tensor] = []
        for feature, denoiser, padding_mask in zip(
            features,
            self.denoisers,
            padding_masks,
        ):
            batch_size, _, height, width = feature.shape
            if padding_mask is None:
                valid_mask = feature.new_ones((batch_size, 1, height, width))
                valid_sizes = [(height, width)] * batch_size
            else:
                valid_mask = (~padding_mask).unsqueeze(1).to(dtype=feature.dtype)
                valid_sizes = [
                    (
                        int((~padding_mask[index]).any(dim=1).sum().item()),
                        int((~padding_mask[index]).any(dim=0).sum().item()),
                    )
                    for index in range(batch_size)
                ]

            foreground_mask = torch.stack(
                [
                    self._foreground_mask(
                        targets[index]["boxes"],
                        height,
                        width,
                        valid_height=valid_sizes[index][0],
                        valid_width=valid_sizes[index][1],
                        dtype=feature.dtype,
                        device=feature.device,
                    )
                    for index in range(batch_size)
                ],
                dim=0,
            )
            foreground_mask = foreground_mask * valid_mask
            background_mask = valid_mask - foreground_mask
            smoothed_background = denoiser(feature * background_mask) * background_mask
            preserved_foreground = feature * foreground_mask
            outputs.append(smoothed_background + preserved_foreground + feature * (1 - valid_mask))
        return outputs
