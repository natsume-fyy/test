# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Semantic feature and query prediction consistency for EMA teacher-student training."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import nn

from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from rfdetr.utilities.tensors import NestedTensor

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _channel_stats(channels: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ImageNet normalization statistics for an arbitrary channel count.

    Args:
        channels: Number of image channels.
        reference: Tensor providing device and dtype.

    Returns:
        Broadcastable mean and standard-deviation tensors.
    """
    mean = [_IMAGENET_MEAN[index % 3] for index in range(channels)]
    std = [_IMAGENET_STD[index % 3] for index in range(channels)]
    return (
        reference.new_tensor(mean).view(1, channels, 1, 1),
        reference.new_tensor(std).view(1, channels, 1, 1),
    )


def make_strong_view(samples: NestedTensor, strength: float = 0.7) -> NestedTensor:
    """Create a geometry-preserving strong photometric view of a normalized batch.

    Brightness, contrast, saturation, Gaussian noise, and a light blur are applied
    after undoing ImageNet normalization. Spatial operations are deliberately
    excluded so the original targets and Teacher boxes remain aligned.

    Args:
        samples: Weakly augmented, normalized input batch.
        strength: Photometric perturbation strength in ``[0, 1]``.

    Returns:
        Strongly augmented batch with the original padding mask.

    Raises:
        ValueError: If strength is outside ``[0, 1]``.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}.")
    tensors = samples.tensors
    if strength == 0.0:
        return NestedTensor(tensors.clone(), samples.mask)

    mean, std = _channel_stats(tensors.shape[1], tensors)
    images = (tensors * std + mean).clamp(0.0, 1.0)
    batch = images.shape[0]
    rand_shape = (batch, 1, 1, 1)

    brightness = (
        1.0 + (torch.rand(rand_shape, device=images.device, dtype=images.dtype) * 2.0 - 1.0) * 0.35 * strength
    )
    contrast = (
        1.0 + (torch.rand(rand_shape, device=images.device, dtype=images.dtype) * 2.0 - 1.0) * 0.50 * strength
    )
    saturation = (
        1.0 + (torch.rand(rand_shape, device=images.device, dtype=images.dtype) * 2.0 - 1.0) * 0.50 * strength
    )
    images = images * brightness
    spatial_mean = images.mean(dim=(-2, -1), keepdim=True)
    images = (images - spatial_mean) * contrast + spatial_mean
    if images.shape[1] >= 3:
        gray = images[:, :3].mean(dim=1, keepdim=True)
        rgb = (images[:, :3] - gray) * saturation + gray
        images = torch.cat((rgb, images[:, 3:]), dim=1) if images.shape[1] > 3 else rgb

    if torch.rand((), device=images.device) < 0.5 * strength:
        images = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
    noise = torch.randn_like(images) * (0.04 * strength)
    images = (images + noise).clamp(0.0, 1.0)
    augmented = (images - mean) / std

    if samples.mask is not None:
        augmented = torch.where(samples.mask[:, None], tensors, augmented)
    return NestedTensor(augmented, samples.mask)


class SPCConsistency(nn.Module):
    """Compute semantic feature consistency (SCC) and DETR query consistency.

    Args:
        confidence_threshold: Minimum Teacher sigmoid confidence for a query.
        max_queries: Maximum number of Teacher object queries matched per image.
        query_cost: Weight of query-embedding cosine distance in matching.
        class_cost: Weight of class-probability distance in matching.
        bbox_cost: Weight of box L1 distance in matching.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_queries: int = 100,
        query_cost: float = 1.0,
        class_cost: float = 2.0,
        bbox_cost: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.max_queries = max_queries
        self.query_cost = query_cost
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.eps = eps

    def _instance_normalize(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize each instance and channel over valid spatial positions."""
        valid = (~mask).unsqueeze(1).to(feature.dtype)
        count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean = (feature * valid).sum(dim=(-2, -1), keepdim=True) / count
        variance = ((feature - mean).square() * valid).sum(dim=(-2, -1), keepdim=True) / count
        return (feature - mean) * torch.rsqrt(variance + self.eps) * valid

    def semantic_feature_loss(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute masked SCC after per-instance feature normalization.

        Args:
            student_features: Selected Student feature pyramid levels.
            teacher_features: Corresponding detached Teacher levels.
            masks: Padding mask for each feature level.

        Returns:
            Mean normalized feature consistency loss.

        Raises:
            ValueError: If the three sequences do not have equal non-zero length.
        """
        if not student_features or not (len(student_features) == len(teacher_features) == len(masks)):
            raise ValueError("student_features, teacher_features, and masks must have equal non-zero length.")
        losses = []
        for student, teacher, mask in zip(student_features, teacher_features, masks, strict=True):
            student_float = student.float()
            teacher_float = teacher.detach().float()
            student_normalized = self._instance_normalize(student_float, mask)
            teacher_normalized = self._instance_normalize(teacher_float, mask)
            valid = (~mask).unsqueeze(1).to(student_float.dtype)
            denominator = (valid.sum() * student.shape[1]).clamp_min(1.0)
            losses.append(((student_normalized - teacher_normalized).square() * valid).sum() / denominator)
        return torch.stack(losses).mean()

    def query_prediction_loss(
        self,
        student_logits: torch.Tensor,
        student_boxes: torch.Tensor,
        student_queries: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_boxes: torch.Tensor,
        teacher_queries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Hungarian-match Teacher/Student queries and compute prediction consistency.

        Args:
            student_logits: Student class logits of shape ``[B, Qs, C]``.
            student_boxes: Student normalized cxcywh boxes of shape ``[B, Qs, 4]``.
            student_queries: Student decoder query embeddings of shape ``[B, Qs, D]``.
            teacher_logits: Teacher class logits of shape ``[B, Qt, C]``.
            teacher_boxes: Teacher normalized cxcywh boxes of shape ``[B, Qt, 4]``.
            teacher_queries: Teacher decoder query embeddings of shape ``[B, Qt, D]``.

        Returns:
            Classification and bounding-box consistency losses.
        """
        classification_losses: list[torch.Tensor] = []
        box_losses: list[torch.Tensor] = []
        for batch_index in range(student_logits.shape[0]):
            teacher_probability = teacher_logits[batch_index].detach().float().sigmoid()
            student_probability = student_logits[batch_index].float().sigmoid()
            confidence = teacher_probability.max(dim=-1).values
            candidate_indices = torch.nonzero(confidence >= self.confidence_threshold, as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            if candidate_indices.numel() > self.max_queries:
                top = confidence[candidate_indices].topk(self.max_queries).indices
                candidate_indices = candidate_indices[top]

            teacher_probability_selected = teacher_probability[candidate_indices]
            teacher_boxes_selected = teacher_boxes[batch_index, candidate_indices].detach().float()
            teacher_queries_selected = teacher_queries[batch_index, candidate_indices].detach()
            with torch.no_grad():
                class_distance = torch.cdist(teacher_probability_selected.float(), student_probability.float(), p=1)
                class_distance = class_distance / max(1, teacher_probability_selected.shape[-1])
                bbox_distance = torch.cdist(
                    teacher_boxes_selected.float(), student_boxes[batch_index].float(), p=1
                )
                teacher_query_unit = F.normalize(teacher_queries_selected.float(), dim=-1)
                student_query_unit = F.normalize(student_queries[batch_index].float(), dim=-1)
                query_distance = 1.0 - teacher_query_unit @ student_query_unit.transpose(0, 1)
                cost = (
                    self.class_cost * class_distance
                    + self.bbox_cost * bbox_distance
                    + self.query_cost * query_distance
                )
                teacher_match, student_match = linear_sum_assignment(cost.detach().cpu().numpy())
                teacher_match = torch.as_tensor(teacher_match, dtype=torch.long, device=student_logits.device)
                student_match = torch.as_tensor(student_match, dtype=torch.long, device=student_logits.device)

            target_probability = teacher_probability_selected[teacher_match].clamp(self.eps, 1.0 - self.eps)
            matched_probability = student_probability[student_match].clamp(self.eps, 1.0 - self.eps)
            bernoulli_kl = target_probability * (target_probability.log() - matched_probability.log())
            bernoulli_kl += (1.0 - target_probability) * (
                (1.0 - target_probability).log() - (1.0 - matched_probability).log()
            )
            classification_losses.append(bernoulli_kl.mean())

            target_boxes = teacher_boxes_selected[teacher_match]
            matched_boxes = student_boxes[batch_index, student_match].float()
            l1_loss = F.l1_loss(matched_boxes, target_boxes, reduction="mean")
            giou = generalized_box_iou(box_cxcywh_to_xyxy(matched_boxes), box_cxcywh_to_xyxy(target_boxes))
            box_losses.append(l1_loss + (1.0 - giou.diag()).mean())

        zero = student_logits.float().sum() * 0.0
        return {
            "loss_spc_query_cls": torch.stack(classification_losses).mean() if classification_losses else zero,
            "loss_spc_query_bbox": torch.stack(box_losses).mean() if box_losses else zero,
        }
