# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for DETR-adapted semantic and query prediction consistency."""

import torch

from rfdetr.training.spc import SPCConsistency, make_strong_view
from rfdetr.utilities.tensors import NestedTensor


def test_scc_removes_per_instance_channel_style() -> None:
    """Instance normalization should remove channel-wise affine style changes."""
    teacher = torch.randn(2, 8, 5, 7)
    student = teacher * 2.5 + 4.0
    mask = torch.zeros(2, 5, 7, dtype=torch.bool)
    module = SPCConsistency(confidence_threshold=0.0)

    loss = module.semantic_feature_loss([student], [teacher], [mask])

    assert loss.item() < 1e-5


def test_qpc_hungarian_matching_handles_query_permutation() -> None:
    """Permuting otherwise identical queries should produce nearly zero QPC loss."""
    teacher_logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [6.0, -6.0]]])
    teacher_boxes = torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3]]])
    teacher_queries = torch.randn(1, 3, 16)
    permutation = torch.tensor([2, 0, 1])
    student_logits = teacher_logits[:, permutation].clone().requires_grad_()
    student_boxes = teacher_boxes[:, permutation].clone().requires_grad_()
    student_queries = teacher_queries[:, permutation].clone().requires_grad_()
    module = SPCConsistency(confidence_threshold=0.0, max_queries=3)

    losses = module.query_prediction_loss(
        student_logits,
        student_boxes,
        student_queries,
        teacher_logits,
        teacher_boxes,
        teacher_queries,
    )

    assert losses["loss_spc_query_cls"].item() < 1e-5
    assert losses["loss_spc_query_bbox"].item() < 1e-5


def test_strong_view_preserves_geometry_and_padding() -> None:
    """Strong augmentation should preserve tensor/mask shapes and padded pixels."""
    tensors = torch.randn(2, 3, 12, 16)
    mask = torch.zeros(2, 12, 16, dtype=torch.bool)
    mask[:, :, -3:] = True
    samples = NestedTensor(tensors, mask)

    strong = make_strong_view(samples, strength=0.8)

    assert strong.tensors.shape == samples.tensors.shape
    assert torch.equal(strong.mask, samples.mask)
    assert torch.equal(strong.tensors.masked_select(mask[:, None]), tensors.masked_select(mask[:, None]))
