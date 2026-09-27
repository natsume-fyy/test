# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for HazyDet representative visualization helpers."""

from pathlib import Path

import numpy as np

from visualize_hazydet import ImageRecord, classify_target_size, match_detections


def test_classify_target_size_uses_coco_area_thresholds() -> None:
    """Target sizes should follow the standard COCO 32/96 pixel boundaries."""
    assert classify_target_size(31 * 31) == "small"
    assert classify_target_size(32 * 32) == "medium"
    assert classify_target_size(95 * 95) == "medium"
    assert classify_target_size(96 * 96) == "large"


def test_match_detections_reports_metrics_by_target_size() -> None:
    """Matched and missed ground truths should be attributed to their size group."""
    record = ImageRecord(
        image_id=1,
        path=Path("image.jpg"),
        width=200,
        height=200,
        gt_boxes=np.asarray([[10, 10, 30, 30], [60, 60, 120, 120], [20, 100, 140, 199]], dtype=float),
        gt_classes=np.asarray([0, 1, 2], dtype=int),
        gt_sizes=("small", "medium", "large"),
        pred_boxes=np.asarray([[10, 10, 30, 30], [62, 62, 118, 118], [150, 150, 180, 180]], dtype=float),
        pred_classes=np.asarray([0, 1, 2], dtype=int),
        pred_scores=np.asarray([0.9, 0.8, 0.7], dtype=float),
    )

    match_detections(record, iou_threshold=0.5)

    assert record.size_metrics["small"].tp == 1
    assert record.size_metrics["medium"].tp == 1
    assert record.size_metrics["large"].fn == 1
    assert record.fp == 1
