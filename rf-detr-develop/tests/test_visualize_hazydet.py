# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the HazyDet visualization utility."""

from pathlib import Path

import numpy as np

from visualize_hazydet import ImageResult, match_detections


def test_match_detections_counts_true_false_positives_and_false_negatives() -> None:
    """Matching should require both class agreement and sufficient IoU."""
    record = ImageResult(
        image_id=1,
        path=Path("image.jpg"),
        width=100,
        height=100,
        gt_boxes=np.asarray([[10, 10, 30, 30], [60, 60, 80, 80]], dtype=float),
        gt_classes=np.asarray([0, 1], dtype=int),
        object_count=2,
        mean_area_ratio=0.04,
        small_ratio=0.0,
        brightness=0.5,
        contrast=0.2,
        saturation=0.1,
        pred_boxes=np.asarray([[10, 10, 30, 30], [60, 60, 80, 80]], dtype=float),
        pred_classes=np.asarray([0, 0], dtype=int),
        pred_scores=np.asarray([0.9, 0.8], dtype=float),
    )

    match_detections(record, iou_threshold=0.5)

    assert record.tp == 1
    assert record.fp == 1
    assert record.fn == 1
    assert record.precision == 0.5
    assert record.recall == 0.5
    assert record.f1 == 0.5
