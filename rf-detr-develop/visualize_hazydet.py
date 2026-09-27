# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Generate fixed HazyDet visualizations and metrics for small, medium, and large targets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

SIZE_GROUPS = ("small", "medium", "large")
SIZE_LABELS = {"small": "Small targets", "medium": "Medium targets", "large": "Large targets"}
SIZE_COLORS = {"small": "#2ca02c", "medium": "#ffbf00", "large": "#d62728"}
SMALL_AREA = 32**2
LARGE_AREA = 96**2


@dataclass
class SizeMetrics:
    """Detection counts and IoUs attributed to one ground-truth size group."""

    gt: int = 0
    tp: int = 0
    fn: int = 0
    ious: list[float] = field(default_factory=list)


@dataclass
class ImageRecord:
    """One annotated image plus predictions and matching results."""

    image_id: int
    path: Path
    width: int
    height: int
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    gt_sizes: tuple[str, ...]
    pred_boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=float))
    pred_classes: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    pred_scores: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    matched_gt: set[int] = field(default_factory=set)
    matched_pred: set[int] = field(default_factory=set)
    matched_ious: dict[int, float] = field(default_factory=dict)
    size_metrics: dict[str, SizeMetrics] = field(default_factory=dict)
    tp: int = 0
    fp: int = 0
    fn: int = 0


def classify_target_size(area: float) -> str:
    """Classify a bounding-box area using standard COCO size thresholds."""
    if area < SMALL_AREA:
        return "small"
    if area < LARGE_AREA:
        return "medium"
    return "large"


def _xywh_to_xyxy(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert COCO ``xywh`` boxes to an ``N x 4`` xyxy array."""
    result = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    if len(result):
        result[:, 2] += result[:, 0]
        result[:, 3] += result[:, 1]
    return result


def load_validation_records(dataset_dir: str | Path, split: str) -> tuple[list[ImageRecord], dict[int, str]]:
    """Load COCO annotations and attach a COCO size group to every ground-truth box."""
    split_dir = Path(dataset_dir) / split
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = sorted(coco.get("categories", []), key=lambda item: int(item["id"]))
    category_to_label = {int(category["id"]): index for index, category in enumerate(categories)}
    label_to_name = {index: str(category["name"]) for index, category in enumerate(categories)}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    records = []
    for image_info in sorted(coco.get("images", []), key=lambda item: int(item["id"])):
        image_id = int(image_info["id"])
        annotations = annotations_by_image.get(image_id, [])
        boxes_xywh = [annotation["bbox"] for annotation in annotations]
        gt_boxes = _xywh_to_xyxy(boxes_xywh)
        gt_classes = np.asarray(
            [category_to_label[int(annotation["category_id"])] for annotation in annotations], dtype=int
        )
        gt_sizes = tuple(classify_target_size(float(box[2]) * float(box[3])) for box in boxes_xywh)
        records.append(
            ImageRecord(
                image_id=image_id,
                path=split_dir / str(image_info["file_name"]),
                width=int(image_info["width"]),
                height=int(image_info["height"]),
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                gt_sizes=gt_sizes,
            )
        )
    return records, label_to_name


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    top_left = np.maximum(box_a[:2], box_b[:2])
    bottom_right = np.minimum(box_a[2:], box_b[2:])
    intersection = float(np.prod(np.maximum(bottom_right - top_left, 0.0)))
    area_a = float(np.prod(np.maximum(box_a[2:] - box_a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(box_b[2:] - box_b[:2], 0.0)))
    return intersection / max(area_a + area_b - intersection, 1e-12)


def match_detections(record: ImageRecord, iou_threshold: float) -> None:
    """Greedily match same-class predictions and calculate metrics by GT target size."""
    record.matched_gt.clear()
    record.matched_pred.clear()
    record.matched_ious.clear()
    candidates: list[tuple[float, int, int]] = []
    for gt_index, (gt_box, gt_class) in enumerate(zip(record.gt_boxes, record.gt_classes)):
        for pred_index, (pred_box, pred_class) in enumerate(zip(record.pred_boxes, record.pred_classes)):
            if int(gt_class) == int(pred_class):
                iou = _iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    candidates.append((iou, gt_index, pred_index))

    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in record.matched_gt or pred_index in record.matched_pred:
            continue
        record.matched_gt.add(gt_index)
        record.matched_pred.add(pred_index)
        record.matched_ious[gt_index] = iou

    record.tp = len(record.matched_gt)
    record.fp = len(record.pred_boxes) - len(record.matched_pred)
    record.fn = len(record.gt_boxes) - len(record.matched_gt)
    record.size_metrics = {group: SizeMetrics() for group in SIZE_GROUPS}
    for gt_index, group in enumerate(record.gt_sizes):
        metrics = record.size_metrics[group]
        metrics.gt += 1
        if gt_index in record.matched_gt:
            metrics.tp += 1
            metrics.ious.append(record.matched_ious[gt_index])
        else:
            metrics.fn += 1


def run_inference(
    records: Sequence[ImageRecord],
    checkpoint: str | Path,
    confidence_threshold: float,
    iou_threshold: float,
    device: str,
) -> None:
    """Run RF-DETR inference and populate predictions and matching results."""
    from rfdetr import RFDETR

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = RFDETR.from_checkpoint(str(checkpoint), device=device)
    model.optimize_for_inference(compile=False)

    for index, record in enumerate(records, start=1):
        detections = model.predict(str(record.path), threshold=confidence_threshold, device=device)
        record.pred_boxes = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
        record.pred_classes = np.asarray(detections.class_id, dtype=int).reshape(-1)
        record.pred_scores = np.asarray(detections.confidence, dtype=float).reshape(-1)
        match_detections(record, iou_threshold)
        print(f"Inference: {index}/{len(records)}", end="\r")
    print()


def select_representatives(
    records: Sequence[ImageRecord], samples_per_group: int
) -> dict[str, list[ImageRecord]]:
    """Select distinct images rich in small, medium, or large ground-truth targets."""
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be greater than zero")
    selected: dict[str, list[ImageRecord]] = {group: [] for group in SIZE_GROUPS}
    used: set[int] = set()
    for group in SIZE_GROUPS:
        candidates = [record for record in records if group in record.gt_sizes and record.image_id not in used]
        candidates.sort(
            key=lambda record: (
                -record.gt_sizes.count(group),
                -record.gt_sizes.count(group) / max(len(record.gt_sizes), 1),
                record.path.name,
                record.image_id,
            )
        )
        if len(candidates) < samples_per_group:
            raise ValueError(
                f"Need {samples_per_group} distinct images containing {group} targets, but only "
                f"{len(candidates)} are available after selecting earlier groups."
            )
        selected[group] = candidates[:samples_per_group]
        used.update(record.image_id for record in selected[group])
    return selected


def load_or_create_fixed_selection(
    records: Sequence[ImageRecord], sample_file: str | Path, split: str, samples_per_group: int
) -> dict[str, list[ImageRecord]]:
    """Reuse a size-aware sample manifest or replace an incompatible legacy manifest."""
    sample_file = Path(sample_file)
    by_id = {record.image_id: record for record in records}
    if sample_file.exists():
        with sample_file.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        groups = manifest.get("groups", {})
        compatible = (
            manifest.get("version") == 2
            and manifest.get("split") == split
            and manifest.get("samples_per_group") == samples_per_group
            and set(groups) == set(SIZE_GROUPS)
            and all(len(groups[group]) == samples_per_group for group in SIZE_GROUPS)
        )
        if compatible:
            try:
                return {group: [by_id[int(image_id)] for image_id in groups[group]] for group in SIZE_GROUPS}
            except KeyError:
                print("Fixed sample manifest references missing images; regenerating it.")
        else:
            print("Fixed sample manifest is not size-aware or has different settings; regenerating it.")

    selected = select_representatives(records, samples_per_group)
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 2,
        "split": split,
        "samples_per_group": samples_per_group,
        "size_definition": {"small": "area < 32^2", "medium": "32^2 <= area < 96^2", "large": "area >= 96^2"},
        "groups": {group: [record.image_id for record in selected[group]] for group in SIZE_GROUPS},
    }
    with sample_file.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return selected


def _draw_box(axis: Any, box: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
    """Draw a bounding box and compact label."""
    x1, y1, x2, y2 = box
    axis.add_patch(
        Rectangle(
            (x1, y1),
            max(x2 - x1, 0),
            max(y2 - y1, 0),
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            linestyle=linestyle,
        )
    )
    axis.text(
        x1,
        max(y1 - 2, 0),
        label,
        color="white",
        fontsize=6,
        bbox={"facecolor": color, "alpha": 0.8, "pad": 1},
    )


def _draw_record(axis: Any, record: ImageRecord, label_to_name: dict[int, str], focus_group: str) -> None:
    """Draw ground truths and predictions, highlighting the requested target-size group."""
    with Image.open(record.path) as image:
        axis.imshow(image.convert("RGB"))
    for gt_index, (box, class_id, size_group) in enumerate(zip(record.gt_boxes, record.gt_classes, record.gt_sizes)):
        color = SIZE_COLORS[size_group] if size_group == focus_group else "#7f7f7f"
        match_label = "TP" if gt_index in record.matched_gt else "FN"
        name = label_to_name.get(int(class_id), str(class_id))
        _draw_box(axis, box, color, f"GT {name} {size_group[0].upper()} {match_label}")
    for pred_index, (box, class_id, score) in enumerate(
        zip(record.pred_boxes, record.pred_classes, record.pred_scores)
    ):
        color = "#00bfff" if pred_index in record.matched_pred else "#ff00ff"
        name = label_to_name.get(int(class_id), str(class_id))
        _draw_box(axis, box, color, f"Pred {name} {score:.2f}", linestyle="--")
    group_metrics = record.size_metrics.get(focus_group, SizeMetrics())
    axis.set_title(
        f"{record.path.name}\n{SIZE_LABELS[focus_group]}: "
        f"GT={group_metrics.gt}, TP={group_metrics.tp}, FN={group_metrics.fn}",
        fontsize=8,
    )
    axis.axis("off")


def _aggregate_size_metrics(records: Sequence[ImageRecord]) -> dict[str, dict[str, float | int]]:
    """Aggregate ground-truth detection metrics for each size group."""
    result: dict[str, dict[str, float | int]] = {}
    for group in SIZE_GROUPS:
        gt = sum(record.size_metrics[group].gt for record in records)
        tp = sum(record.size_metrics[group].tp for record in records)
        fn = sum(record.size_metrics[group].fn for record in records)
        ious = [iou for record in records for iou in record.size_metrics[group].ious]
        result[group] = {
            "gt": gt,
            "tp": tp,
            "fn": fn,
            "recall": tp / gt if gt else 0.0,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
        }
    return result


def save_visualizations(
    records: Sequence[ImageRecord],
    selected: dict[str, list[ImageRecord]],
    label_to_name: dict[int, str],
    output_dir: str | Path,
) -> None:
    """Save combined/per-size figures, selected-sample metadata, and size metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_per_group = len(selected[SIZE_GROUPS[0]])
    fig, axes = plt.subplots(3, samples_per_group, figsize=(4 * samples_per_group, 10), squeeze=False)
    sample_rows: list[list[Any]] = []
    for row, group in enumerate(SIZE_GROUPS):
        group_dir = output_dir / f"{group}_targets"
        group_dir.mkdir(parents=True, exist_ok=True)
        for column, record in enumerate(selected[group]):
            _draw_record(axes[row, column], record, label_to_name, group)
            single_fig, single_axis = plt.subplots(figsize=(8, 6))
            _draw_record(single_axis, record, label_to_name, group)
            single_fig.tight_layout()
            single_fig.savefig(group_dir / f"{column + 1:02d}_{record.path.stem}.png", dpi=180)
            plt.close(single_fig)
            sample_rows.append(
                [
                    group,
                    column + 1,
                    record.image_id,
                    record.path.name,
                    record.gt_sizes.count(group),
                    record.tp,
                    record.fp,
                    record.fn,
                ]
            )
    fig.suptitle("HazyDet representative results by COCO target size\nGT: solid, predictions: dashed", fontsize=14)
    fig.savefig(output_dir / "representative_grid_by_size.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "representative_samples.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["size_group", "sample", "image_id", "file_name", "group_gt", "image_tp", "image_fp", "image_fn"]
        )
        writer.writerows(sample_rows)

    size_metrics = _aggregate_size_metrics(records)
    with (output_dir / "size_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["size_group", "gt", "tp", "fn", "recall", "mean_iou"])
        for group in SIZE_GROUPS:
            metrics = size_metrics[group]
            writer.writerow(
                [group, metrics["gt"], metrics["tp"], metrics["fn"], metrics["recall"], metrics["mean_iou"]]
            )
    with (output_dir / "size_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(size_metrics, file, ensure_ascii=False, indent=2)


def generate_representative_visualization(
    dataset_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "valid",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    sample_file: str | Path | None = None,
    samples_per_group: int = 3,
    device: str = "cuda",
) -> None:
    """Generate fixed small/medium/large target visualizations and aggregate metrics."""
    records, label_to_name = load_validation_records(dataset_dir, split)
    if sample_file is None:
        sample_file = Path(dataset_dir) / f"fixed_samples_{split}_by_size.json"
    selected = load_or_create_fixed_selection(records, sample_file, split, samples_per_group)
    run_inference(records, checkpoint, confidence_threshold, iou_threshold, device)
    save_visualizations(records, selected, label_to_name, output_dir)
    print(f"Representative visualizations saved to: {output_dir}")


def main() -> None:
    """Parse command-line arguments and generate HazyDet visualizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--sample-file")
    parser.add_argument("--samples-per-group", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    generate_representative_visualization(
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        sample_file=args.sample_file,
        samples_per_group=args.samples_per_group,
        device=args.device,
    )


if __name__ == "__main__":
    main()
