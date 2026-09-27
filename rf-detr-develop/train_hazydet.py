# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Small with CBAM on HazyDet and visualize results by target size."""

from __future__ import annotations

from typing import Any

from rfdetr import RFDETRSmall

from visualize_hazydet import generate_representative_visualization

DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/baseline/rf-detr-develop/output/hazydet_small_cbam"

VISUALIZE_AFTER_TRAINING = True
SAMPLES_PER_GROUP = 3
FIXED_SAMPLE_FILE = (
    "/root/autodl-tmp/baseline/rf-detr-develop/output/"
    "hazydet_fixed_samples_valid_3_per_size_group.json"
)

HBS_ENABLED = True
HBS_REDUCTION = 4
HBS_LOSS_COEF = 0.25


def _supported_options(config_class: type, requested: dict[str, Any], component: str) -> dict[str, Any]:
    """Keep optional custom-branch settings only when the active config supports them."""
    model_fields = getattr(config_class, "model_fields", {})
    supported = {name: value for name, value in requested.items() if name in model_fields}
    unsupported = sorted(set(requested) - set(supported))
    if unsupported:
        print(
            f"[配置提示] 当前版本的 {component} 不支持 {', '.join(unsupported)}；"
            "这些参数将被跳过，Backbone 中的残差式 CBAM 仍会正常启用。"
        )
    return supported


def main() -> None:
    """Train the model and optionally generate fixed size-group visualizations."""
    model_options = _supported_options(
        RFDETRSmall._model_config_class,
        {"hbs_enabled": HBS_ENABLED, "hbs_reduction": HBS_REDUCTION},
        "ModelConfig",
    )
    model = RFDETRSmall(**model_options)

    train_options: dict[str, Any] = {
        "dataset_dir": DATASET_DIR,
        "output_dir": OUTPUT_DIR,
        "epochs": 36,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "lr": 1e-4,
        "device": "cuda",
        "num_workers": 8,
        "use_ema": False,
        "checkpoint_interval": 5,
        "early_stopping": False,
    }
    train_options.update(
        _supported_options(
            model._train_config_class,
            {"hbs_loss_coef": HBS_LOSS_COEF},
            "TrainConfig",
        )
    )
    model.train(**train_options)

    if VISUALIZE_AFTER_TRAINING:
        generate_representative_visualization(
            dataset_dir=DATASET_DIR,
            checkpoint=f"{OUTPUT_DIR}/checkpoint_best_total.pth",
            output_dir=f"{OUTPUT_DIR}/representative_visualization",
            split="valid",
            confidence_threshold=0.30,
            iou_threshold=0.50,
            sample_file=FIXED_SAMPLE_FILE,
            samples_per_group=SAMPLES_PER_GROUP,
            device="cuda",
        )


if __name__ == "__main__":
    main()
