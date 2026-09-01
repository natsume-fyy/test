# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Small on HazyDet with DETR-adapted SPC."""

from rfdetr import RFDETRSmall

DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/test/rf-detr-develop/output/hazydet_small_spc"


def main() -> None:
    """Run RF-DETR Small EMA Teacher-Student training with SPC enabled."""
    model = RFDETRSmall(use_spc=True)
    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=36,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        device="cuda",
        num_workers=8,
        use_ema=True,
        ema_decay=0.999,
        checkpoint_interval=5,
        early_stopping=False,
        spc_feature_levels=[-2, -1],
        spc_feature_loss_coef=1.0,
        spc_query_class_loss_coef=1.0,
        spc_query_bbox_loss_coef=2.0,
        spc_confidence_threshold=0.3,
        spc_max_queries=100,
        spc_strong_augmentation_strength=0.7,
    )


if __name__ == "__main__":
    main()
