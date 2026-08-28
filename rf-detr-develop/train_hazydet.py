# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Small on HazyDet with spectral perturbation enabled."""

from rfdetr import RFDETRSmall

DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/test/rf-detr-develop/output/hazydet_small_sp"


def main() -> None:
    """Run RF-DETR Small training with SP augmentation enabled."""
    model = RFDETRSmall()
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
        checkpoint_interval=5,
        early_stopping=False,
        sp_prob=0.5,
        sp_v1_scale=0.005,
        sp_v2_scale=0.7,
    )


if __name__ == "__main__":
    main()
