# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Small with FBM on HazyDet."""

from rfdetr import RFDETRSmall

DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/test/rf-detr-develop/output/hazydet_small_fbm"


def main() -> None:
    """Run RF-DETR Small training with FBM enabled."""
    model = RFDETRSmall(
        use_fbm=True,
        fbm_k_list=[2, 4, 8],
        fbm_lowfreq_att=False,
        fbm_spatial_group=32,
    )

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
    )


if __name__ == "__main__":
    main()
