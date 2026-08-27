# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Small with FDAM FreqScale on HazyDet."""

from rfdetr import RFDETRSmall

DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_freqscale"


def main() -> None:
    """Run RF-DETR Small training with FreqScale enabled."""
    model = RFDETRSmall(
        freq_scale=True,
        freq_scale_group=32,
        freq_scale_num_filters=4,
        freq_scale_init_scale=1e-5,
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
