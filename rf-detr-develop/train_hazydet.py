# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Small on HazyDet with fog-aware frequency regulation."""

from rfdetr import RFDETRSmall


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/test/rf-detr-develop/output/hazydet_small_fog_frequency_residual"


def main() -> None:
    """Run RF-DETR Small training with fog-aware frequency regulation."""
    model = RFDETRSmall(
        use_fog_frequency_regulator=True,
        fog_frequency_probability=0.8,
        fog_frequency_tau1_range=(0.03, 0.18),
        fog_frequency_tau2_range=(0.60, 0.92),
        fog_frequency_max_low_strength=0.35,
        fog_frequency_max_high_strength=0.28,
        fog_frequency_warmup_fraction=0.1,
        fog_frequency_residual_mix=0.25,
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
