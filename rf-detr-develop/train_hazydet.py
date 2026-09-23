# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fine-tune RF-DETR Small with frequency refinement on HazyDet."""

from rfdetr import RFDETRSmall


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_frm"

# RESUME_PATH = "/root/autodl-tmp/rf-detr/output/hazydet_small_frm/last.ckpt"


def main() -> None:
    """Train RF-DETR Small with FRM on its default P4 projector output."""
    model = RFDETRSmall(frm_levels=["P4"])

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
        # resume=RESUME_PATH,
    )


if __name__ == "__main__":
    main()
