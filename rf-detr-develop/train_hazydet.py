# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Small with FBM on a COCO-format HazyDet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rfdetr import RFDETRSmall


def parse_args() -> argparse.Namespace:
    """Parse HazyDet training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/hazydet_fbm"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Build the FBM-enhanced model and start training."""
    args = parse_args()
    model = RFDETRSmall(
        use_fbm=True,
        fbm_k_list=[2, 4, 8],
        fbm_lowfreq_att=False,
        fbm_spatial_group=1,
    )
    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "resolution": args.resolution,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.resume is not None:
        train_kwargs["resume"] = str(args.resume)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
