"""Train RF-DETR Small with HBS on the HazyDet dataset."""

import sys
from pathlib import Path


# Always import the RF-DETR source tree next to this script. This prevents an
# older editable installation elsewhere on the machine from shadowing the HBS
# implementation in this repository.
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

import rfdetr  # noqa: E402
from rfdetr import RFDETRSmall  # noqa: E402


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_hbs_official"


def main() -> None:
    """Configure and train the HBS-enabled RF-DETR Small model."""
    print(f"Using RF-DETR source: {Path(rfdetr.__file__).resolve()}")
    model = RFDETRSmall(
        hbs_enabled=True,
        hbs_reduction=4,
        device="cuda",
    )

    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=36,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        num_workers=8,
        use_ema=True,
        checkpoint_interval=5,
        early_stopping=False,
        # SET uses lambda=1.0; 0.5 made the HBS branch too weak to shape the backbone.
        hbs_loss_coef=1.0,
    )


if __name__ == "__main__":
    main()
