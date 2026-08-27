"""Train RF-DETR Small with HBS on the HazyDet dataset."""

from rfdetr import RFDETRSmall


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_hbs"


def main() -> None:
    """Configure and train the HBS-enabled RF-DETR Small model."""
    model = RFDETRSmall(
        hbs_enabled=True,
        hbs_reduction=4,
        hbs_foreground_scale=0.1,
        hbs_attention_kernel_size=7,
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
        hbs_loss_coef=0.5,
    )


if __name__ == "__main__":
    main()
