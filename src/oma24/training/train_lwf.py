"""Entry point for LWF-DLR U-Net training — called by the SLURM batch script."""

import argparse
import logging
from pathlib import Path

import yaml

from oma24.training.lwf_unet_aspp_trainer import LWFUNetASPPTrainer
from oma24.training.lwf_unet_loss_trainer import LWFUNetLossTrainer
from oma24.training.lwf_unet_skeleton_trainer import LWFUNetSkeletonTrainer
from oma24.training.lwf_unet_trainer import LWFUNetTrainer
from oma24.training.lwf_farseg_trainer import LWF_FarSeg_Trainer

logger = logging.getLogger(__name__)


def build_trainer(cfg: dict):
    """Instantiate the correct trainer from config."""
    common = dict(
        user=cfg["user"],
        csv_name=cfg.get("csv_name", "dev_1000_file.csv"),
        epochs=cfg.get("epochs", 32),
        batch_size=cfg.get("batch_size", 12),
        num_workers=cfg.get("num_workers", 16),
        prefetch_factor=cfg.get("prefetch_factor", 8),
        experiment_id=cfg.get("experiment_id", "baseline_unet_slurm"),
    )

    class_weights = cfg.get("class_weights", [1.0, 50.0, 5.0])
    model_name = cfg.get("model_name", "UNet")
    use_class_weights = cfg.get("use_class_weights", True)
    use_dice_loss = cfg.get("use_dice_loss", True)
    dice_loss_weight = cfg.get("dice_loss_weight", 0.3)
    use_residual = cfg.get("use_residual", True)
    use_aspp = cfg.get("use_aspp", True)
    aspp_rates = tuple(cfg.get("aspp_rates", [3, 6, 9, 12]))

    if model_name == "UNet":
        if cfg.get("use_skeleton_trainer", False):
            return LWFUNetSkeletonTrainer(
                **common,
                use_class_weights=use_class_weights,
                class_weights=class_weights,
                use_dice_loss=use_dice_loss,
                dice_loss_weight=dice_loss_weight,
                use_residual=use_residual,
                use_aspp=use_aspp,
                aspp_rates=aspp_rates,
                use_distance=cfg.get("use_distance", False),
                distance_max=cfg.get("distance_max", 128),
                use_input_skeleton=cfg.get("use_input_skeleton", False),
                use_dual_head=cfg.get("use_dual_head", False),
                skeleton_class=cfg.get("skeleton_class", 1),
                skeleton_loss_weight=cfg.get("skeleton_loss_weight", 1.0),
            )

        if cfg.get("use_aspp_trainer", False):
            return LWFUNetASPPTrainer(
                **common,
                use_class_weights=use_class_weights,
                class_weights=class_weights,
                use_dice_loss=use_dice_loss,
                dice_loss_weight=dice_loss_weight,
                use_residual=use_residual,
                use_aspp=use_aspp,
                aspp_rates=aspp_rates,
            )

        if cfg.get("use_loss_trainer", False):
            return LWFUNetLossTrainer(
                **common,
                class_weights=class_weights,
                dice_loss_weight=dice_loss_weight,
            )

        return LWFUNetTrainer(**common)

    elif model_name == "FarSeg":
        backbone = cfg.get("backbone", "resnet34")
        return LWF_FarSeg_Trainer(
            **common,
            backbone=backbone,
            use_class_weights=use_class_weights,
            use_dice_loss=use_dice_loss,
                dice_loss_weight=dice_loss_weight,
                use_residual=use_residual,
                use_aspp=use_aspp,
                aspp_rates=aspp_rates
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Run LWF-DLR U-Net training from a YAML config"
    )
    parser.add_argument("--config", type=Path, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    trainer = build_trainer(cfg)
    logger.info("Starting training: %s", cfg.get("experiment_id"))
    trainer()


if __name__ == "__main__":
    main()
