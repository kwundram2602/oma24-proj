"""Entry point for LWF-DLR segmentation model training — called by the SLURM batch script."""

import argparse
import logging
from pathlib import Path

import yaml

from oma24.training.lwf_farseg_trainer import LWF_FarSeg_Trainer
from oma24.training.lwf_deeplab_trainer import LWF_DeepLab_Trainer
from oma24.training.lwf_swin_upernet_trainer import LWF_SwinUPerNet_Trainer
from oma24.training.lwf_segformer_trainer import LWF_SegFormer_Trainer
from oma24.training.lwf_torchgeo_unet_trainer import LWF_TorchGeoUNet_Trainer

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
        experiment_id=cfg.get("experiment_id", "baseline"),
    )

    class_weights = cfg.get("class_weights", [1.0, 50.0, 5.0])
    model_name = cfg.get("model_name", "FarSeg")
    use_class_weights = cfg.get("use_class_weights", True)

    if model_name == "FarSeg":
        return LWF_FarSeg_Trainer(
            **common,
            backbone=cfg.get("backbone", "resnet34"),
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            test_split=cfg.get("test_split", 0.0),
        )

    elif model_name == "DeepLabV3":
        return LWF_DeepLab_Trainer(
            **common,
            backbone=cfg.get("backbone", "resnet50"),
            backbone_weights=cfg.get("backbone_weights", None),
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            test_split=cfg.get("test_split", 0.0),
        )

    elif model_name == "SwinUPerNet":
        return LWF_SwinUPerNet_Trainer(
            **common,
            backbone=cfg.get("backbone", "swin_tiny"),
            encoder_weights=cfg.get("encoder_weights", None),
            img_size=cfg.get("img_size", 1024),
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            test_split=cfg.get("test_split", 0.0),
        )

    elif model_name == "SegFormer":
        return LWF_SegFormer_Trainer(
            **common,
            backbone=cfg.get("backbone", "mit_b2"),
            encoder_weights=cfg.get("encoder_weights", "imagenet"),
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            test_split=cfg.get("test_split", 0.0),
        )

    elif model_name == "TorchGeoUNet":
        return LWF_TorchGeoUNet_Trainer(
            **common,
            pretrained_weights=cfg.get("pretrained_weights", "OAM_RGB_RESNET34_TCD"),
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            test_split=cfg.get("test_split", 0.0),
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Run LWF-DLR segmentation model training from a YAML config"
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
    logger.info("Config (%s):\n%s", args.config, yaml.dump(cfg, default_flow_style=False, allow_unicode=True).strip())
    logger.info("Starting training: %s", cfg.get("experiment_id"))
    trainer()


if __name__ == "__main__":
    main()
