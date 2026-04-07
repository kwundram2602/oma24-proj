import logging
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchgeo.models import unet, Unet_Weights

from oma24.training.train_utils import (
    build_train_val_test_loaders,
    compute_confusion_matrix,
    compute_map_iou_thresholds,
    evaluate_model,
    generate_plots,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

_PRETRAINED_WEIGHTS = {
    "OAM_RGB_RESNET34_TCD": Unet_Weights.OAM_RGB_RESNET34_TCD,   # ResNet34, 3ch, tree-canopy
    "OAM_RGB_RESNET50_TCD": Unet_Weights.OAM_RGB_RESNET50_TCD,   # ResNet50, 3ch, tree-canopy
    "SENTINEL2_3CLASS_FTW": Unet_Weights.SENTINEL2_3CLASS_FTW,   # EfficientNet-B3, 8ch, field boundaries
    None: None,
}


class LWF_TorchGeoUNet_Trainer:
    """TorchGeo UNet (smp-based) trainer for LWF-DLR vegetation segmentation.

    Loads a pre-trained torchgeo UNet checkpoint and fine-tunes it with
    a randomly re-initialized segmentation head for ``num_classes`` output classes.

    ``OAM_RGB_RESNET34_TCD`` (tree-canopy, RGB) is the most thematically
    similar pre-training to LWF-DLR vegetation and is used as the default.

    Model output is a plain tensor — no dict wrapper.
    """

    def __init__(
        self,
        user: str,
        csv_name: str = "full_set_file.csv",
        pretrained_weights: str | None = "OAM_RGB_RESNET34_TCD",
        epochs: int = 32,
        batch_size: int = 12,
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        num_workers: int = 16,
        prefetch_factor: int = 8,
        val_split: float = 0.2,
        test_split: float = 0.0,
        val_every_n_steps: int = 250,
        val_every_n_steps_warmup: int = 50,
        warmup_steps: int = 100,
        save_every_n_steps: int = 20,
        experiment_group: str = "LWF-DLR",
        experiment_id: str = "torchgeo_unet_baseline",
        use_class_weights: bool = True,
        class_weights: list = [1.0, 50.0, 5.0],
    ):
        """
        Parameters
        ----------
        pretrained_weights : str or None
            Which torchgeo pre-trained checkpoint to load.
            ``"OAM_RGB_RESNET34_TCD"`` — ResNet34 encoder, RGB, tree canopy (recommended).
            ``"OAM_RGB_RESNET50_TCD"`` — ResNet50 encoder, RGB, tree canopy.
            ``"SENTINEL2_3CLASS_FTW"`` — EfficientNet-B3, 8-channel Sentinel-2 (requires in_channels=8).
            ``None``                   — random init, ResNet34 encoder by default.
        """
        if pretrained_weights not in _PRETRAINED_WEIGHTS:
            raise ValueError(
                f"Unsupported pretrained_weights '{pretrained_weights}'. "
                f"Choose from {list(_PRETRAINED_WEIGHTS)}"
            )

        self.user = user
        self.csv_name = csv_name
        self.pretrained_weights = pretrained_weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.val_split = val_split
        self.test_split = test_split
        self.val_every_n_steps = val_every_n_steps
        self.val_every_n_steps_warmup = val_every_n_steps_warmup
        self.warmup_steps = warmup_steps
        self.save_every_n_steps = save_every_n_steps
        self.experiment_group = experiment_group
        self.experiment_id = experiment_id
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights

        self.root_hpc = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026")
        self.user_path = self.root_hpc / user
        self.data_root = self.root_hpc / "data" / "LWF-DLR"

        self.experiment_dir = self.user_path / f"experiments/{experiment_group}" / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.device = None
        self.model = None
        self.writer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._last_val_metrics = None
        self.criterion_ce = None
        self.optimizer = None
        self.scheduler = None

    def setup(self) -> None:
        """Set up device, model, data loaders, loss, optimiser, scheduler."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        weights_enum = _PRETRAINED_WEIGHTS[self.pretrained_weights]

        # Determine input channels from the pre-trained weight metadata
        in_channels = weights_enum.meta["in_chans"] if weights_enum is not None else 3

        csv_path = self.data_root / self.csv_name
        npz_names = pd.read_csv(csv_path)["npz_path"].tolist()

        self.train_loader, self.val_loader, self.test_loader = build_train_val_test_loaders(
            npz_names=npz_names,
            data_root=self.data_root,
            val_split=self.val_split,
            test_split=self.test_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            repeat_channels=in_channels,
        )

        num_classes = len(self.class_weights)
        # torchgeo's unet() automatically re-initialises the segmentation head
        # when classes != weights.meta['num_classes']
        self.model = unet(weights=weights_enum, classes=num_classes).to(self.device)

        encoder_name = (weights_enum.meta["encoder"] if weights_enum is not None else "resnet34")
        logger.info(
            "TorchGeoUNet | weights: %s | encoder: %s | in_channels: %d | classes: %d | parameters: %s",
            self.pretrained_weights, encoder_name, in_channels, num_classes,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        if self.use_class_weights:
            class_weights_tensor = torch.tensor(self.class_weights, device=self.device)
            self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.criterion_ce = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0.0
        )
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir))

    def train(self) -> None:
        """Step-based training loop."""
        global_step = 0
        validation_cycle = 0
        best_val_loss = float("inf")
        train_running_loss = 0.0
        train_samples = 0
        train_loss_history: list[tuple[int, float]] = []
        val_loss_history: list[tuple[int, float]] = []

        logger.info("=" * 80)
        logger.info("TRAINING — %s", self.experiment_id)
        logger.info("Epochs: %d | val every %d steps (warmup: %d for first %d steps)",
                    self.epochs, self.val_every_n_steps,
                    self.val_every_n_steps_warmup, self.warmup_steps)
        logger.info("=" * 80)

        for epoch in range(self.epochs):
            logger.info("\nEpoch %d/%d", epoch + 1, self.epochs)
            self.model.train()

            for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion_ce(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_size = inputs.size(0)
                train_running_loss += loss.item() * batch_size
                train_samples += batch_size

                self.writer.add_scalar("Train/learning_rate",
                                       self.optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

                val_freq = (self.val_every_n_steps_warmup if global_step <= self.warmup_steps
                            else self.val_every_n_steps)

                if global_step % val_freq == 0:
                    validation_cycle += 1

                    if train_samples > 0:
                        interval_train_loss = train_running_loss / train_samples
                        self.writer.add_scalar("Train/total_loss_interval",
                                               interval_train_loss, global_step)
                        train_loss_history.append((global_step, interval_train_loss))
                    train_running_loss = 0.0
                    train_samples = 0

                    val_metrics = evaluate_model(
                        self.model, self.val_loader, self.criterion_ce, self.device,
                        num_classes=len(self.class_weights), prefix="val",
                    )
                    self._last_val_metrics = val_metrics
                    val_loss = val_metrics["val_loss"]
                    val_loss_history.append((global_step, val_loss))

                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f"Val/{key}", value, global_step)

                    logger.info("[Step %d | Cycle %d | Epoch %d]  Val Loss: %.4f  mIoU: %.4f",
                                global_step, validation_cycle, epoch + 1,
                                val_loss, val_metrics["mean_iou"])

                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    if global_step % self.save_every_n_steps == 0 or is_best:
                        save_checkpoint(
                            self.checkpoint_dir, self.model, self.optimizer,
                            epoch, global_step, val_loss, is_best=is_best,
                        )

                    self.model.train()

            self.scheduler.step()
            logger.info("Epoch %d complete. LR: %.6f", epoch + 1,
                        self.optimizer.param_groups[0]["lr"])

        # ── Final model save ─────────────────────────────────────────────────
        final_path = self.checkpoint_dir / "final_model.pth"
        torch.save({
            "epoch": self.epochs,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, final_path)
        logger.info("\nTraining complete. Steps: %d | Best Val Loss: %.4f", global_step, best_val_loss)

        # ── Post-training evaluation + plots ─────────────────────────────────
        logger.info("=" * 60)
        logger.info("POST-TRAINING EVALUATION")
        logger.info("=" * 60)

        val_map_data = None
        val_conf_matrix = None
        if self._last_val_metrics is not None:
            logger.info("Computing val mAP...")
            val_map_data = compute_map_iou_thresholds(self.model, self.val_loader, self.device)
            self.writer.add_scalar("Val/overall_map", val_map_data["overall_map"], global_step)
            logger.info("Val mAP: %.4f", val_map_data["overall_map"])

            logger.info("Computing val confusion matrix...")
            val_conf_matrix = compute_confusion_matrix(
                self.model, self.val_loader, self.device,
                num_classes=len(self.class_weights),
            )

        test_metrics = None
        test_map_data = None
        test_conf_matrix = None
        if self.test_loader is not None:
            logger.info("Running test evaluation...")
            test_metrics = evaluate_model(
                self.model, self.test_loader, self.criterion_ce, self.device,
                num_classes=len(self.class_weights), prefix="test",
            )
            for key, value in test_metrics.items():
                self.writer.add_scalar(f"Test/{key}", value, global_step)
            logger.info("Test Loss: %.4f  Test mIoU: %.4f",
                        test_metrics["test_loss"], test_metrics["mean_iou"])

            logger.info("Computing test mAP...")
            test_map_data = compute_map_iou_thresholds(self.model, self.test_loader, self.device)
            self.writer.add_scalar("Test/overall_map", test_map_data["overall_map"], global_step)
            logger.info("Test mAP: %.4f", test_map_data["overall_map"])

            logger.info("Computing test confusion matrix...")
            test_conf_matrix = compute_confusion_matrix(
                self.model, self.test_loader, self.device,
                num_classes=len(self.class_weights),
            )
        else:
            logger.info("No test split — skipping test evaluation.")

        if self._last_val_metrics is not None and val_map_data is not None:
            generate_plots(
                self.writer, self.plots_dir,
                val_metrics=self._last_val_metrics,
                test_metrics=test_metrics,
                val_map_data=val_map_data,
                test_map_data=test_map_data,
                global_step=global_step,
                model_name=f"TorchGeoUNet-{self.pretrained_weights}",
                train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
                val_conf_matrix=val_conf_matrix,
                test_conf_matrix=test_conf_matrix,
            )
            logger.info("Plots saved to: %s", self.plots_dir)

        self.writer.close()

    def __call__(self) -> str:
        logger.info("Starting training: %s", self.experiment_id)
        self.setup()
        self.train()
        return str(self.checkpoint_dir / "final_model.pth")
