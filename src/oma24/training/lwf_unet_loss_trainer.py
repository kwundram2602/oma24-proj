"""LWF-DLR U-Net segmentation training with Weighted CE + Dice Loss."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Dataset for loading NPZ files with segmentation data."""

    def __init__(self, npz_names, base_path):
        """
        Initialize the segmentation dataset.

        Parameters
        ----------
        npz_names : list[str]
            List of NPZ file names to load.
        base_path : Path
            Base path containing the npz directory.
        """
        self.npz_names = npz_names
        self.base_path = Path(base_path)
        self.npz_dir = self.base_path / "npz"

        # Define class remapping
        self.remap_dict = {
            0: 0,  # background
            1: 1,  # linear_vegetation
            2: 2,  # patchy_vegetation
            3: 2,  # patchy_vegetation
            4: 2,  # patchy_vegetation
            5: 2,  # patchy_vegetation
        }

        self.class_names = {
            0: "background",
            1: "linear_vegetation",
            2: "patchy_vegetation",
        }

    def __len__(self):
        return len(self.npz_names)

    def __getitem__(self, idx):
        # Build full path to a single NPZ file
        npz_path = self.npz_dir / self.npz_names[idx]

        # Load data from npz
        data = np.load(npz_path)
        label_mask = data["label"]  # Shape: (H, W)

        # Remap labels
        remapped_mask = np.zeros_like(label_mask, dtype=np.int64)
        for old_label, new_label in self.remap_dict.items():
            remapped_mask[label_mask == old_label] = new_label

        # Create binary input: 1 where any vegetation exists, 0 for background
        binary_mask = (remapped_mask > 0).astype(np.float32)

        # Convert to torch tensors
        x = torch.from_numpy(binary_mask).unsqueeze(0)  # Shape: (1, H, W)
        y = torch.from_numpy(remapped_mask).long()  # Shape: (H, W)

        return x, y


class DoubleConv(nn.Module):
    """Two consecutive convolution blocks with BatchNorm and ReLU.

    This is the basic building block of U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Basic U-Net for semantic segmentation.

    Args:
        in_channels: Number of input channels (1 for binary/grayscale)
        num_classes: Number of output classes (3 for our problem)
        base_channels: Number of channels in first layer (default: 32)
    """

    def __init__(self, in_channels=1, num_classes=3, base_channels=32):
        super().__init__()

        # ENCODER
        self.enc1 = DoubleConv(in_channels, base_channels)  # 32 channels
        self.enc2 = DoubleConv(base_channels, base_channels * 2)  # 64 channels
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)  # 128 channels
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)  # 256 channels

        self.pool = nn.MaxPool2d(2)  # Downsampling by 2

        # BOTTLENECK (Bottom of U)
        self.bottleneck = DoubleConv(
            base_channels * 8, base_channels * 16
        )  # 512 channels

        # DECODER
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # FINAL LAYER: 1×1 convolution to get class scores
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path (save features for skip connections)
        enc1 = self.enc1(x)  # 1024×1024×32
        enc2 = self.enc2(self.pool(enc1))  # 512×512×64
        enc3 = self.enc3(self.pool(enc2))  # 256×256×128
        enc4 = self.enc4(self.pool(enc3))  # 128×128×256

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # 64×64×512

        # Decoder path (with skip connections)
        dec4 = self.up4(bottleneck)  # 128×128×256
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concatenate skip connection
        dec4 = self.dec4(dec4)  # 128×128×256

        dec3 = self.up3(dec4)  # 256×256×128
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)  # 256×256×128

        dec2 = self.up2(dec3)  # 512×512×64
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)  # 512×512×64

        dec1 = self.up1(dec2)  # 1024×1024×32
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)  # 1024×1024×32

        # Final output
        out = self.out(dec1)  # 1024×1024×3
        return out


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks.

    Dice coefficient measures overlap between prediction and ground truth.
    Dice Loss = 1 - Dice coefficient

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        apply_to_classes: List of class indices to compute loss for (None = all classes)
    """

    def __init__(self, smooth=1.0, apply_to_classes=None):
        super().__init__()
        self.smooth = smooth
        self.apply_to_classes = apply_to_classes

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output (B, C, H, W) - raw scores before softmax
            targets: Ground truth (B, H, W) - class indices

        Returns:
            Dice loss (scalar)
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

        num_classes = probs.shape[1]

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Determine which classes to compute loss for
        if self.apply_to_classes is not None:
            class_indices = self.apply_to_classes
        else:
            class_indices = range(num_classes)

        dice_loss = 0.0
        num_classes_used = 0

        for c in class_indices:
            # Get predictions and targets for class c
            pred_c = probs[:, c, :, :]  # (B, H, W)
            target_c = targets_one_hot[:, c, :, :]  # (B, H, W)

            # Flatten spatial dimensions
            pred_c = pred_c.reshape(pred_c.shape[0], -1)  # (B, H*W)
            target_c = target_c.reshape(target_c.shape[0], -1)  # (B, H*W)

            # Compute Dice coefficient
            intersection = (pred_c * target_c).sum(dim=1)  # (B,)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)  # (B,)

            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_coeff).mean()
            num_classes_used += 1

        # Average over classes
        return dice_loss / num_classes_used


class LWFUNetLossTrainer:
    """Training class for LWF-DLR U-Net with Weighted CE + Dice Loss."""

    def __init__(
        self,
        user: str,
        csv_name: str = "full_set_file.csv",
        num_classes: int = 3,
        in_channels: int = 1,
        base_channels: int = 32,
        epochs: int = 32,
        batch_size: int = 12,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        num_workers: int = 16,
        prefetch_factor: int = 8,
        val_split: float = 0.2,
        val_every_n_steps: int = 250,
        val_every_n_steps_warmup: int = 50,
        warmup_steps: int = 100,
        save_every_n_steps: int = 20,
        experiment_group: str = "LWF-DLR",
        experiment_id: str = "weighted_dice_slurm",
        class_weights: list[float] | None = None,
        dice_loss_weight: float = 0.3,
        dice_loss_smooth: float = 1.0,
        dice_loss_classes: list[int] | None = None,
    ):
        """
        Initialize the LWF U-Net trainer with advanced loss functions.

        Parameters
        ----------
        user : str
            Username for HPC paths.
        csv_name : str, optional
            Name of CSV file with NPZ paths (default: "full_set_file.csv").
        num_classes : int, optional
            Number of output classes (default: 3).
        in_channels : int, optional
            Number of input channels (default: 1).
        base_channels : int, optional
            Base number of channels in U-Net (default: 32).
        epochs : int, optional
            Number of training epochs (default: 32).
        batch_size : int, optional
            Batch size for training (default: 12).
        lr : float, optional
            Initial learning rate (default: 0.001).
        weight_decay : float, optional
            Weight decay for optimizer (default: 0.0001).
        num_workers : int, optional
            Number of data loading workers (default: 16).
        prefetch_factor : int, optional
            Number of batches to prefetch (default: 8).
        val_split : float, optional
            Fraction of data for validation (default: 0.2).
        val_every_n_steps : int, optional
            Validate every N steps (default: 250).
        val_every_n_steps_warmup : int, optional
            Validate every N steps during warmup (default: 50).
        warmup_steps : int, optional
            Number of warmup steps (default: 100).
        save_every_n_steps : int, optional
            Save checkpoint every N steps (default: 20).
        experiment_group : str, optional
            Experiment group name (default: "LWF-DLR").
        experiment_id : str, optional
            Unique experiment identifier (default: "weighted_dice_slurm").
        class_weights : list[float] | None, optional
            Class weights for CE loss (default: [1.0, 50.0, 5.0]).
        dice_loss_weight : float, optional
            Weight for Dice loss term (default: 0.3).
        dice_loss_smooth : float, optional
            Smoothing factor for Dice loss (default: 1.0).
        dice_loss_classes : list[int] | None, optional
            Classes to apply Dice loss to (None = all classes).
        """
        self.user = user
        self.csv_name = csv_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.val_split = val_split
        self.val_every_n_steps = val_every_n_steps
        self.val_every_n_steps_warmup = val_every_n_steps_warmup
        self.warmup_steps = warmup_steps
        self.save_every_n_steps = save_every_n_steps
        self.experiment_group = experiment_group
        self.experiment_id = experiment_id

        # Loss configuration
        self.class_weights = class_weights or [1.0, 50.0, 5.0]
        self.dice_loss_weight = dice_loss_weight
        self.dice_loss_smooth = dice_loss_smooth
        self.dice_loss_classes = dice_loss_classes

        self.root_hpc = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026")
        self.user_path = self.root_hpc / user
        self.data_path = self.root_hpc / "data"
        self.data_root = self.data_path / "LWF-DLR"

        self.experiment_dir = self.user_path / f"experiments/{experiment_group}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.experiment_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = None
        self.model = None
        self.writer = None
        self.train_loader = None
        self.val_loader = None
        self.criterion_ce = None
        self.criterion_dice = None
        self.optimizer = None
        self.scheduler = None

    def setup(self) -> None:
        """Set up device, model, and data loaders."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Load dataset
        csv_path = self.data_root / self.csv_name
        dataset_df = pd.read_csv(csv_path)
        npz_names = dataset_df["npz_path"].tolist()

        # Train/val split
        split_idx = int((1 - self.val_split) * len(npz_names))
        train_names = npz_names[:split_idx]
        val_names = npz_names[split_idx:]

        logger.info("Total samples: %d", len(npz_names))
        logger.info("Training samples: %d", len(train_names))
        logger.info("Validation samples: %d", len(val_names))

        # Create datasets and dataloaders
        train_dataset = SegmentationDataset(train_names, self.data_root)
        val_dataset = SegmentationDataset(val_names, self.data_root)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 5,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )

        # Create model
        self.model = UNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
        ).to(self.device)

        logger.info(
            "Model Parameters: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        # Loss functions
        class_weights_tensor = torch.tensor(self.class_weights, device=self.device)
        self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights_tensor)
        self.criterion_dice = DiceLoss(
            smooth=self.dice_loss_smooth, apply_to_classes=self.dice_loss_classes
        )

        logger.info("Loss Configuration:")
        logger.info("  Weighted CE class weights: %s", self.class_weights)
        logger.info("  Dice loss weight (λ_dice): %.3f", self.dice_loss_weight)
        logger.info("  Dice loss smooth: %.1f", self.dice_loss_smooth)
        logger.info("  Dice loss classes: %s", self.dice_loss_classes or "all")

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0.0
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir))

    def validate_model(self) -> dict[str, float]:
        """Run validation and return loss metrics."""
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_loader, desc="Validation", leave=False
            ):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                # Compute individual losses
                ce_loss = self.criterion_ce(outputs, targets)
                dice_loss = self.criterion_dice(outputs, targets)

                # Combined loss
                loss = ce_loss + self.dice_loss_weight * dice_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_dice_loss += dice_loss.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_ce_loss = total_ce_loss / len(self.val_loader)
        avg_dice_loss = total_dice_loss / len(self.val_loader)

        return {
            "loss": avg_loss,
            "ce_loss": avg_ce_loss,
            "dice_loss": avg_dice_loss,
        }

    def save_checkpoint(
        self, epoch: int, global_step: int, val_loss: float, is_best: bool = False
    ) -> Path:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{global_step}.pth"

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
            logger.info("Best model saved! (Val Loss: %.4f)", val_loss)

        return checkpoint_path

    def train(self) -> None:
        """Run the step-based training loop."""
        global_step = 0
        validation_cycle = 0
        best_val_loss = float("inf")

        # Training metrics accumulator (since last validation)
        train_running_loss_since_val = 0.0
        train_running_ce_loss_since_val = 0.0
        train_running_dice_loss_since_val = 0.0
        train_samples_since_val = 0

        logger.info("=" * 80)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info("Epochs: %d", self.epochs)
        logger.info("Loss: Weighted CE + Dice (λ_dice=%.3f)", self.dice_loss_weight)
        logger.info("Class weights: %s", self.class_weights)
        logger.info("Step-based validation:")
        logger.info(
            "  - Warmup: every %d steps for first %d steps",
            self.val_every_n_steps_warmup,
            self.warmup_steps,
        )
        logger.info("  - Normal: every %d steps", self.val_every_n_steps)
        logger.info(
            "Checkpoint saving: every %d steps + best model", self.save_every_n_steps
        )
        logger.info("=" * 80)

        for epoch in range(self.epochs):
            logger.info("\n" + "=" * 60)
            logger.info("Epoch %d/%d", epoch + 1, self.epochs)
            logger.info("=" * 60)

            self.model.train()

            for batch_idx, (inputs, targets) in enumerate(
                tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
            ):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute individual losses
                ce_loss = self.criterion_ce(outputs, targets)
                dice_loss = self.criterion_dice(outputs, targets)

                # Combined loss
                loss = ce_loss + self.dice_loss_weight * dice_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Accumulate training metrics
                batch_size = inputs.size(0)
                train_running_loss_since_val += loss.item() * batch_size
                train_running_ce_loss_since_val += ce_loss.item() * batch_size
                train_running_dice_loss_since_val += dice_loss.item() * batch_size
                train_samples_since_val += batch_size

                # Log learning rate every step
                self.writer.add_scalar(
                    "Train/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )

                global_step += 1

                # STEP-BASED VALIDATION
                # Determine validation frequency (warmup vs normal)
                if global_step <= self.warmup_steps:
                    current_val_freq = self.val_every_n_steps_warmup
                else:
                    current_val_freq = self.val_every_n_steps

                # Check if it's time to validate
                if global_step % current_val_freq == 0:
                    validation_cycle += 1

                    # Log averaged training loss since last validation
                    if train_samples_since_val > 0:
                        train_loss_avg = (
                            train_running_loss_since_val / train_samples_since_val
                        )
                        train_ce_loss_avg = (
                            train_running_ce_loss_since_val / train_samples_since_val
                        )
                        train_dice_loss_avg = (
                            train_running_dice_loss_since_val / train_samples_since_val
                        )

                        self.writer.add_scalar(
                            "Train/total_loss_interval", train_loss_avg, global_step
                        )
                        self.writer.add_scalar(
                            "Train/ce_loss_interval", train_ce_loss_avg, global_step
                        )
                        self.writer.add_scalar(
                            "Train/dice_loss_interval",
                            train_dice_loss_avg,
                            global_step,
                        )

                    # Reset training accumulators
                    train_running_loss_since_val = 0.0
                    train_running_ce_loss_since_val = 0.0
                    train_running_dice_loss_since_val = 0.0
                    train_samples_since_val = 0

                    # Run validation
                    val_metrics = self.validate_model()

                    # TensorBoard Logging
                    self.writer.add_scalar(
                        "Val/total_loss", val_metrics["loss"], global_step
                    )
                    self.writer.add_scalar(
                        "Val/ce_loss", val_metrics["ce_loss"], global_step
                    )
                    self.writer.add_scalar(
                        "Val/dice_loss", val_metrics["dice_loss"], global_step
                    )

                    # Console output
                    logger.info(
                        "\n[Step %d | Cycle %d | Epoch %d]",
                        global_step,
                        validation_cycle,
                        epoch + 1,
                    )
                    logger.info("  Val Loss: %.4f", val_metrics["loss"])
                    logger.info("    ├─ CE Loss:   %.4f", val_metrics["ce_loss"])
                    logger.info("    └─ Dice Loss: %.4f", val_metrics["dice_loss"])

                    # Checkpoint Saving
                    is_best = val_metrics["loss"] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics["loss"]

                    # Save checkpoint every N steps or if best
                    if global_step % self.save_every_n_steps == 0 or is_best:
                        self.save_checkpoint(
                            epoch, global_step, val_metrics["loss"], is_best=is_best
                        )
                        if global_step % self.save_every_n_steps == 0:
                            logger.info("Checkpoint saved at step %d", global_step)

                    # Return to training mode
                    self.model.train()

            # Update learning rate at end of epoch
            self.scheduler.step()
            logger.info(
                "\nEpoch %d complete. LR: %.6f",
                epoch + 1,
                self.optimizer.param_groups[0]["lr"],
            )

        # Final Save
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Total steps: %d", global_step)
        logger.info("Validation cycles: %d", validation_cycle)
        logger.info("Best Val Loss: %.4f", best_val_loss)

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pth"
        torch.save(
            {
                "epoch": self.epochs,
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            final_path,
        )
        logger.info("Final model saved to: %s", final_path)
        logger.info("Best model saved to: %s", self.checkpoint_dir / "best_model.pth")
        logger.info("TensorBoard logs: %s", self.experiment_dir)
        logger.info("To view logs, run: tensorboard --logdir=%s", self.experiment_dir)
        logger.info("=" * 60)

        # Close TensorBoard writer
        self.writer.close()

    def __call__(self) -> str:
        """
        Run the main training loop (called by submitit).

        Returns
        -------
        str
            Path to the saved final model.
        """
        logger.info("Starting training: %s", self.experiment_id)
        self.setup()
        self.train()
        return str(self.checkpoint_dir / "final_model.pth")
