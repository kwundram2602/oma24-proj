"""LWF-DLR U-Net with optional skeleton enhancements (distance, skeleton input, dual-head)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class SkeletonSegmentationDataset(Dataset):
    """Dataset with optional skeleton features for multi-task learning."""

    def __init__(
        self,
        npz_names,
        base_path,
        use_distance=False,
        distance_max=128,
        use_input_skeleton=False,
        skeleton_class=1,
    ):
        """
        Initialize the skeleton-enhanced segmentation dataset.

        Parameters
        ----------
        npz_names : list[str]
            List of NPZ file names to load.
        base_path : Path
            Base path containing the npz directory.
        use_distance : bool
            Whether to include distance transform channel (default: False).
        distance_max : int
            Maximum distance for normalization (default: 128).
        use_input_skeleton : bool
            Whether to include skeleton as input channel (default: False).
        skeleton_class : int
            Class index to extract skeleton from (1 = linear_vegetation).
        """
        self.npz_names = npz_names
        self.base_path = Path(base_path)
        self.npz_dir = self.base_path / "npz"
        self.use_distance = use_distance
        self.distance_max = distance_max
        self.use_input_skeleton = use_input_skeleton
        self.skeleton_class = skeleton_class

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

    def _binarize_mask(self, label_mask: np.ndarray) -> np.ndarray:
        """Convert multi-class mask to binary (0=background, 1=vegetation)."""
        return (label_mask > 0).astype(np.uint8)

    def _get_distance_transform(self, binary_mask: np.ndarray) -> np.ndarray:
        """Compute normalized distance transform from vegetation boundaries."""
        distance = distance_transform_edt(binary_mask)
        distance = np.clip(distance, 0, self.distance_max) / self.distance_max
        return distance.astype(np.float32)

    def _skeletonize_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extract skeleton (medial axis) from binary mask."""
        skeleton = skeletonize(binary_mask.astype(bool))
        return skeleton.astype(np.uint8)

    def _extract_class_skeleton(self, label_mask: np.ndarray) -> np.ndarray:
        """Extract skeleton for a specific class."""
        class_mask = (label_mask == self.skeleton_class).astype(np.uint8)
        return self._skeletonize_mask(class_mask)

    def __getitem__(self, idx):
        # Load data
        npz_path = self.npz_dir / self.npz_names[idx]
        data = np.load(npz_path)
        label_mask = data["label"]  # Shape: (H, W)

        # Remap labels
        remapped_mask = np.zeros_like(label_mask, dtype=np.int64)
        for old_label, new_label in self.remap_dict.items():
            remapped_mask[label_mask == old_label] = new_label

        # Create binary mask (base channel)
        binary_mask = self._binarize_mask(remapped_mask).astype(np.float32)

        # Build input channels
        channels = [binary_mask[..., np.newaxis]]  # Base channel

        if self.use_distance:
            distance = self._get_distance_transform(binary_mask)
            channels.append(distance[..., np.newaxis])

        if self.use_input_skeleton:
            skeleton = self._skeletonize_mask(binary_mask).astype(np.float32)
            channels.append(skeleton[..., np.newaxis])

        # Combine channels: (H, W, C) -> (C, H, W)
        x = np.concatenate(channels, axis=-1)
        x = np.transpose(x, (2, 0, 1))

        # Segmentation target
        y_seg = remapped_mask.astype(np.int64)

        # Skeleton target (for linear vegetation only)
        y_skel = self._extract_class_skeleton(remapped_mask)
        y_skel = y_skel.astype(np.float32)

        # Convert to tensors
        x = torch.from_numpy(x).float()  # (C, H, W)
        y_seg = torch.from_numpy(y_seg).long()  # (H, W)
        y_skel = torch.from_numpy(y_skel).float()  # (H, W)

        return x, y_seg, y_skel


class DoubleConv(nn.Module):
    """Two consecutive convolution blocks with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.residual_conv is not None:
                x = self.residual_conv(x)
            out = out + x
        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels, out_channels, rates=(3, 6, 9, 12)):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        total_channels = (1 + len(rates) + 1) * out_channels
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        features = [self.conv1x1(x)]
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))
        global_feat = self.global_pool(x)
        global_feat = nn.functional.interpolate(
            global_feat, size=(h, w), mode="bilinear", align_corners=False
        )
        features.append(global_feat)
        out = torch.cat(features, dim=1)
        return self.project(out)


class DualHeadUNet(nn.Module):
    """U-Net with optional dual heads for segmentation and skeleton prediction.

    Args:
        in_channels: Number of input channels (1-3 depending on features)
        num_classes: Number of segmentation classes (3)
        base_channels: Base number of channels (default: 32)
        use_residual: Whether to use residual connections (default: True)
        use_aspp: Whether to use ASPP in bottleneck (default: True)
        aspp_rates: Dilation rates for ASPP (default: [3, 6, 9, 12])
        use_dual_head: Whether to use dual-head architecture (default: False)
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_channels=32,
        use_residual=True,
        use_aspp=True,
        aspp_rates=(3, 6, 9, 12),
        use_dual_head=False,
    ):
        super().__init__()
        self.use_aspp = use_aspp
        self.use_dual_head = use_dual_head

        # SHARED ENCODER
        self.enc1 = DoubleConv(in_channels, base_channels, use_residual)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, use_residual)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, use_residual)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, use_residual)
        self.pool = nn.MaxPool2d(2)

        # SHARED BOTTLENECK
        if use_aspp:
            self.bottleneck = ASPP(
                base_channels * 8, base_channels * 16, rates=aspp_rates
            )
        else:
            self.bottleneck = DoubleConv(
                base_channels * 8, base_channels * 16, use_residual
            )

        # SEGMENTATION DECODER (always present)
        self.seg_up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.seg_dec4 = DoubleConv(base_channels * 16, base_channels * 8, use_residual)

        self.seg_up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.seg_dec3 = DoubleConv(base_channels * 8, base_channels * 4, use_residual)

        self.seg_up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.seg_dec2 = DoubleConv(base_channels * 4, base_channels * 2, use_residual)

        self.seg_up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.seg_dec1 = DoubleConv(base_channels * 2, base_channels, use_residual)

        self.seg_out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # SKELETON DECODER (optional - only if dual-head enabled)
        if use_dual_head:
            self.skel_up4 = nn.ConvTranspose2d(
                base_channels * 16, base_channels * 8, kernel_size=2, stride=2
            )
            self.skel_dec4 = DoubleConv(
                base_channels * 16, base_channels * 8, use_residual
            )

            self.skel_up3 = nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4, kernel_size=2, stride=2
            )
            self.skel_dec3 = DoubleConv(
                base_channels * 8, base_channels * 4, use_residual
            )

            self.skel_up2 = nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, kernel_size=2, stride=2
            )
            self.skel_dec2 = DoubleConv(
                base_channels * 4, base_channels * 2, use_residual
            )

            self.skel_up1 = nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=2, stride=2
            )
            self.skel_dec1 = DoubleConv(base_channels * 2, base_channels, use_residual)

            self.skel_out = nn.Sequential(
                nn.Conv2d(base_channels, 1, kernel_size=1), nn.Sigmoid()
            )

    def forward(self, x):
        # Shared encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Shared bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Segmentation decoder
        seg_dec4 = self.seg_up4(bottleneck)
        seg_dec4 = torch.cat([seg_dec4, enc4], dim=1)
        seg_dec4 = self.seg_dec4(seg_dec4)

        seg_dec3 = self.seg_up3(seg_dec4)
        seg_dec3 = torch.cat([seg_dec3, enc3], dim=1)
        seg_dec3 = self.seg_dec3(seg_dec3)

        seg_dec2 = self.seg_up2(seg_dec3)
        seg_dec2 = torch.cat([seg_dec2, enc2], dim=1)
        seg_dec2 = self.seg_dec2(seg_dec2)

        seg_dec1 = self.seg_up1(seg_dec2)
        seg_dec1 = torch.cat([seg_dec1, enc1], dim=1)
        seg_dec1 = self.seg_dec1(seg_dec1)

        seg_out = self.seg_out(seg_dec1)

        # Skeleton decoder (if dual-head)
        if self.use_dual_head:
            skel_dec4 = self.skel_up4(bottleneck)
            skel_dec4 = torch.cat([skel_dec4, enc4], dim=1)
            skel_dec4 = self.skel_dec4(skel_dec4)

            skel_dec3 = self.skel_up3(skel_dec4)
            skel_dec3 = torch.cat([skel_dec3, enc3], dim=1)
            skel_dec3 = self.skel_dec3(skel_dec3)

            skel_dec2 = self.skel_up2(skel_dec3)
            skel_dec2 = torch.cat([skel_dec2, enc2], dim=1)
            skel_dec2 = self.skel_dec2(skel_dec2)

            skel_dec1 = self.skel_up1(skel_dec2)
            skel_dec1 = torch.cat([skel_dec1, enc1], dim=1)
            skel_dec1 = self.skel_dec1(skel_dec1)

            skel_out = self.skel_out(skel_dec1)

            return seg_out, skel_out
        else:
            return seg_out


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""

    def __init__(self, smooth=1.0, apply_to_classes=None):
        super().__init__()
        self.smooth = smooth
        self.apply_to_classes = apply_to_classes

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        if self.apply_to_classes is not None:
            class_indices = self.apply_to_classes
        else:
            class_indices = range(num_classes)

        dice_loss = 0.0
        num_classes_used = 0

        for c in class_indices:
            pred_c = probs[:, c, :, :].reshape(probs.shape[0], -1)
            target_c = targets_one_hot[:, c, :, :].reshape(targets_one_hot.shape[0], -1)

            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)

            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_coeff).mean()
            num_classes_used += 1

        return dice_loss / num_classes_used


class LWFUNetSkeletonTrainer:
    """Training class for LWF-DLR U-Net with optional skeleton enhancements."""

    def __init__(
        self,
        user: str,
        csv_name: str = "full_set_file.csv",
        num_classes: int = 3,
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
        experiment_id: str = "skeleton_unet_slurm",
        # Loss configuration
        use_class_weights: bool = True,
        class_weights: list[float] | None = None,
        use_dice_loss: bool = True,
        dice_loss_weight: float = 0.3,
        dice_loss_smooth: float = 1.0,
        dice_loss_classes: list[int] | None = None,
        # Architecture configuration
        use_residual: bool = True,
        use_aspp: bool = True,
        aspp_rates: tuple[int, ...] = (3, 6, 9, 12),
        # Skeleton enhancement configuration
        use_distance: bool = False,
        distance_max: int = 128,
        use_input_skeleton: bool = False,
        use_dual_head: bool = False,
        skeleton_class: int = 1,
        skeleton_loss_weight: float = 1.0,
    ):
        """
        Initialize the LWF U-Net trainer with optional skeleton enhancements.

        Parameters
        ----------
        user : str
            Username for HPC paths.
        csv_name : str, optional
            Name of CSV file with NPZ paths (default: "full_set_file.csv").
        num_classes : int, optional
            Number of output classes (default: 3).
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
            Unique experiment identifier (default: "skeleton_unet_slurm").
        use_class_weights : bool, optional
            Whether to use class weights for CE loss (default: True).
        class_weights : list[float] | None, optional
            Class weights for CE loss (default: [1.0, 50.0, 5.0]).
        use_dice_loss : bool, optional
            Whether to use Dice loss (default: True).
        dice_loss_weight : float, optional
            Weight for Dice loss term (default: 0.3).
        dice_loss_smooth : float, optional
            Smoothing factor for Dice loss (default: 1.0).
        dice_loss_classes : list[int] | None, optional
            Classes to apply Dice loss to (None = all classes).
        use_residual : bool, optional
            Whether to use residual connections (default: True).
        use_aspp : bool, optional
            Whether to use ASPP in bottleneck (default: True).
        aspp_rates : tuple[int, ...], optional
            Dilation rates for ASPP (default: (3, 6, 9, 12)).
        use_distance : bool, optional
            Whether to include distance transform as input (default: False).
        distance_max : int, optional
            Maximum distance for normalization (default: 128).
        use_input_skeleton : bool, optional
            Whether to include skeleton as input (default: False).
        use_dual_head : bool, optional
            Whether to use dual-head architecture for skeleton prediction (default: False).
        skeleton_class : int, optional
            Class index to extract skeleton from (default: 1 = linear_vegetation).
        skeleton_loss_weight : float, optional
            Weight for skeleton BCE loss (default: 1.0).
        """
        self.user = user
        self.csv_name = csv_name
        self.num_classes = num_classes
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
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights or [1.0, 50.0, 5.0]
        self.use_dice_loss = use_dice_loss
        self.dice_loss_weight = dice_loss_weight
        self.dice_loss_smooth = dice_loss_smooth
        self.dice_loss_classes = dice_loss_classes

        # Architecture configuration
        self.use_residual = use_residual
        self.use_aspp = use_aspp
        self.aspp_rates = aspp_rates

        # Skeleton enhancement configuration
        self.use_distance = use_distance
        self.distance_max = distance_max
        self.use_input_skeleton = use_input_skeleton
        self.use_dual_head = use_dual_head
        self.skeleton_class = skeleton_class
        self.skeleton_loss_weight = skeleton_loss_weight

        # Compute number of input channels based on features
        self.in_channels = 1  # Base binary mask
        if self.use_distance:
            self.in_channels += 1
        if self.use_input_skeleton:
            self.in_channels += 1

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
        self.criterion_skeleton = None
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

        # Create datasets with skeleton features
        train_dataset = SkeletonSegmentationDataset(
            train_names,
            self.data_root,
            use_distance=self.use_distance,
            distance_max=self.distance_max,
            use_input_skeleton=self.use_input_skeleton,
            skeleton_class=self.skeleton_class,
        )
        val_dataset = SkeletonSegmentationDataset(
            val_names,
            self.data_root,
            use_distance=self.use_distance,
            distance_max=self.distance_max,
            use_input_skeleton=self.use_input_skeleton,
            skeleton_class=self.skeleton_class,
        )

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

        # Create model with flexible architecture
        self.model = DualHeadUNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            use_residual=self.use_residual,
            use_aspp=self.use_aspp,
            aspp_rates=self.aspp_rates,
            use_dual_head=self.use_dual_head,
        ).to(self.device)

        logger.info(
            "Model Parameters: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        # Loss functions
        if self.use_class_weights:
            class_weights_tensor = torch.tensor(self.class_weights, device=self.device)
            self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.criterion_ce = nn.CrossEntropyLoss()

        if self.use_dice_loss:
            self.criterion_dice = DiceLoss(
                smooth=self.dice_loss_smooth, apply_to_classes=self.dice_loss_classes
            )
        else:
            self.criterion_dice = None

        if self.use_dual_head:
            self.criterion_skeleton = nn.BCELoss()
        else:
            self.criterion_skeleton = None

        # Log configuration
        logger.info("=" * 80)
        logger.info("CONFIGURATION")
        logger.info("=" * 80)

        # Input features
        logger.info("Input Features (in_channels=%d):", self.in_channels)
        logger.info("  - Binary mask: Yes")
        if self.use_distance:
            logger.info("  - Distance transform: Yes (max=%d)", self.distance_max)
        else:
            logger.info("  - Distance transform: No")
        if self.use_input_skeleton:
            logger.info("  - Input skeleton: Yes (class=%d)", self.skeleton_class)
        else:
            logger.info("  - Input skeleton: No")

        # Architecture
        logger.info("Architecture Configuration:")
        arch_components = ["U-Net"]
        if self.use_residual:
            arch_components.append("Residual")
        if self.use_aspp:
            arch_components.append("ASPP")
        if self.use_dual_head:
            arch_components.append("Dual-Head")
        logger.info("  Components: %s", " + ".join(arch_components))
        if self.use_aspp:
            logger.info("  ASPP rates: %s", self.aspp_rates)

        # Loss configuration
        logger.info("Loss Configuration:")
        loss_components = []
        if self.use_class_weights:
            loss_components.append(f"Weighted CE {self.class_weights}")
        else:
            loss_components.append("Vanilla CE")
        if self.use_dice_loss:
            loss_components.append(f"Dice (λ={self.dice_loss_weight})")
        if self.use_dual_head:
            loss_components.append(f"Skeleton BCE (λ={self.skeleton_loss_weight})")
        logger.info("  Loss: %s", " + ".join(loss_components))

        logger.info("=" * 80)

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
        total_skeleton_loss = 0

        with torch.no_grad():
            for inputs, targets_seg, targets_skel in tqdm(
                self.val_loader, desc="Validation", leave=False
            ):
                inputs = inputs.to(self.device)
                targets_seg = targets_seg.to(self.device)
                targets_skel = targets_skel.to(self.device)

                # Forward pass
                if self.use_dual_head:
                    outputs_seg, outputs_skel = self.model(inputs)
                else:
                    outputs_seg = self.model(inputs)

                # Compute CE loss
                ce_loss = self.criterion_ce(outputs_seg, targets_seg)
                loss = ce_loss

                # Add Dice loss if enabled
                if self.use_dice_loss:
                    dice_loss = self.criterion_dice(outputs_seg, targets_seg)
                    loss = loss + self.dice_loss_weight * dice_loss
                    total_dice_loss += dice_loss.item()

                # Add skeleton loss if dual-head enabled
                if self.use_dual_head:
                    skeleton_loss = self.criterion_skeleton(
                        outputs_skel.squeeze(1), targets_skel
                    )
                    loss = loss + self.skeleton_loss_weight * skeleton_loss
                    total_skeleton_loss += skeleton_loss.item()

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_ce_loss = total_ce_loss / len(self.val_loader)

        result = {
            "loss": avg_loss,
            "ce_loss": avg_ce_loss,
        }

        if self.use_dice_loss:
            avg_dice_loss = total_dice_loss / len(self.val_loader)
            result["dice_loss"] = avg_dice_loss

        if self.use_dual_head:
            avg_skeleton_loss = total_skeleton_loss / len(self.val_loader)
            result["skeleton_loss"] = avg_skeleton_loss

        return result

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
        train_running_skeleton_loss_since_val = 0.0
        train_samples_since_val = 0

        logger.info("=" * 80)
        logger.info("TRAINING START")
        logger.info("=" * 80)
        logger.info("Epochs: %d", self.epochs)
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

            for batch_idx, (inputs, targets_seg, targets_skel) in enumerate(
                tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
            ):
                inputs = inputs.to(self.device)
                targets_seg = targets_seg.to(self.device)
                targets_skel = targets_skel.to(self.device)

                # Forward pass
                if self.use_dual_head:
                    outputs_seg, outputs_skel = self.model(inputs)
                else:
                    outputs_seg = self.model(inputs)

                # Compute CE loss
                ce_loss = self.criterion_ce(outputs_seg, targets_seg)
                loss = ce_loss

                # Add Dice loss if enabled
                dice_loss = None
                if self.use_dice_loss:
                    dice_loss = self.criterion_dice(outputs_seg, targets_seg)
                    loss = loss + self.dice_loss_weight * dice_loss

                # Add skeleton loss if dual-head enabled
                skeleton_loss = None
                if self.use_dual_head:
                    skeleton_loss = self.criterion_skeleton(
                        outputs_skel.squeeze(1), targets_skel
                    )
                    loss = loss + self.skeleton_loss_weight * skeleton_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Accumulate training metrics
                batch_size = inputs.size(0)
                train_running_loss_since_val += loss.item() * batch_size
                train_running_ce_loss_since_val += ce_loss.item() * batch_size
                if self.use_dice_loss:
                    train_running_dice_loss_since_val += dice_loss.item() * batch_size
                if self.use_dual_head:
                    train_running_skeleton_loss_since_val += (
                        skeleton_loss.item() * batch_size
                    )
                train_samples_since_val += batch_size

                # Log learning rate every step
                self.writer.add_scalar(
                    "Train/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )

                global_step += 1

                # STEP-BASED VALIDATION
                if global_step <= self.warmup_steps:
                    current_val_freq = self.val_every_n_steps_warmup
                else:
                    current_val_freq = self.val_every_n_steps

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

                        self.writer.add_scalar(
                            "Train/total_loss_interval", train_loss_avg, global_step
                        )
                        self.writer.add_scalar(
                            "Train/ce_loss_interval", train_ce_loss_avg, global_step
                        )

                        if self.use_dice_loss:
                            train_dice_loss_avg = (
                                train_running_dice_loss_since_val
                                / train_samples_since_val
                            )
                            self.writer.add_scalar(
                                "Train/dice_loss_interval",
                                train_dice_loss_avg,
                                global_step,
                            )

                        if self.use_dual_head:
                            train_skeleton_loss_avg = (
                                train_running_skeleton_loss_since_val
                                / train_samples_since_val
                            )
                            self.writer.add_scalar(
                                "Train/skeleton_loss_interval",
                                train_skeleton_loss_avg,
                                global_step,
                            )

                    # Reset training accumulators
                    train_running_loss_since_val = 0.0
                    train_running_ce_loss_since_val = 0.0
                    train_running_dice_loss_since_val = 0.0
                    train_running_skeleton_loss_since_val = 0.0
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
                    if "dice_loss" in val_metrics:
                        self.writer.add_scalar(
                            "Val/dice_loss", val_metrics["dice_loss"], global_step
                        )
                    if "skeleton_loss" in val_metrics:
                        self.writer.add_scalar(
                            "Val/skeleton_loss",
                            val_metrics["skeleton_loss"],
                            global_step,
                        )

                    # Console output
                    logger.info(
                        "\n[Step %d | Cycle %d | Epoch %d]",
                        global_step,
                        validation_cycle,
                        epoch + 1,
                    )
                    logger.info("  Val Loss: %.4f", val_metrics["loss"])
                    logger.info("    ├─ CE Loss:       %.4f", val_metrics["ce_loss"])
                    if "dice_loss" in val_metrics:
                        logger.info(
                            "    ├─ Dice Loss:     %.4f", val_metrics["dice_loss"]
                        )
                    if "skeleton_loss" in val_metrics:
                        logger.info(
                            "    └─ Skeleton Loss: %.4f", val_metrics["skeleton_loss"]
                        )

                    # Checkpoint Saving
                    is_best = val_metrics["loss"] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics["loss"]

                    if global_step % self.save_every_n_steps == 0 or is_best:
                        self.save_checkpoint(
                            epoch,
                            global_step,
                            val_metrics["loss"],
                            is_best=is_best,
                        )
                        if global_step % self.save_every_n_steps == 0:
                            logger.info("  → Checkpoint saved at step %d", global_step)

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
