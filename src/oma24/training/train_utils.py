"""Shared utilities for LWF-DLR segmentation training scripts."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required on HPC clusters (no display)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Default class names used across all segmentation trainers
CLASS_NAMES = {0: "background", 1: "linear_veg", 2: "patchy_veg"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """Load NPZ segmentation patches and return (input_tensor, label_tensor).

    Parameters
    ----------
    npz_names : list[str]
        File names of the NPZ patches.
    base_path : Path
        Root directory that contains the ``npz/`` subfolder.
    repeat_channels : int
        Number of input channels.  Use 1 for single-channel models (UNet).
        Use 3 to repeat the binary mask to three channels for backbone models
        that expect RGB-like input (FarSeg, DeepLabV3, …).
    """

    remap_dict = {
        0: 0,  # background
        1: 1,  # linear_vegetation
        2: 2,  # patchy_vegetation
        3: 2,
        4: 2,
        5: 2,
    }

    def __init__(self, npz_names: list, base_path: Path, repeat_channels: int = 1):
        self.npz_names = npz_names
        self.npz_dir = Path(base_path) / "npz"
        self.repeat_channels = repeat_channels

    def __len__(self) -> int:
        return len(self.npz_names)

    def __getitem__(self, idx: int):
        data = np.load(self.npz_dir / self.npz_names[idx])
        label_mask = data["label"]  # (H, W)

        remapped = np.zeros_like(label_mask, dtype=np.int64)
        for old, new in self.remap_dict.items():
            remapped[label_mask == old] = new

        binary = (remapped > 0).astype(np.float32)
        x = torch.from_numpy(binary).unsqueeze(0)  # (1, H, W)
        if self.repeat_channels > 1:
            x = x.repeat(self.repeat_channels, 1, 1)  # (C, H, W)
        y = torch.from_numpy(remapped).long()  # (H, W)
        return x, y


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_train_val_test_loaders(
    npz_names: list,
    data_root: Path,
    val_split: float,
    test_split: float,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    repeat_channels: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Split *npz_names* into train/val/test and return DataLoaders.

    The split is deterministic and order-based:
    - train : first ``(1 - val_split - test_split)`` fraction
    - val   : next  ``val_split`` fraction
    - test  : last  ``test_split`` fraction  (None when *test_split* == 0)

    Returns
    -------
    train_loader, val_loader, test_loader  (test_loader is None if test_split=0)
    """
    n = len(npz_names)
    train_end = int((1 - val_split - test_split) * n)
    val_end = int((1 - test_split) * n)

    train_names = npz_names[:train_end]
    val_names = npz_names[train_end:val_end]
    test_names = npz_names[val_end:] if test_split > 0.0 else None

    logger.info("Dataset split — train: %d | val: %d | test: %s",
                len(train_names), len(val_names),
                len(test_names) if test_names else "—")

    _loader_kwargs = dict(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )

    train_loader = DataLoader(
        SegmentationDataset(train_names, data_root, repeat_channels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_names, data_root, repeat_channels),
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs,
    )
    test_loader = None
    if test_names is not None:
        test_loader = DataLoader(
            SegmentationDataset(test_names, data_root, repeat_channels),
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs,
        )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    is_best: bool = False,
) -> Path:
    """Save a training checkpoint; also writes *best_model.pth* when *is_best*."""
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pth"
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(state, best_path)
        logger.info("Best model saved! (Val Loss: %.4f)", val_loss)

    return checkpoint_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: dict | None = None,
    prefix: str = "val",
    output_key: str | None = None,
) -> dict:
    """Evaluate *model* on *loader*; return loss + per-class IoU/Precision/Recall.

    Parameters
    ----------
    output_key : str or None
        When the model returns a dict (e.g. ``{"out": logits}`` for DeepLabV3),
        set *output_key* to the key to extract the logit tensor.
        Leave as *None* for models that return a plain tensor.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    model.eval()
    total_loss = 0.0
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average=None).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average=None).to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"{prefix.capitalize()} Evaluation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if output_key is not None:
                outputs = outputs[output_key]

            total_loss += criterion(outputs, targets).item()
            preds = outputs.argmax(dim=1)
            iou_metric.update(preds, targets)
            precision_metric.update(preds, targets)
            recall_metric.update(preds, targets)

    iou_per_class = iou_metric.compute()
    precision_per_class = precision_metric.compute()
    recall_per_class = recall_metric.compute()

    metrics = {
        f"{prefix}_loss": total_loss / len(loader),
        "mean_iou": iou_per_class.mean().item(),
    }
    for c in range(num_classes):
        name = class_names.get(c, str(c))
        metrics[f"iou/{name}"] = iou_per_class[c].item()
        metrics[f"precision/{name}"] = precision_per_class[c].item()
        metrics[f"recall/{name}"] = recall_per_class[c].item()

    return metrics


def compute_map_iou_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    foreground_classes: tuple = (1, 2),
    output_key: str | None = None,
) -> dict:
    """Compute mAP@[0.50:0.95] using per-image binary-mask IoU.

    For each threshold τ ∈ {0.50, 0.55, …, 0.95}:
      - per image, per foreground class *c*: IoU(pred_c_mask, gt_c_mask)
      - AP_c(τ) = fraction of images where IoU_c ≥ τ
      - mAP(τ)  = mean over foreground classes
    overall_map = mean over all τ.

    Corner case: union == 0 (both masks empty) → IoU = 1.0.
    """
    thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()
    image_ious: dict[int, list] = {c: [] for c in foreground_classes}

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="mAP Evaluation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if output_key is not None:
                outputs = outputs[output_key]
            preds = outputs.argmax(dim=1)  # (N, H, W)

            for i in range(preds.shape[0]):
                for c in foreground_classes:
                    pred_c = preds[i] == c
                    gt_c = targets[i] == c
                    intersection = (pred_c & gt_c).sum().item()
                    union = (pred_c | gt_c).sum().item()
                    image_ious[c].append(intersection / union if union > 0 else 1.0)

    for c in foreground_classes:
        image_ious[c] = np.array(image_ious[c])

    ap_per_class_per_threshold: dict[int, list] = {c: [] for c in foreground_classes}
    map_per_threshold = []
    for tau in thresholds:
        per_class_ap = []
        for c in foreground_classes:
            ap_c = float((image_ious[c] >= tau).mean())
            ap_per_class_per_threshold[c].append(ap_c)
            per_class_ap.append(ap_c)
        map_per_threshold.append(float(np.mean(per_class_ap)))

    return {
        "thresholds": thresholds,
        "map_per_threshold": map_per_threshold,
        "overall_map": float(np.mean(map_per_threshold)),
        "ap_per_class_per_threshold": ap_per_class_per_threshold,
    }


def compute_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    output_key: str | None = None,
) -> np.ndarray:
    """Return a (num_classes × num_classes) confusion matrix as a numpy array.

    Rows = true class, columns = predicted class.
    """
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Confusion Matrix", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if output_key is not None:
                outputs = outputs[output_key]
            preds = outputs.argmax(dim=1)
            cm_metric.update(preds, targets)
    return cm_metric.compute().cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_conf_matrix(ax, cm: np.ndarray, class_names: list, title: str) -> None:
    """Draw a row-normalised confusion matrix with raw counts as annotations."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
    ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{int(cm[i, j]):,}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", color=color, fontsize=8)


def generate_plots(
    writer,
    plots_dir: Path,
    val_metrics: dict,
    test_metrics: dict | None,
    val_map_data: dict,
    test_map_data: dict | None,
    global_step: int,
    class_names: list | None = None,
    model_name: str = "",
    train_loss_history: list | None = None,
    val_loss_history: list | None = None,
    val_conf_matrix: np.ndarray | None = None,
    test_conf_matrix: np.ndarray | None = None,
) -> None:
    """Save IoU, Precision/Recall, mAP-threshold, loss curve, and confusion matrix plots.

    Parameters
    ----------
    writer : SummaryWriter
        Active TensorBoard writer.
    plots_dir : Path
        Directory where PNG files are written.
    model_name : str
        Included in every plot title (e.g. ``"FarSeg-resnet50"``).
    train_loss_history : list of (step, loss) tuples or None
    val_loss_history : list of (step, loss) tuples or None
    val_conf_matrix / test_conf_matrix : np.ndarray (num_classes × num_classes) or None
    class_names : list[str] or None
        Defaults to ``["background", "linear_veg", "patchy_veg"]``.
    """
    if class_names is None:
        class_names = ["background", "linear_veg", "patchy_veg"]

    prefix = f"{model_name} — " if model_name else ""
    x = np.arange(len(class_names))
    width = 0.35

    # ── Plot 1: IoU per class ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    val_ious = [val_metrics[f"iou/{c}"] for c in class_names]
    ax.bar(x - width / 2, val_ious, width, label="Val", color="steelblue")
    if test_metrics is not None:
        test_ious = [test_metrics[f"iou/{c}"] for c in class_names]
        ax.bar(x + width / 2, test_ious, width, label="Test", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("IoU")
    ax.set_title(f"{prefix}IoU per Class — Val vs Test")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    iou_path = plots_dir / "iou_per_class.png"
    fig.savefig(iou_path, dpi=150)
    writer.add_figure("Plots/iou_per_class", fig, global_step)
    plt.close(fig)
    logger.info("Saved IoU plot: %s", iou_path)

    # ── Plot 2: Precision & Recall per class ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric_key, title in [
        (ax1, "precision", f"{prefix}Precision per Class"),
        (ax2, "recall", f"{prefix}Recall per Class"),
    ]:
        val_vals = [val_metrics[f"{metric_key}/{c}"] for c in class_names]
        ax.bar(x - width / 2, val_vals, width, label="Val", color="steelblue")
        if test_metrics is not None:
            test_vals = [test_metrics[f"{metric_key}/{c}"] for c in class_names]
            ax.bar(x + width / 2, test_vals, width, label="Test", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel(metric_key.capitalize())
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 1)
    fig.tight_layout()
    pr_path = plots_dir / "precision_recall_per_class.png"
    fig.savefig(pr_path, dpi=150)
    writer.add_figure("Plots/precision_recall_per_class", fig, global_step)
    plt.close(fig)
    logger.info("Saved Precision/Recall plot: %s", pr_path)

    # ── Plot 3: mAP vs IoU threshold ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    thresholds = val_map_data["thresholds"]
    ax.plot(
        thresholds, val_map_data["map_per_threshold"],
        marker="o", label=f"Val mAP={val_map_data['overall_map']:.3f}", color="steelblue",
    )
    if test_map_data is not None:
        ax.plot(
            thresholds, test_map_data["map_per_threshold"],
            marker="s", label=f"Test mAP={test_map_data['overall_map']:.3f}", color="darkorange",
        )
    ax.set_xlabel("IoU Threshold τ")
    ax.set_ylabel("mAP")
    ax.set_title(f"{prefix}mAP at IoU Thresholds — Val vs Test")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    fig.tight_layout()
    map_path = plots_dir / "map_iou_thresholds.png"
    fig.savefig(map_path, dpi=150)
    writer.add_figure("Plots/map_iou_thresholds", fig, global_step)
    plt.close(fig)
    logger.info("Saved mAP plot: %s", map_path)

    # ── Plot 4: Loss curves ───────────────────────────────────────────────────
    if train_loss_history or val_loss_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        if train_loss_history:
            steps, losses = zip(*train_loss_history)
            ax.plot(steps, losses, label="Train Loss", color="steelblue")
        if val_loss_history:
            steps, losses = zip(*val_loss_history)
            ax.plot(steps, losses, label="Val Loss", color="darkorange", marker="o", markersize=4)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{prefix}Training & Validation Loss")
        ax.legend()
        fig.tight_layout()
        loss_path = plots_dir / "loss_curve.png"
        fig.savefig(loss_path, dpi=150)
        writer.add_figure("Plots/loss_curve", fig, global_step)
        plt.close(fig)
        logger.info("Saved loss plot: %s", loss_path)

    # ── Plot 5: Confusion matrix ──────────────────────────────────────────────
    if val_conf_matrix is not None or test_conf_matrix is not None:
        matrices = [(val_conf_matrix, "Val"), (test_conf_matrix, "Test")]
        matrices = [(cm, split) for cm, split in matrices if cm is not None]
        fig, axes = plt.subplots(1, len(matrices), figsize=(7 * len(matrices), 6))
        if len(matrices) == 1:
            axes = [axes]
        for ax, (cm, split) in zip(axes, matrices):
            _plot_conf_matrix(ax, cm, class_names, f"{prefix}Confusion Matrix — {split}")
        fig.tight_layout()
        cm_path = plots_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        writer.add_figure("Plots/confusion_matrix", fig, global_step)
        plt.close(fig)
        logger.info("Saved confusion matrix: %s", cm_path)
