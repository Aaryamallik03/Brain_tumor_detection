#!/usr/bin/env python3
"""
train.py - Train brain MRI tumor classification model (ResNet family).

Expected dataset layout:
dataset_root/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/        (optional when --stratify is used)
    glioma/
    meningioma/
    notumor/
    pituitary/

New features (added without breaking existing logic):
  --stratify        Pool Training+Testing and do stratified 80/20 split (default: True)
  --no-stratify     Fall back to original folder-based train/val split
  Outputs after training:
    <stem>_confusion_matrix.png
    <stem>_curves.png
    <stem>_classification_report.txt
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import datasets, models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ── Transform-aware subset wrapper ────────────────────────────────────────────

class _TransformSubset(Dataset):
    """Wraps a Subset and applies a transform, keeping labels intact."""
    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple:
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {"train": train_tf, "val": eval_tf}


def build_model(num_classes: int, backbone: str = "resnet50") -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Use 'resnet18' or 'resnet50'.")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# ── Stratified split ──────────────────────────────────────────────────────────

def make_stratified_loaders(
    data_root: Path,
    tf: Dict[str, transforms.Compose],
    batch_size: int,
    workers: int,
    is_cpu: bool,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str], dict]:
    """
    Pool Training/ + Testing/ (if present), apply stratified 80/20 split,
    return (train_loader, val_loader, class_names, class_to_idx).
    """
    train_dir = data_root / "Training"
    test_dir  = data_root / "Testing"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    # Load without transforms so PIL images are returned for splitting
    train_folder = datasets.ImageFolder(str(train_dir), transform=None)
    all_samples: List[Tuple[str, int]] = list(train_folder.samples)
    class_to_idx = train_folder.class_to_idx

    if test_dir.exists():
        test_folder = datasets.ImageFolder(str(test_dir), transform=None)
        if test_folder.class_to_idx == class_to_idx:
            all_samples += list(test_folder.samples)
        else:
            print("WARNING: Testing/ class mapping differs from Training/ — ignoring Testing/ for stratified split.")

    # Build a unified dataset from all_samples
    combined = _SampleListDataset(all_samples)
    all_labels = [s[1] for s in all_samples]

    indices = list(range(len(combined)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=all_labels,
        random_state=random_state,
    )

    train_subset = Subset(combined, train_idx)
    val_subset   = Subset(combined, val_idx)

    train_ds = _TransformSubset(train_subset, tf["train"])
    val_ds   = _TransformSubset(val_subset,   tf["val"])

    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    print(f"Stratified 80/20 split — Train: {len(train_ds)} | Val: {len(val_ds)}")
    _print_split_distribution(all_labels, train_idx, val_idx, class_names)

    pin = not is_cpu
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin)
    return train_loader, val_loader, class_names, class_to_idx


class _SampleListDataset(Dataset):
    """Minimal dataset that reads images from a (path, label) list."""
    def __init__(self, samples: List[Tuple[str, int]]) -> None:
        from PIL import Image
        self._samples = samples
        self._loader  = Image.open

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, label = self._samples[idx]
        img = self._loader(path).convert("RGB")
        return img, label


def _print_split_distribution(
    all_labels: List[int], train_idx: List[int], val_idx: List[int], class_names: List[str]
) -> None:
    train_labels = [all_labels[i] for i in train_idx]
    val_labels   = [all_labels[i] for i in val_idx]
    print(f"  {'Class':<20} {'Train':>8} {'Val':>8}")
    print(f"  {'-'*38}")
    for ci, name in enumerate(class_names):
        t = train_labels.count(ci)
        v = val_labels.count(ci)
        print(f"  {name:<20} {t:>8} {v:>8}")


# ── Post-training evaluation & plots ─────────────────────────────────────────

def evaluate_and_report(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    out_stem: Path,
) -> None:
    """Run full evaluation on val set: classification report + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds  = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\n── Classification Report ──────────────────────────────")
    print(report)
    report_path = out_stem.parent / f"{out_stem.name}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved classification report → {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="lightgray",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix — Validation Set", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = out_stem.parent / f"{out_stem.name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix    → {cm_path}")


def plot_curves(history: List[dict], out_stem: Path) -> None:
    """Save train vs validation accuracy and loss graphs side by side."""
    epochs     = [r["epoch"]      for r in history]
    train_acc  = [r["train_acc"]  * 100 for r in history]
    val_acc    = [r["val_acc"]    * 100 for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"]   for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax1.plot(epochs, train_acc, "o-", color="#1565c0", linewidth=2, label="Train Accuracy")
    ax1.plot(epochs, val_acc,   "s--", color="#e53935", linewidth=2, label="Val Accuracy")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Train vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Loss
    ax2.plot(epochs, train_loss, "o-", color="#1565c0", linewidth=2, label="Train Loss")
    ax2.plot(epochs, val_loss,   "s--", color="#e53935", linewidth=2, label="Val Loss")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Train vs Validation Loss", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    fig.suptitle("Training Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    curves_path = out_stem.parent / f"{out_stem.name}_curves.png"
    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves     → {curves_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Brain MRI Tumor Classifier (ResNet)")
    parser.add_argument("--data-root", required=True,
                        help="Dataset root containing Training/ (and optionally Testing/)")
    parser.add_argument("--epochs",     type=int,   default=12,   help="Number of epochs")
    parser.add_argument("--batch-size", type=int,   default=16,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--workers",    type=int,   default=2,    help="DataLoader workers")
    parser.add_argument(
        "--backbone", choices=["resnet18", "resnet50"], default="resnet50",
        help="Backbone architecture (default: resnet50)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="CPU-friendly quick training mode (resnet18, 160px, fewer workers)",
    )
    parser.add_argument(
        "--output", default="model/brain_tumor_resnet50.pth",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--no-stratify", action="store_true",
        help="Fall back to original folder-based split (Training/ vs Testing/) instead of stratified 80/20",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cpu = device.type == "cpu"

    image_size = 224
    backbone   = args.backbone
    workers    = args.workers
    batch_size = args.batch_size
    output     = args.output

    if args.fast:
        image_size = 160
        backbone   = "resnet18"
        workers    = 0
        batch_size = min(batch_size, 8)
        if output == "model/brain_tumor_resnet50.pth":
            output = "model/brain_tumor_resnet18_fast.pth"
    elif is_cpu and workers > 0:
        workers = 0

    data_root = Path(args.data_root)
    tf        = build_transforms(image_size=image_size)

    # ── Build data loaders ────────────────────────────────────────────────────
    use_stratify = not args.no_stratify

    if use_stratify:
        print("Using stratified 80/20 split (pool Training/ + Testing/).")
        train_loader, val_loader, class_names, class_to_idx = make_stratified_loaders(
            data_root, tf, batch_size, workers, is_cpu,
            test_size=0.20, random_state=42,
        )
    else:
        print("Using original folder-based split (Training/ vs Testing/).")
        train_dir = data_root / "Training"
        val_dir   = data_root / "Testing"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"Could not find expected folders: '{train_dir}' and '{val_dir}'."
            )
        train_ds = datasets.ImageFolder(str(train_dir), transform=tf["train"])
        val_ds   = datasets.ImageFolder(str(val_dir),   transform=tf["val"])
        if train_ds.class_to_idx != val_ds.class_to_idx:
            raise RuntimeError("Training and Testing class mappings differ.")
        class_names  = train_ds.classes
        class_to_idx = train_ds.class_to_idx
        pin          = not is_cpu
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=workers, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=workers, pin_memory=pin)
        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    print(f"Classes : {class_names}")
    print(f"Device  : {device}")
    print(
        f"Config  : backbone={backbone}, image_size={image_size}, "
        f"batch_size={batch_size}, workers={workers}, fast={args.fast}"
    )

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model     = build_model(num_classes=len(class_names), backbone=backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history    = []

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n── Training ───────────────────────────────────────────")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc  += accuracy_from_logits(logits.detach(), labels)

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss += loss.item()
                val_acc  += accuracy_from_logits(logits, labels)

        train_loss /= max(len(train_loader), 1)
        train_acc  /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader), 1)
        val_acc    /= max(len(val_loader), 1)
        scheduler.step(val_acc)

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "lr":         optimizer.param_groups[0]["lr"],
            "backbone":   backbone,
            "image_size": image_size,
        }
        history.append(row)
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ── Save best model weights ───────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc":  round(best_acc, 5),
                "classes":       class_names,
                "class_to_idx":  class_to_idx,
                "epochs":        args.epochs,
                "batch_size":    batch_size,
                "workers":       workers,
                "backbone":      backbone,
                "image_size":    image_size,
                "fast_mode":     args.fast,
                "stratified":    use_stratify,
                "history":       history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(f"Best validation accuracy : {best_acc*100:.2f}%")
    print(f"Saved weights            → {out_path}")
    print(f"Saved metrics JSON       → {metrics_path}")

    # ── Post-training evaluation ──────────────────────────────────────────────
    out_stem = out_path.with_suffix("")  # strip .pth for derived filenames
    model.load_state_dict(best_state)

    print("\n── Post-training Evaluation ───────────────────────────")
    evaluate_and_report(model, val_loader, class_names, device, out_stem)
    plot_curves(history, out_stem)

    print("\nAll outputs saved to:", out_path.parent)


if __name__ == "__main__":
    main()
