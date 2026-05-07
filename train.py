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
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Brain MRI Tumor Classifier (ResNet)")
    parser.add_argument("--data-root", required=True, help="Dataset root containing Training/ and Testing/")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet50"],
        default="resnet50",
        help="Backbone architecture (default: resnet50)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="CPU-friendly quick training mode (resnet18, 160px, fewer workers)",
    )
    parser.add_argument(
        "--output",
        default="model/brain_tumor_resnet50.pth",
        help="Output checkpoint path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cpu = device.type == "cpu"

    image_size = 224
    backbone = args.backbone
    workers = args.workers
    batch_size = args.batch_size
    output = args.output

    if args.fast:
        image_size = 160
        backbone = "resnet18"
        workers = 0
        batch_size = min(batch_size, 8)
        if output == "model/brain_tumor_resnet50.pth":
            output = "model/brain_tumor_resnet18_fast.pth"
    elif is_cpu and workers > 0:
        # Windows + CPU training can appear stalled with worker processes.
        workers = 0

    data_root = Path(args.data_root)
    train_dir = data_root / "Training"
    val_dir = data_root / "Testing"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Could not find expected folders: '{train_dir}' and '{val_dir}'."
        )

    tf = build_transforms(image_size=image_size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=tf["train"])
    val_ds = datasets.ImageFolder(str(val_dir), transform=tf["val"])

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise RuntimeError("Training and Testing class mappings differ. Fix dataset folder names.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=not is_cpu,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=not is_cpu,
    )

    model = build_model(num_classes=len(train_ds.classes), backbone=backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    print("Classes:", train_ds.classes)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Device: {device}")
    print(
        f"Config: backbone={backbone}, image_size={image_size}, "
        f"batch_size={batch_size}, workers={workers}, fast={args.fast}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy_from_logits(logits.detach(), labels)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_acc += accuracy_from_logits(logits, labels)

        train_loss /= max(len(train_loader), 1)
        train_acc /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        val_acc /= max(len(val_loader), 1)
        scheduler.step(val_acc)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "backbone": backbone,
            "image_size": image_size,
        }
        history.append(row)
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": round(best_acc, 5),
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "workers": workers,
                "backbone": backbone,
                "image_size": image_size,
                "fast_mode": args.fast,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")
    print(f"Saved weights to: {out_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
