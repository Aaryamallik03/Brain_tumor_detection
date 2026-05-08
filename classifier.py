"""
classifier.py — Model loading and inference for Brain MRI Tumor Classifier.

Classes: Glioma | Meningioma | Pituitary Tumor | No Tumor
Architecture: ResNet-50 with fine-tuned final FC layer (transfer learning)
"""

from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# Standard ImageNet normalisation (used during training)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

CLASS_INFO = {
    "Glioma": {
        "description": "A tumor arising from glial cells in the brain or spine.",
        "color": "#e05252",
        "icon": "🔴",
    },
    "Meningioma": {
        "description": "A tumor forming on the membranes surrounding the brain and spinal cord.",
        "color": "#e08c3a",
        "icon": "🟠",
    },
    "No Tumor": {
        "description": "No tumor detected in the MRI scan.",
        "color": "#4caf7d",
        "icon": "🟢",
    },
    "Pituitary Tumor": {
        "description": "A tumor in the pituitary gland at the base of the brain.",
        "color": "#7a6fe0",
        "icon": "🟣",
    },
}


# ── Model Builder ─────────────────────────────────────────────────────────────

def _build_model(num_classes: int = 4, backbone: str = "resnet50") -> nn.Module:
    """Create a ResNet model with a replaced fully-connected head."""
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _transform_for_size(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _canonical_label(raw: str) -> str:
    lookup = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "no_tumor": "No Tumor",
        "pituitary": "Pituitary Tumor",
    }
    key = raw.strip().lower().replace(" ", "").replace("-", "_")
    key = key.replace("_tumor", "tumor")
    return lookup.get(key, raw)


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(model_path: Path) -> tuple[Optional[nn.Module], Optional[str]]:
    """
    Load trained weights from *model_path*.
    Returns (model, None) on success, (None, error_message) on failure.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        msg = (
            f"Model file not found at '{model_path}'. "
            "Please download the weights and place them in the 'model/' folder. "
            "See README.md for instructions."
        )
        logger.warning(msg)
        return None, msg

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(model_path, map_location=device)
        # Support both raw state_dict and checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Infer architecture from FC layer width:
        #   resnet18 -> fc.weight shape [num_classes, 512]
        #   resnet50 -> fc.weight shape [num_classes, 2048]
        backbone = "resnet50"
        fc_w = state.get("fc.weight") if isinstance(state, dict) else None
        if isinstance(fc_w, torch.Tensor):
            if fc_w.ndim == 2 and fc_w.shape[1] == 512:
                backbone = "resnet18"
            elif fc_w.ndim == 2 and fc_w.shape[1] == 2048:
                backbone = "resnet50"

        net = _build_model(num_classes=len(CLASS_NAMES), backbone=backbone)

        net.load_state_dict(state)
        net.to(device)
        net.eval()
        # Defaults for inference metadata
        net._input_size = 224
        net._class_names = CLASS_NAMES

        # If sidecar metrics exist, align preprocessing/class labels with training.
        metrics_path = model_path.with_suffix(".metrics.json")
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                image_size = int(metrics.get("image_size", 224))
                classes = metrics.get("classes", [])
                if image_size > 0:
                    net._input_size = image_size
                if isinstance(classes, list) and len(classes) == len(CLASS_NAMES):
                    net._class_names = [_canonical_label(c) for c in classes]
            except Exception:  # noqa: BLE001
                logger.warning("Could not parse metrics sidecar at %s", metrics_path)

        logger.info(
            "Model loaded successfully from %s (device=%s, backbone=%s)",
            model_path,
            device,
            backbone,
        )
        return net, None

    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load model weights: {exc}"
        logger.error(msg)
        return None, msg


def predict(
    model: nn.Module,
    image_path: Path,
) -> tuple[Optional[str], Optional[float], Optional[list[dict]], Optional[str]]:
    """
    Run inference on a single image.

    Returns:
        (predicted_class, confidence_pct, all_probs_list, error_message)
        all_probs_list: [{"label": str, "prob": float, "color": str, "icon": str}, ...]
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return None, None, None, f"Image file not found: {image_path}"

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"Cannot open image: {exc}"

    try:
        device = next(model.parameters()).device
        image_size = getattr(model, "_input_size", 224)
        class_names = getattr(model, "_class_names", CLASS_NAMES)
        tensor = _transform_for_size(image_size)(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top_idx    = int(probs.argmax())
        prediction = class_names[top_idx]
        confidence = float(probs[top_idx]) * 100

        all_probs = [
            {
                "label": class_names[i],
                "prob":  round(float(probs[i]) * 100, 1),
                "color": CLASS_INFO[class_names[i]]["color"],
                "icon":  CLASS_INFO[class_names[i]]["icon"],
            }
            for i in range(len(class_names))
        ]
        # Sort descending
        all_probs.sort(key=lambda x: x["prob"], reverse=True)

        return prediction, round(confidence, 1), all_probs, None

    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"Inference error: {exc}"


def localize_tumor(
    model: nn.Module,
    image_path: Path,
    heatmap_path: Path,
    annotation_path: Optional[Path] = None,
) -> tuple[Optional[float], Optional[str]]:
    """
    Generate two visualizations:
      - heatmap_path   : Original gradient saliency red overlay (unchanged from v1).
      - annotation_path: Clean original MRI with bounding box and "Tumor Region" label.

    Returns:
        (tumor_area_pct, error_message)
    """
    image_path = Path(image_path)
    heatmap_path = Path(heatmap_path)
    if annotation_path is not None:
        annotation_path = Path(annotation_path)

    if not image_path.exists():
        return None, f"Image file not found: {image_path}"

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return None, f"Cannot open image: {exc}"

    try:
        device = next(model.parameters()).device
        image_size = getattr(model, "_input_size", 224)

        # ── Original gradient saliency (unchanged) ─────────────────────────────
        tensor = _transform_for_size(image_size)(img).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        top_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, top_idx]
        score.backward()

        grad = tensor.grad.detach().abs()[0]  # [C, H, W]
        heat = grad.mean(dim=0)               # [H, W]
        heat = heat / (heat.max() + 1e-8)

        mask = heat > 0.55
        area_pct = float(mask.float().mean().item()) * 100.0

        heat_img = transforms.ToPILImage()(heat.cpu()).convert("L")
        heat_img = heat_img.resize(img.size)

        red_overlay = Image.new("RGB", img.size, (255, 50, 50))
        blended = Image.blend(img, red_overlay, alpha=0.35)
        heatmap_out = Image.composite(blended, img, heat_img)

        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_out.save(heatmap_path)

        # ── Bounding box from the saliency mask for annotation ─────────────────
        bbox = None
        mask_np = mask.cpu().numpy()
        # Upsample mask to original image size
        mask_pil = transforms.ToPILImage()(heat.cpu()).convert("L").resize(img.size, Image.BILINEAR)
        mask_full = np.array(mask_pil).astype(np.float32) / 255.0 > 0.55

        if mask_full.any():
            rows = np.any(mask_full, axis=1)
            cols = np.any(mask_full, axis=0)
            rmin = int(np.where(rows)[0][0])
            rmax = int(np.where(rows)[0][-1])
            cmin = int(np.where(cols)[0][0])
            cmax = int(np.where(cols)[0][-1])
            bbox = (cmin, rmin, cmax, rmax)

        # ── IMAGE 2: Annotated original MRI ───────────────────────────────────
        if annotation_path is not None:
            ann = img.copy()
            draw = ImageDraw.Draw(ann)
            w, h = img.size

            if bbox is not None:
                cmin, rmin, cmax, rmax = bbox
                lw = max(3, min(w, h) // 100)

                # Corner L-shaped tick marks
                tick = max(16, int(min(cmax - cmin, rmax - rmin) * 0.18))
                corners = [
                    [(cmin, rmin + tick), (cmin, rmin), (cmin + tick, rmin)],
                    [(cmax - tick, rmin), (cmax, rmin), (cmax, rmin + tick)],
                    [(cmin, rmax - tick), (cmin, rmax), (cmin + tick, rmax)],
                    [(cmax - tick, rmax), (cmax, rmax), (cmax, rmax - tick)],
                ]
                for pts in corners:
                    draw.line(
                        [(pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (pts[2][0], pts[2][1])],
                        fill=(255, 60, 50), width=lw,
                    )

                # Dashed bounding box
                dash_len, gap_len = 10, 6

                def _dashed(x0, y0, x1, y1):
                    dx, dy = x1 - x0, y1 - y0
                    length = max(abs(dx), abs(dy))
                    if length == 0:
                        return
                    sx, sy = dx / length, dy / length
                    pos, on = 0.0, True
                    while pos < length:
                        seg = dash_len if on else gap_len
                        end = min(pos + seg, length)
                        if on:
                            draw.line(
                                [(int(x0 + sx * pos), int(y0 + sy * pos)),
                                 (int(x0 + sx * end), int(y0 + sy * end))],
                                fill=(255, 60, 50), width=max(1, lw - 1),
                            )
                        pos += seg
                        on = not on

                _dashed(cmin, rmin, cmax, rmin)
                _dashed(cmax, rmin, cmax, rmax)
                _dashed(cmax, rmax, cmin, rmax)
                _dashed(cmin, rmax, cmin, rmin)

                # Red banner label above box
                banner_h = 26
                tag_y = max(0, rmin - banner_h - 4)
                tag_x2 = min(cmin + 222, w - 1)
                draw.rectangle([cmin, tag_y, tag_x2, tag_y + banner_h], fill=(220, 40, 40))
                draw.text((cmin + 6, tag_y + 5), "TUMOR REGION DETECTED", fill=(255, 255, 255))

                # Coordinate readout below box
                info_y = min(rmax + 6, h - 14)
                draw.text((cmin, info_y), f"ROI  x:{cmin}-{cmax}  y:{rmin}-{rmax}",
                          fill=(255, 200, 60))
            else:
                draw.rectangle([4, 4, 238, 30], fill=(40, 160, 80))
                draw.text((10, 8), "No suspicious region detected", fill=(255, 255, 255))

            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            ann.save(annotation_path)

        return round(area_pct, 1), None

    except Exception as exc:  # noqa: BLE001
        return None, f"Localization error: {exc}"
