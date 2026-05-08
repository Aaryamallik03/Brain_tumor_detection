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
    Grad-CAM tumor region visualization.

    Saves two images:
      - heatmap_path   : Grad-CAM heat overlay blended onto the original scan.
      - annotation_path: Clean original MRI with a bounding box and
                         "Tumor Region Detected" label (if a region is found).

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
        tensor = _transform_for_size(image_size)(img).unsqueeze(0).to(device)

        # ── Grad-CAM hooks on the last conv block ─────────────────────────────
        activations: dict = {}
        gradients: dict = {}

        def _fwd_hook(module, inp, out):
            activations["layer4"] = out.detach()

        def _bwd_hook(module, grad_in, grad_out):
            gradients["layer4"] = grad_out[0].detach()

        hook_fwd = model.layer4.register_forward_hook(_fwd_hook)
        hook_bwd = model.layer4.register_full_backward_hook(_bwd_hook)

        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        top_idx = int(torch.argmax(logits, dim=1).item())
        logits[0, top_idx].backward()

        hook_fwd.remove()
        hook_bwd.remove()

        # ── Compute Grad-CAM heatmap ───────────────────────────────────────────
        grads   = gradients["layer4"][0]           # [C, H, W]
        acts    = activations["layer4"][0]          # [C, H, W]
        weights = grads.mean(dim=(1, 2))            # [C]
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # [H, W]
        cam = cam / (cam.max() + 1e-8)

        # Upsample to original image size
        cam_pil = transforms.ToPILImage()(cam.cpu()).resize(img.size, Image.BILINEAR)
        cam_np  = np.array(cam_pil).astype(np.float32) / 255.0  # 0–1, HxW

        # ── Compute tumor area and bounding box ────────────────────────────────
        threshold = 0.40
        mask = cam_np > threshold
        area_pct = float(mask.mean()) * 100.0

        bbox = None
        if mask.any():
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin = int(np.where(rows)[0][0])
            rmax = int(np.where(rows)[0][-1])
            cmin = int(np.where(cols)[0][0])
            cmax = int(np.where(cols)[0][-1])
            bbox = (cmin, rmin, cmax, rmax)

        # ── IMAGE 1: Grad-CAM heat overlay ────────────────────────────────────
        r = np.clip(cam_np * 2.0,        0.0, 1.0)
        g = np.clip(cam_np * 2.0 - 0.7,  0.0, 1.0)
        b = np.zeros_like(cam_np)
        heat_rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat_rgb, mode="RGB")

        orig_np = np.array(img).astype(np.float32)
        heat_f  = np.array(heat_img).astype(np.float32)
        alpha   = (cam_np[:, :, np.newaxis] * 0.65).clip(0.0, 0.65)
        blended = (orig_np * (1.0 - alpha) + heat_f * alpha).clip(0, 255).astype(np.uint8)
        heatmap_out = Image.fromarray(blended)

        if bbox is not None:
            cmin, rmin, cmax, rmax = bbox
            lw = max(2, min(img.width, img.height) // 120)
            draw = ImageDraw.Draw(heatmap_out)
            draw.rectangle([cmin - 1, rmin - 1, cmax + 1, rmax + 1],
                           outline=(0, 0, 0), width=lw + 2)
            draw.rectangle([cmin, rmin, cmax, rmax],
                           outline=(255, 80, 60), width=lw)
            tag_x, tag_y = cmin, max(0, rmin - 22)
            draw.rectangle([tag_x, tag_y, tag_x + 118, tag_y + 20], fill=(255, 80, 60))
            draw.text((tag_x + 4, tag_y + 3), "Tumor Region", fill=(255, 255, 255))

        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_out.save(heatmap_path)

        # ── IMAGE 2: Clean annotation on original MRI ─────────────────────────
        if annotation_path is not None:
            ann = img.copy()
            draw = ImageDraw.Draw(ann)

            if bbox is not None:
                cmin, rmin, cmax, rmax = bbox
                lw = max(3, min(img.width, img.height) // 100)
                w, h = img.size

                # Corner tick marks instead of full box for a clinical look
                tick = max(16, int(min(cmax - cmin, rmax - rmin) * 0.18))
                corners = [
                    [(cmin, rmin + tick), (cmin, rmin), (cmin + tick, rmin)],
                    [(cmax - tick, rmin), (cmax, rmin), (cmax, rmin + tick)],
                    [(cmin, rmax - tick), (cmin, rmax), (cmin + tick, rmax)],
                    [(cmax - tick, rmax), (cmax, rmax), (cmax, rmax - tick)],
                ]
                for pts in corners:
                    draw.line([(pts[0][0], pts[0][1]),
                               (pts[1][0], pts[1][1]),
                               (pts[2][0], pts[2][1])],
                              fill=(255, 60, 50), width=lw)

                # Dashed bounding box
                dash_len, gap_len = 10, 6
                def _dashed_line(x0, y0, x1, y1):
                    dx, dy = x1 - x0, y1 - y0
                    length = max(abs(dx), abs(dy))
                    if length == 0:
                        return
                    sx, sy = dx / length, dy / length
                    pos, draw_seg = 0.0, True
                    while pos < length:
                        seg = dash_len if draw_seg else gap_len
                        end = min(pos + seg, length)
                        if draw_seg:
                            draw.line(
                                [(int(x0 + sx * pos), int(y0 + sy * pos)),
                                 (int(x0 + sx * end), int(y0 + sy * end))],
                                fill=(255, 60, 50), width=max(1, lw - 1)
                            )
                        pos += seg
                        draw_seg = not draw_seg

                _dashed_line(cmin, rmin, cmax, rmin)
                _dashed_line(cmax, rmin, cmax, rmax)
                _dashed_line(cmax, rmax, cmin, rmax)
                _dashed_line(cmin, rmax, cmin, rmin)

                # Banner label at top-left of box
                banner_h = 26
                tag_y = max(0, rmin - banner_h - 4)
                tag_x = cmin
                tag_w = min(cmin + 220, w - 1)
                draw.rectangle([tag_x, tag_y, tag_w, tag_y + banner_h],
                               fill=(220, 40, 40))
                draw.text((tag_x + 6, tag_y + 5),
                          "⚠  TUMOR REGION DETECTED", fill=(255, 255, 255))

                # Coordinate readout below box
                info_y = min(rmax + 6, h - 14)
                draw.text((cmin, info_y),
                          f"ROI  x:{cmin}-{cmax}  y:{rmin}-{rmax}",
                          fill=(255, 180, 60))
            else:
                # No tumor found — green label
                draw.rectangle([4, 4, 220, 30], fill=(40, 160, 80))
                draw.text((10, 8), "No suspicious region detected", fill=(255, 255, 255))

            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            ann.save(annotation_path)

        return round(area_pct, 1), None

    except Exception as exc:  # noqa: BLE001
        return None, f"Localization error: {exc}"
