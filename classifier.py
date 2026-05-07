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

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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
    output_path: Path,
) -> tuple[Optional[float], Optional[str]]:
    """
    Create a lightweight tumor saliency overlay image.

    Returns:
        (tumor_area_pct, error_message)
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
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
        out = Image.composite(blended, img, heat_img)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path)
        return round(area_pct, 1), None

    except Exception as exc:  # noqa: BLE001
        return None, f"Localization error: {exc}"
