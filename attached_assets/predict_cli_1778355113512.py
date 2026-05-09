#!/usr/bin/env python3
"""
predict_cli.py — Command-line inference for Brain MRI Tumor Classifier.

Usage:
    python predict_cli.py <image_path> [--model model/brain_tumor_resnet50.pth]

Example:
    python predict_cli.py sample_images/glioma_test.jpg
"""

import argparse
import sys
from pathlib import Path

from classifier import load_model, predict, CLASS_INFO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brain MRI Tumor Classifier — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "⚠️  For educational / research purposes only.\n"
            "    NOT intended for medical diagnosis."
        ),
    )
    parser.add_argument("image", help="Path to MRI image (JPG/PNG/BMP/TIFF)")
    parser.add_argument(
        "--model",
        default="model/brain_tumor_resnet50.pth",
        help="Path to model weights (default: model/brain_tumor_resnet50.pth)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n🧠  Brain MRI Tumor Classifier")
    print(f"{'─' * 40}")
    print(f"  Model : {model_path}")
    print(f"  Image : {image_path}")
    print()

    model, err = load_model(model_path)
    if err:
        print(f"❌  Model error: {err}")
        sys.exit(1)

    # ── Run prediction ────────────────────────────────────────────────────────
    prediction, confidence, all_probs, pred_err = predict(model, image_path)
    if pred_err:
        print(f"❌  Prediction error: {pred_err}")
        sys.exit(1)

    icon = CLASS_INFO[prediction]["icon"]
    desc = CLASS_INFO[prediction]["description"]
    tumor_detected = prediction != "No Tumor"

    print(f"  Result     : {icon}  {prediction}")
    print(f"  Detection  : {'Positive' if tumor_detected else 'Negative'}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"  Info       : {desc}")
    print()
    print("  All class probabilities:")
    for p in all_probs:
        bar_len = int(p["prob"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        marker = " ◀" if p["label"] == prediction else ""
        print(f"    {p['icon']}  {p['label']:<18}  {bar}  {p['prob']:5.1f}%{marker}")

    print()
    print("⚠️  For educational / research purposes only. Not for medical diagnosis.")
    print()


if __name__ == "__main__":
    main()
