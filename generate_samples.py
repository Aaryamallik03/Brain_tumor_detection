#!/usr/bin/env python3
"""
generate_samples.py — Creates simple synthetic grayscale MRI-like test images.

These are NOT real MRI scans — they are placeholder images for testing the
upload pipeline and CLI tool before you have real data.

Usage:
    python generate_samples.py
"""

from pathlib import Path
try:
    from PIL import Image, ImageDraw, ImageFilter
    import random, math
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    raise

OUTPUT_DIR = Path(__file__).parent / "sample_images"
OUTPUT_DIR.mkdir(exist_ok=True)

LABELS = ["glioma", "meningioma", "no_tumor", "pituitary"]

def make_brain_image(label: str, seed: int = 42) -> Image.Image:
    """Draw a rough synthetic brain cross-section silhouette."""
    random.seed(seed)
    W, H = 256, 256
    img = Image.new("RGB", (W, H), (8, 8, 12))
    draw = ImageDraw.Draw(img)

    # Brain outline (ellipse)
    cx, cy = W // 2, H // 2
    draw.ellipse([cx-90, cy-75, cx+90, cy+80], outline=(140,140,155), width=2)
    # Inner hemisphere line
    draw.line([cx-88, cy, cx+88, cy], fill=(80,80,90), width=1)

    # Random "tissue" texture dots
    for _ in range(400):
        x = random.randint(cx-80, cx+80)
        y = random.randint(cy-65, cy+70)
        r = random.randint(1, 3)
        g = random.randint(50, 90)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(g, g, g+5))

    # Draw a "lesion" marker for tumor classes
    if label != "no_tumor":
        positions = {
            "glioma":      (cx+20, cy-20),
            "meningioma":  (cx-30, cy-55),
            "pituitary":   (cx,    cy+55),
        }
        lx, ly = positions[label]
        lr = random.randint(12, 20)
        brightness = random.randint(190, 230)
        draw.ellipse([lx-lr, ly-lr, lx+lr, ly+lr],
                     fill=(brightness, brightness-20, brightness-40),
                     outline=(255,255,200), width=1)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


if __name__ == "__main__":
    for i, label in enumerate(LABELS):
        path = OUTPUT_DIR / f"{label}_sample.jpg"
        img = make_brain_image(label, seed=i*7+3)
        img.save(str(path), quality=90)
        print(f"  Created: {path}")

    print(f"\n✅  {len(LABELS)} sample images saved to '{OUTPUT_DIR}'")
    print("   These are SYNTHETIC placeholders — not real MRI scans.")
