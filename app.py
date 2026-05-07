"""
Brain MRI Tumor Classifier — Flask Web App
Educational / Research use only. NOT for medical diagnosis.
"""

import os
import uuid
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from classifier import load_model, predict

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_CANDIDATES = [
    BASE_DIR / "model" / "brain_tumor_resnet18_fast.pth",
    BASE_DIR / "model" / "brain_tumor_resnet50.pth",
]
ALLOWED_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])

# Load model once at startup (graceful if missing)
model, model_error = load_model(MODEL_PATH)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", model_ready=(model is not None), model_error=model_error)


@app.route("/predict", methods=["POST"])
def predict_route():
    if model is None:
        flash(f"Model not loaded: {model_error}", "error")
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        flash(f"Unsupported file type '{ext}'. Please upload a JPG, PNG, BMP, or TIFF.", "error")
        return redirect(url_for("index"))

    # Save with unique name to avoid collisions
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_FOLDER / unique_name
    file.save(str(save_path))

    prediction, confidence, all_probs, pred_error = predict(model, save_path)

    if pred_error:
        flash(f"Prediction failed: {pred_error}", "error")
        return redirect(url_for("index"))

    image_url = url_for("static", filename=f"uploads/{unique_name}")
    tumor_detected = prediction != "No Tumor"
    return render_template(
        "result.html",
        image_url=image_url,
        prediction=prediction,
        confidence=confidence,
        tumor_detected=tumor_detected,
        all_probs=all_probs,
    )


@app.errorhandler(413)
def too_large(_):
    flash("File too large. Maximum size is 16 MB.", "error")
    return redirect(url_for("index"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
