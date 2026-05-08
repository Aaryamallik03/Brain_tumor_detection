"""
Brain MRI Tumor Classifier — Flask Web App
Educational / Research use only. NOT for medical diagnosis.
"""

import os
import uuid
from datetime import datetime
import csv
import io
import sqlite3
from collections import Counter
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, Response
from classifier import load_model, predict, localize_tumor

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_CANDIDATES = [
    BASE_DIR / "model" / "brain_tumor_resnet18_fast.pth",
    BASE_DIR / "model" / "brain_tumor_resnet50.pth",
]
ALLOWED_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
DB_PATH = BASE_DIR / "monitoring.db"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                tumor_detected INTEGER NOT NULL,
                tumor_area_pct REAL,
                image_url TEXT NOT NULL,
                heatmap_url TEXT
            )
            """
        )
        conn.commit()


def insert_prediction(row: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (created_at, prediction, confidence, tumor_detected, tumor_area_pct, image_url, heatmap_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["time"],
                row["prediction"],
                row["confidence"],
                1 if row["tumor_detected"] else 0,
                row["tumor_area_pct"],
                row["image_url"],
                row["heatmap_url"],
            ),
        )
        conn.commit()


def recent_predictions(limit: int = 100) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, prediction, confidence, tumor_detected, tumor_area_pct, image_url, heatmap_url
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "time": r["created_at"],
            "prediction": r["prediction"],
            "confidence": round(float(r["confidence"]), 1),
            "tumor_detected": bool(r["tumor_detected"]),
            "tumor_area_pct": round(float(r["tumor_area_pct"]), 1) if r["tumor_area_pct"] is not None else None,
            "image_url": r["image_url"],
            "heatmap_url": r["heatmap_url"],
        }
        for r in rows
    ]

# Load model once at startup (graceful if missing)
model, model_error = load_model(MODEL_PATH)
init_db()
if model is None:
    app.logger.warning(
        "Model not loaded on startup. Checked candidates: %s",
        ", ".join(str(p) for p in MODEL_CANDIDATES),
    )


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

    stem = Path(unique_name).stem
    heatmap_name     = f"{stem}_heatmap{ext}"
    annotation_name  = f"{stem}_annotation{ext}"
    tumor_spot_name  = f"{stem}_spot{ext}"
    heatmap_path     = UPLOAD_FOLDER / heatmap_name
    annotation_path  = UPLOAD_FOLDER / annotation_name
    tumor_spot_path  = UPLOAD_FOLDER / tumor_spot_name

    area_pct, loc_error = localize_tumor(
        model, save_path, heatmap_path, annotation_path, tumor_spot_path
    )
    if loc_error:
        area_pct = None

    image_url       = url_for("static", filename=f"uploads/{unique_name}")
    heatmap_url     = url_for("static", filename=f"uploads/{heatmap_name}") if heatmap_path.exists() else None
    annotation_url  = url_for("static", filename=f"uploads/{annotation_name}") if annotation_path.exists() else None
    tumor_spot_url  = url_for("static", filename=f"uploads/{tumor_spot_name}") if tumor_spot_path.exists() else None
    tumor_detected  = prediction != "No Tumor"
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": prediction,
        "confidence": confidence,
        "tumor_detected": tumor_detected,
        "tumor_area_pct": area_pct,
        "image_url": image_url,
        "heatmap_url": heatmap_url,
    }
    insert_prediction(row)
    return render_template(
        "result.html",
        image_url=image_url,
        heatmap_url=heatmap_url,
        annotation_url=annotation_url,
        tumor_spot_url=tumor_spot_url,
        prediction=prediction,
        confidence=confidence,
        tumor_detected=tumor_detected,
        tumor_area_pct=area_pct,
        all_probs=all_probs,
    )


@app.route("/monitor")
def monitor():
    predictions = recent_predictions(limit=200)
    class_counts = Counter(p["prediction"] for p in predictions)
    ordered_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
    chart_data = [{"label": label, "count": class_counts.get(label, 0)} for label in ordered_labels]
    return render_template("monitor.html", predictions=predictions, chart_data=chart_data)


@app.route("/monitor.csv")
def monitor_csv():
    predictions = recent_predictions(limit=2000)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["time", "prediction", "detection", "confidence", "tumor_area_pct", "image_url", "heatmap_url"])
    for p in predictions:
        writer.writerow(
            [
                p["time"],
                p["prediction"],
                "Positive" if p["tumor_detected"] else "Negative",
                p["confidence"],
                p["tumor_area_pct"] if p["tumor_area_pct"] is not None else "",
                p["image_url"],
                p["heatmap_url"] or "",
            ]
        )

    csv_data = stream.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_monitoring.csv"},
    )


@app.route("/health")
def health():
    status = "ok" if model is not None else "degraded"
    return {
        "status": status,
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "db_path": str(DB_PATH),
    }, 200 if model is not None else 503


@app.errorhandler(413)
def too_large(_):
    flash("File too large. Maximum size is 16 MB.", "error")
    return redirect(url_for("index"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
