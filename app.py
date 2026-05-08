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
from flask import Flask, request, render_template, redirect, url_for, flash, Response, session
from classifier import load_model, predict, localize_tumor

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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

    session["last_result"] = {
        "stem": stem,
        "ext": ext,
        "prediction": prediction,
        "confidence": confidence,
        "tumor_detected": tumor_detected,
        "tumor_area_pct": area_pct,
        "timestamp": row["time"],
        "all_probs": all_probs,
    }

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


@app.route("/download_pdf")
def download_pdf():
    data = session.get("last_result")
    if not data:
        flash("No result to export. Please classify a scan first.", "error")
        return redirect(url_for("index"))

    stem      = data["stem"]
    ext       = data["ext"]
    prediction    = data["prediction"]
    confidence    = data["confidence"]
    tumor_detected = data["tumor_detected"]
    tumor_area_pct = data.get("tumor_area_pct")
    timestamp     = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    all_probs     = data.get("all_probs", [])

    orig_path  = UPLOAD_FOLDER / f"{stem}{ext}"
    heat_path  = UPLOAD_FOLDER / f"{stem}_heatmap{ext}"
    spot_path  = UPLOAD_FOLDER / f"{stem}_spot{ext}"
    ann_path   = UPLOAD_FOLDER / f"{stem}_annotation{ext}"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", fontSize=20, textColor=colors.HexColor("#1a237e"),
                                  spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_style   = ParagraphStyle("sub", fontSize=9, textColor=colors.grey,
                                  spaceAfter=2, alignment=TA_CENTER)
    h2_style    = ParagraphStyle("h2", fontSize=13, textColor=colors.HexColor("#1a237e"),
                                  spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("body", fontSize=9, textColor=colors.HexColor("#333333"),
                                  spaceAfter=4, leading=14)
    warn_style  = ParagraphStyle("warn", fontSize=8, textColor=colors.HexColor("#7a5000"),
                                  backColor=colors.HexColor("#fff8e1"), borderPadding=6,
                                  spaceAfter=6, leading=12)

    IMG_W = 7.5 * cm

    def add_image(path):
        if path.exists():
            try:
                return RLImage(str(path), width=IMG_W, height=IMG_W)
            except Exception:
                pass
        return Paragraph("<i>Image not available</i>", body_style)

    story = []

    story.append(Paragraph("Brain MRI Tumor Classifier", title_style))
    story.append(Paragraph("Classification Result Report", sub_style))
    story.append(Paragraph(f"Generated: {timestamp}", sub_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "⚠️  DISCLAIMER: This report is for educational and research purposes only. "
        "It is NOT a medical device and must not be used for clinical diagnosis, "
        "treatment decisions, or any medical purpose. Always consult a qualified healthcare professional.",
        warn_style,
    ))

    story.append(Paragraph("Classification Result", h2_style))
    detection_text = "POSITIVE — Tumor Detected" if tumor_detected else "NEGATIVE — No Tumor Detected"
    detection_color = colors.HexColor("#c62828") if tumor_detected else colors.HexColor("#2e7d32")
    result_data = [
        ["Predicted Class", prediction],
        ["Confidence", f"{confidence}%"],
        ["Tumor Detection", detection_text],
        ["Tumor Area", f"{tumor_area_pct}% of scan" if tumor_area_pct is not None else "N/A"],
        ["Scan Timestamp", timestamp],
    ]
    result_table = Table(result_data, colWidths=[5*cm, 11.5*cm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR",    (0, 0), (0, -1), colors.HexColor("#1a237e")),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f7f8fc")]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",    (1, 2), (1, 2), detection_color),
        ("FONTNAME",     (1, 2), (1, 2), "Helvetica-Bold"),
    ]))
    story.append(result_table)

    if all_probs:
        story.append(Paragraph("Class Probabilities", h2_style))
        prob_data = [["Class", "Probability"]]
        for p in all_probs:
            prob_data.append([p["label"], f"{p['prob']}%"])
        prob_table = Table(prob_data, colWidths=[9*cm, 7.5*cm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a237e")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fc")]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        story.append(prob_table)

    story.append(Paragraph("Scan Images", h2_style))
    img_row = []
    img_labels = []
    for path, label in [
        (orig_path, "Original MRI"),
        (spot_path, "Tumor Region Overlay"),
        (heat_path, "Saliency Heatmap"),
        (ann_path,  "Annotation"),
    ]:
        img_row.append(add_image(path))
        img_labels.append(Paragraph(label, ParagraphStyle("lbl", fontSize=8,
                          alignment=TA_CENTER, textColor=colors.HexColor("#555555"))))

    img_table = Table(
        [img_row, img_labels],
        colWidths=[4*cm, 4*cm, 4*cm, 4*cm],
        hAlign="CENTER",
    )
    img_table.setStyle(TableStyle([
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This document was generated by the Brain MRI Tumor Classifier — "
        "an educational tool built with PyTorch (ResNet-18) and Flask. "
        "Not validated for clinical use.",
        ParagraphStyle("footer", fontSize=7, textColor=colors.grey, alignment=TA_CENTER),
    ))

    doc.build(story)
    buf.seek(0)

    filename = f"brain_mri_report_{stem[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        buf.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
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
