"""
report.py
---------
Generates the downloadable Medical Report — both a plain-text version
(always available) and a formatted PDF version (when reportlab is
installed).
"""

from __future__ import annotations

import io
from typing import Dict, Optional

DISCLAIMER = (
    "This AI prediction is intended for educational purposes only and "
    "should not be considered a medical diagnosis. Always consult a "
    "qualified radiologist or healthcare professional."
)

MODEL_NAME = "Custom CNN (4-class MRI classifier)"
MODEL_ARCHITECTURE = "Sequential CNN — 4× Conv2D/MaxPooling blocks, Dense(512), Dropout(0.3), Softmax(4)"
NUM_CLASSES = 4


def build_report_text(
    patient: Dict[str, str],
    prediction_label: str,
    confidence: float,
    risk_level: str,
    model_accuracy: Optional[str] = None,
) -> str:
    """Build the plain-text medical report."""
    lines = []
    lines.append("=" * 60)
    lines.append("BRAIN TUMOR MRI — AI ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("PATIENT INFORMATION")
    lines.append("-" * 60)
    for label, key in [
        ("Patient ID", "patient_id"),
        ("Full Name", "full_name"),
        ("Age", "age"),
        ("Gender", "gender"),
        ("Height (cm)", "height"),
        ("Weight (kg)", "weight"),
        ("Blood Group", "blood_group"),
        ("MRI Scan Date", "scan_date"),
        ("Hospital", "hospital"),
        ("Referring Doctor", "doctor"),
    ]:
        lines.append(f"{label:<20}: {patient.get(key, '—')}")
    if patient.get("notes"):
        lines.append(f"{'Notes':<20}: {patient.get('notes')}")
    lines.append("")
    lines.append("AI PREDICTION RESULTS")
    lines.append("-" * 60)
    lines.append(f"{'Prediction':<20}: {prediction_label}")
    lines.append(f"{'Confidence':<20}: {confidence * 100:.2f}%")
    lines.append(f"{'Risk Level':<20}: {risk_level}")
    lines.append(f"{'Model':<20}: {MODEL_NAME}")
    if model_accuracy:
        lines.append(f"{'Model Accuracy':<20}: {model_accuracy}")
    lines.append("")
    lines.append("RECOMMENDATION")
    lines.append("-" * 60)
    if prediction_label.lower() == "no tumor":
        lines.append(
            "No tumor features were detected by the AI model. Routine "
            "follow-up as advised by your physician is recommended."
        )
    else:
        lines.append(
            "The AI model detected features consistent with the class "
            "above. Please consult a qualified radiologist or neurologist "
            "for confirmation and next steps."
        )
    lines.append("")
    lines.append("DISCLAIMER")
    lines.append("-" * 60)
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def build_report_pdf(
    patient: Dict[str, str],
    prediction_label: str,
    confidence: float,
    risk_level: str,
    overlay_image=None,
    model_accuracy: Optional[str] = None,
) -> Optional[bytes]:
    """
    Build a formatted PDF report using reportlab. Returns PDF bytes, or
    None if reportlab is not available (caller should fall back to text).
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleBlue", parent=styles["Title"], textColor=colors.HexColor("#2563EB")
    )
    heading_style = ParagraphStyle(
        "HeadingBlue", parent=styles["Heading2"], textColor=colors.HexColor("#2563EB"),
        spaceBefore=12, spaceAfter=6,
    )
    normal = styles["Normal"]

    story = []
    story.append(Paragraph("🧠 Brain Tumor MRI — AI Analysis Report", title_style))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Patient Information", heading_style))
    patient_rows = [
        ["Patient ID", patient.get("patient_id", "—"), "Full Name", patient.get("full_name", "—")],
        ["Age", str(patient.get("age", "—")), "Gender", patient.get("gender", "—")],
        ["Height", f"{patient.get('height', '—')} cm", "Weight", f"{patient.get('weight', '—')} kg"],
        ["Blood Group", patient.get("blood_group", "—"), "Scan Date", str(patient.get("scan_date", "—"))],
        ["Hospital", patient.get("hospital", "—"), "Doctor", patient.get("doctor", "—")],
    ]
    table = Table(patient_rows, colWidths=[3 * cm, 5 * cm, 3 * cm, 5 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#374151")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#374151")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FAFB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    story.append(Paragraph("AI Prediction Results", heading_style))
    risk_color_map = {"High": "#DC2626", "Medium": "#D97706", "Low": "#16A34A"}
    result_rows = [
        ["Prediction", prediction_label],
        ["Confidence", f"{confidence * 100:.2f}%"],
        ["Risk Level", risk_level],
        ["Model", MODEL_NAME],
    ]
    if model_accuracy:
        result_rows.append(["Model Accuracy", model_accuracy])
    result_table = Table(result_rows, colWidths=[4 * cm, 12 * cm])
    result_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor(risk_color_map.get(risk_level, "#374151"))),
    ]))
    story.append(result_table)

    if overlay_image is not None:
        try:
            img_buf = io.BytesIO()
            overlay_image.save(img_buf, format="PNG")
            img_buf.seek(0)
            story.append(Spacer(1, 0.4 * cm))
            story.append(Paragraph("Grad-CAM Overlay", heading_style))
            story.append(RLImage(img_buf, width=6 * cm, height=6 * cm))
        except Exception:
            pass

    story.append(Paragraph("Recommendation", heading_style))
    if prediction_label.lower() == "no tumor":
        rec_text = (
            "No tumor features were detected by the AI model. Routine "
            "follow-up as advised by your physician is recommended."
        )
    else:
        rec_text = (
            "The AI model detected features consistent with the class above. "
            "Please consult a qualified radiologist or neurologist for "
            "confirmation and next steps."
        )
    story.append(Paragraph(rec_text, normal))

    story.append(Paragraph("Disclaimer", heading_style))
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=normal, textColor=colors.HexColor("#6B7280"), fontSize=8, italic=True
    )
    story.append(Paragraph(DISCLAIMER, disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
