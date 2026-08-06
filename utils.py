"""
utils.py
--------
Shared helpers: medical reference data, image metadata, formatting, and
session-state utilities used across the Streamlit app.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict

from PIL import Image

# ------------------------------------------------------------------ #
# Smart Tumor Information Panel — reference content per class.
# Educational summaries only; not a substitute for clinical review.
# ------------------------------------------------------------------ #
TUMOR_INFO: Dict[str, Dict[str, str]] = {
    "glioma": {
        "icon": "🧠",
        "medical_name": "Glioma",
        "description": (
            "A tumor that arises from glial cells, the supportive tissue "
            "of the brain and spinal cord. Gliomas vary widely in grade "
            "and growth rate."
        ),
        "location": "Cerebral hemispheres, brainstem, cerebellum, or spinal cord",
        "characteristics": (
            "Can be low-grade (slow growing) or high-grade (aggressive). "
            "Often infiltrates surrounding brain tissue rather than forming "
            "a clean border."
        ),
        "symptoms": "Headaches, seizures, memory or personality changes, weakness on one side, vision changes",
        "treatment": "Surgical resection, radiotherapy, chemotherapy, or a combination depending on grade",
        "risk_level": "High",
        "color": "#DC2626",
    },
    "meningioma": {
        "icon": "🎗️",
        "medical_name": "Meningioma",
        "description": (
            "A tumor arising from the meninges, the membranes surrounding "
            "the brain and spinal cord. Most meningiomas are benign and "
            "slow growing."
        ),
        "location": "Meninges — often near the skull base, cerebral convexity, or falx",
        "characteristics": (
            "Typically well-defined and slow growing; symptoms usually arise "
            "from pressure on adjacent brain structures rather than invasion."
        ),
        "symptoms": "Headaches, gradual vision changes, weakness, seizures (depending on location)",
        "treatment": "Observation for small/asymptomatic cases, surgery, or radiotherapy for growing tumors",
        "risk_level": "Medium",
        "color": "#D97706",
    },
    "pituitary": {
        "icon": "⚕️",
        "medical_name": "Pituitary Tumor",
        "description": (
            "A growth in the pituitary gland, the small hormone-producing "
            "gland at the base of the brain. Most are benign adenomas."
        ),
        "location": "Pituitary gland, sella turcica (base of the skull)",
        "characteristics": (
            "May be hormone-secreting (functioning) or non-secreting. "
            "Can compress the optic chiasm as it grows."
        ),
        "symptoms": "Vision problems, hormonal imbalances, headaches, fatigue, menstrual or growth changes",
        "treatment": "Medication, surgery (often trans-sphenoidal), or radiotherapy depending on type",
        "risk_level": "Medium",
        "color": "#D97706",
    },
    "notumor": {
        "icon": "✅",
        "medical_name": "No Tumor Detected",
        "description": (
            "The AI model did not identify features consistent with glioma, "
            "meningioma, or pituitary tumor in this scan."
        ),
        "location": "—",
        "characteristics": "No abnormal mass detected by the model in this image.",
        "symptoms": "—",
        "treatment": "No treatment indicated by this result; routine follow-up as advised by a physician",
        "risk_level": "Low",
        "color": "#16A34A",
    },
}


def get_tumor_info(class_key: str) -> Dict[str, str]:
    """Return the reference info dict for a predicted class key."""
    return TUMOR_INFO.get(class_key, TUMOR_INFO["notumor"])


def risk_color(risk_level: str) -> str:
    return {"High": "#DC2626", "Medium": "#D97706", "Low": "#16A34A"}.get(risk_level, "#6B7280")


def confidence_color(level: str) -> str:
    return {"High": "#16A34A", "Medium": "#D97706", "Low": "#DC2626"}.get(level, "#6B7280")


# ------------------------------------------------------------------ #
# Image metadata
# ------------------------------------------------------------------ #
def get_image_metadata(uploaded_file, image: Image.Image) -> Dict[str, str]:
    """Return display-ready metadata for an uploaded image."""
    size_bytes = getattr(uploaded_file, "size", None)
    if size_bytes is None:
        buf = io.BytesIO()
        image.save(buf, format=image.format or "PNG")
        size_bytes = buf.tell()

    if size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{size_bytes / 1024:.1f} KB"

    return {
        "resolution": f"{image.width} × {image.height} px",
        "size": size_str,
        "format": (image.format or getattr(uploaded_file, "type", "Unknown")).upper(),
    }


# ------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------ #
def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_ms(value_ms: float) -> str:
    if value_ms < 1000:
        return f"{value_ms:.0f} ms"
    return f"{value_ms / 1000:.2f} s"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_patient_id() -> str:
    return "PT-" + datetime.now().strftime("%Y%m%d%H%M%S")


# ------------------------------------------------------------------ #
# Session state defaults
# ------------------------------------------------------------------ #
def init_session_state(st) -> None:
    """Initialize all session_state keys used across the app, if absent."""
    defaults = {
        "patient_info": None,
        "uploaded_image": None,
        "uploaded_file_meta": None,
        "prediction_result": None,
        "gradcam_result": None,
        "history": [],
        "dark_mode": False,
        "model": None,
        "model_error": None,
        "stage": "patient_info",  # patient_info -> upload -> preprocess -> predict -> report
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app(st) -> None:
    """Reset all workflow state but keep dark_mode and history."""
    keep_history = st.session_state.get("history", [])
    keep_dark = st.session_state.get("dark_mode", False)
    for key in [
        "patient_info", "uploaded_image", "uploaded_file_meta",
        "prediction_result", "gradcam_result",
    ]:
        st.session_state[key] = None
    st.session_state["stage"] = "patient_info"
    st.session_state["history"] = keep_history
    st.session_state["dark_mode"] = keep_dark
