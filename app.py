"""
app.py
------
Brain Tumor MRI Detection System — main Streamlit application.

A premium medical-AI interface for classifying brain MRI scans into
Glioma / Meningioma / Pituitary Tumor / No Tumor, with Grad-CAM
explainability, a patient information workflow, and a downloadable
medical report.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import random
from datetime import date

import streamlit as st
from PIL import Image

import gradcam
import predict
import report
import utils

# ------------------------------------------------------------------ #
# Page configuration
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Brain Tumor MRI Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_brain_tumor_model.h5"
MODEL_ACCURACY = None  # e.g. "96.4%" — set once you have a saved metrics value
EXAMPLES_DIR = "assets/examples"  # optional: put sample MRI images here for the "Random Example" button


def load_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")
utils.init_session_state(st)

def apply_theme():
    if st.session_state["dark_mode"]:
        st.markdown("""
        <style>
        .stApp {
            background: #0F172A;
            color: #E2E8F0;
        }

        .app-header h1,
        .app-header p,
        h1,h2,h3,h4,h5,h6,
        p,label,span {
            color: #E2E8F0 !important;
        }

        .card,
        .card-soft,
        div[data-testid="stMetric"] {
            background: #1E293B !important;
            color: #E2E8F0 !important;
            border-color: #334155 !important;
        }

        section[data-testid="stSidebar"] {
            background: #111827 !important;
            border-right: 1px solid #334155;
        }

        .status-success {
            background: #064E3B;
            border-color: #10B981;
        }

        .status-warning {
            background: #7F1D1D;
            border-color: #DC2626;
        }

        .status-info {
            background: #1E3A8A;
            border-color: #3B82F6;
        }
        </style>
        """, unsafe_allow_html=True)

utils.init_session_state(st)
apply_theme()

# ------------------------------------------------------------------ #
# Cached model loading
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    return predict.load_model(model_path)


if st.session_state["model"] is None and st.session_state["model_error"] is None:
    try:
        with st.spinner("Loading CNN model..."):
            st.session_state["model"] = get_model(MODEL_PATH)
    except predict.ModelLoadError as e:
        st.session_state["model_error"] = str(e)


# ================================================================== #
# SIDEBAR
# ================================================================== #
with st.sidebar:
    st.markdown("### 🧠 NeuroScan AI")
    st.caption("Brain Tumor MRI Detection System")
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["🏠 Home / Diagnosis", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### Application Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Reset", use_container_width=True):
            utils.reset_app(st)
            st.rerun()
    with col_b:
        dark = st.toggle("🌙 Dark", value=st.session_state["dark_mode"])
        if dark != st.session_state["dark_mode"]:
            st.session_state["dark_mode"] = dark
            st.rerun()

    if os.path.isdir(EXAMPLES_DIR) and any(os.scandir(EXAMPLES_DIR)):
        if st.button("🎲 Random MRI Example", use_container_width=True):
            files = [f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if files:
                choice = random.choice(files)
                path = os.path.join(EXAMPLES_DIR, choice)
                st.session_state["uploaded_image"] = Image.open(path)
                st.session_state["uploaded_file_meta"] = {
                    "resolution": None, "size": f"{os.path.getsize(path)/1024:.1f} KB", "format": choice.split('.')[-1].upper()
                }
                st.session_state["stage"] = "preprocess"
                st.rerun()

    st.markdown("---")
    st.markdown("#### Brain Tumor Classes")
    for key in ["glioma", "meningioma", "pituitary", "notumor"]:
        info = utils.get_tumor_info(key)
        st.markdown(f"{info['icon']} **{info['medical_name']}**")

    st.markdown("---")
    with st.expander("📌 About the Project"):
        st.write(
            "NeuroScan AI is an educational, AI-powered diagnostic-support "
            "platform that classifies brain MRI scans using a convolutional "
            "neural network, with Grad-CAM explainability so clinicians and "
            "students can see *why* the model made a given prediction."
        )

        st.markdown("---")
        st.markdown("#### 🧬 Model Information")
        st.write(f"**Architecture:** {report.MODEL_ARCHITECTURE}")
        st.write(f"**Input size:** {predict.IMG_WIDTH}×{predict.IMG_HEIGHT} px")
        st.write(f"**Classes:** {report.NUM_CLASSES}")
        if MODEL_ACCURACY:
            st.write(f"**Test accuracy:** {MODEL_ACCURACY}")

        st.markdown("---")
            
        st.markdown("#### 📁 Dataset Information")
    

        st.write(
            "Trained on the **Brain Tumor MRI Dataset** (Kaggle, "
            "masoudnickparvar/brain-tumor-mri-dataset), containing labeled "
            "T1-weighted MRI scans across four classes: glioma, meningioma, "
            "pituitary tumor, and no tumor."
        )

        st.markdown("---")
    
        st.markdown("#### 📖 Instructions")
        st.markdown(
            "1. Fill in patient information\n"
            "2. Upload an MRI scan (PNG/JPG/JPEG)\n"
            "3. Review the preprocessing preview\n"
            "4. Click **Analyze MRI**\n"
            "5. Review prediction, Grad-CAM, and the generated report\n"
            "6. Download the report if needed"
        )

        st.markdown("---")
    
        st.markdown("👥 Team Members")
        st.write("(Sara - Data Preprocessing and Augmentation)")
        st.write("John — Data pipeline & CNN Architecture")
        st.write("Eman — Prediction & Grad-CAM explainability")
        st.write("(Seif - Streamlit GUI)")
        
        st.markdown("✉️ Contact")
        st.write("For questions about this project, contact your project supervisor or team lead.")

    st.markdown("---")
    st.caption("Version 1.0.0")


# ================================================================== #
# HEADER
# ================================================================== #
st.markdown(
    """
    <div class="app-header">
        <h1>🧠 Brain Tumor MRI Detection System</h1>
        <p>AI-Powered MRI Classification with Explainable Artificial Intelligence (Grad-CAM)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state["model_error"]:
    st.markdown(
        f"""<div class="status-warning">⚠️ <b>Model could not be loaded.</b><br>{st.session_state['model_error']}
        <br><br>You can still fill in patient information and upload an image, but analysis will be unavailable
        until <code>{MODEL_PATH}</code> is present.</div>""",
        unsafe_allow_html=True,
    )

if nav == "ℹ️ About":

    st.markdown("""
    <div class="card">
        <h2>🧠 About NeuroScan AI</h2>
        <p>
        NeuroScan AI is an educational AI-powered clinical decision support
        system that classifies brain MRI scans into four categories using a
        Convolutional Neural Network (CNN) and provides Grad-CAM
        explainability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧬 Model Information")
    st.markdown(f"""
    <div class="card">
        <b>Architecture:</b> {report.MODEL_ARCHITECTURE}<br>
        <b>Input Size:</b> {predict.IMG_WIDTH} × {predict.IMG_HEIGHT}<br>
        <b>Number of Classes:</b> {report.NUM_CLASSES}<br>
        <b>Test Accuracy:</b> {MODEL_ACCURACY or "N/A"}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📁 Dataset")
    st.markdown("""
    <div class="card">
        Trained on the Brain Tumor MRI Dataset from Kaggle
        (masoudnickparvar/brain-tumor-mri-dataset), containing
        labeled T1-weighted MRI images for:
        <ul>
            <li>Glioma</li>
            <li>Meningioma</li>
            <li>Pituitary Tumor</li>
            <li>No Tumor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📖 Instructions")
    st.markdown("""
    <div class="card">
    <ol>
        <li>Fill in patient information.</li>
        <li>Upload an MRI image.</li>
        <li>Review preprocessing.</li>
        <li>Click <b>Analyze MRI</b>.</li>
        <li>Review the prediction and Grad-CAM.</li>
        <li>Download the report.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👥 Team")
    st.markdown("""
    <div class="card">
    • John — Data Pipeline & CNN Architecture<br>
    • Eman — Prediction & Grad-CAM<br>
    • Seif — Streamlit GUI<br>
    • Sara — Data Preprocessing
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✉️ Contact")
    st.markdown("""
    <div class="card">
    For questions about this project, contact your project supervisor
    or team leader.
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ================================================================== #
# DIAGNOSIS TIMELINE (progress indicator)
# ================================================================== #
STAGES = ["patient_info", "upload", "preprocess", "predict", "report"]
STAGE_LABELS = {
    "patient_info": "Patient Information",
    "upload": "MRI Uploaded",
    "preprocess": "Image Preprocessing",
    "predict": "CNN Prediction",
    "report": "Medical Report Created",
}


def render_timeline():
    current_index = STAGES.index(st.session_state["stage"]) if st.session_state["stage"] in STAGES else 0
    parts = []
    for i, s in enumerate(STAGES):
        done = i <= current_index
        mark = "✔" if done else "○"
        color = "var(--success)" if done else "var(--text-muted)"
        parts.append(f'<span style="color:{color}; font-weight:700;">{mark} {STAGE_LABELS[s]}</span>')
    st.markdown(
        '<div class="card-soft">' + "&nbsp;&nbsp;→&nbsp;&nbsp;".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


render_timeline()

# ================================================================== #
# STEP 1 — PATIENT INFORMATION
# ================================================================== #
st.markdown("## 👤 Patient Information")
with st.container():
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_id = st.text_input("Patient ID", value=utils.new_patient_id())
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            phone = st.text_input("Phone Number (Optional)")
        with c2:
            full_name = st.text_input("Full Name")
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            scan_date = st.date_input("MRI Scan Date", value=date.today())
            doctor = st.text_input("Doctor Name")
        with c3:
            height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
            hospital = st.text_input("Hospital Name")
        notes = st.text_area("Additional Notes", height=80)

        submitted = st.form_submit_button("💾 Save Patient Information", use_container_width=True)
        if submitted:
            st.session_state["patient_info"] = {
                "patient_id": patient_id, "full_name": full_name or "—", "age": age,
                "gender": gender, "height": height, "weight": weight,
                "blood_group": blood_group, "scan_date": scan_date, "hospital": hospital or "—",
                "doctor": doctor or "—", "phone": phone or "—", "notes": notes,
            }
            if st.session_state["stage"] == "patient_info":
                st.session_state["stage"] = "upload"
            st.success("Patient information saved.")

if not st.session_state["patient_info"]:
    st.info("Please save patient information to continue.")
    st.stop()


# ================================================================== #
# STEP 2 — MRI IMAGE UPLOAD
# ================================================================== #
st.markdown("## 📤 MRI Image Upload")
uploaded_file = st.file_uploader(
    "Drag and drop an MRI scan here",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()
        st.session_state["uploaded_image"] = image
        st.session_state["uploaded_file_meta"] = utils.get_image_metadata(uploaded_file, image)
        if st.session_state["stage"] in ("patient_info", "upload"):
            st.session_state["stage"] = "preprocess"
    except Exception:
        st.markdown(
            '<div class="status-warning">⚠️ Could not read this file. Please upload a valid PNG/JPG/JPEG image.</div>',
            unsafe_allow_html=True,
        )
        st.session_state["uploaded_image"] = None

if st.session_state["uploaded_image"] is None:
    st.info("Upload an MRI scan to continue, or use **Random MRI Example** in the sidebar.")
    st.stop()

image = st.session_state["uploaded_image"]
meta = st.session_state["uploaded_file_meta"] or {}

col_img, col_meta = st.columns([2, 1])
with col_img:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col_meta:
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown("**Image Details**")
    st.write(f"Resolution: {meta.get('resolution', f'{image.width} × {image.height} px')}")
    st.write(f"File Size: {meta.get('size', '—')}")
    st.write(f"Format: {meta.get('format', image.format or '—')}")
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================== #
# STEP 3 — PREPROCESSING PREVIEW
# ================================================================== #
st.markdown("## 🔬 Image Preprocessing Preview")
preprocessed_preview = image.convert("RGB").resize(predict.TARGET_SIZE)
p1, p_arrow, p2 = st.columns([2, 0.4, 2])
with p1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Original MRI", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with p_arrow:
    st.markdown("<h2 style='text-align:center; margin-top:3rem;'>→</h2>", unsafe_allow_html=True)
with p2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(preprocessed_preview, caption="Preprocessed MRI", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.caption(
    f"Preprocessing applied: resize to {predict.IMG_WIDTH}×{predict.IMG_HEIGHT} px, "
    f"RGB conversion, and pixel normalization to the [0, 1] range — matching the "
    f"transformations used during model training."
)


# ================================================================== #
# STEP 4 — PREDICTION
# ================================================================== #
st.markdown("## 🧪 AI Analysis")
st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
analyze_clicked = st.button("🔍 Analyze MRI", use_container_width=True, disabled=st.session_state["model"] is None)
st.markdown("</div>", unsafe_allow_html=True)

if analyze_clicked:
    if st.session_state["model"] is None:
        st.markdown(
            '<div class="status-warning">⚠️ Model is not loaded. Cannot run analysis.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Analyzing MRI scan — running CNN inference..."):
                result = predict.predict_image(st.session_state["model"], image)
            st.session_state["prediction_result"] = result

            try:
                with st.spinner("Generating Grad-CAM explanation..."):
                    original_img, heatmap_img, overlay_img, heatmap = gradcam.generate_gradcam(
                        st.session_state["model"], image, result.preprocessed_array,
                        target_size=predict.TARGET_SIZE,
                    )
                st.session_state["gradcam_result"] = {
                    "original": original_img, "heatmap": heatmap_img, "overlay": overlay_img,
                }
            except gradcam.GradCAMError as e:
                st.session_state["gradcam_result"] = None
                st.markdown(f'<div class="status-warning">⚠️ Grad-CAM unavailable: {e}</div>', unsafe_allow_html=True)

            st.session_state["stage"] = "report"
            st.session_state["history"].append({
                "name": st.session_state["patient_info"]["full_name"],
                "prediction": result.predicted_label,
                "confidence": result.confidence,
                "datetime": utils.now_str(),
            })
        except Exception as e:
            st.markdown(f'<div class="status-warning">⚠️ Prediction failed: {e}</div>', unsafe_allow_html=True)

result = st.session_state["prediction_result"]

if result is None:
    st.stop()

render_timeline()


# ================================================================== #
# PATIENT SUMMARY CARD
# ================================================================== #
p = st.session_state["patient_info"]
st.markdown("## 🗂️ Patient Summary")
st.markdown(
    f"""
    <div class="card fade-in">
        <b>{p['full_name']}</b> &nbsp;|&nbsp; ID: {p['patient_id']} &nbsp;|&nbsp; {p['age']} yrs, {p['gender']}<br>
        Blood Group: {p['blood_group']} &nbsp;|&nbsp; Height: {p['height']} cm &nbsp;|&nbsp; Weight: {p['weight']} kg<br>
        Scan Date: {p['scan_date']} &nbsp;|&nbsp; Hospital: {p['hospital']} &nbsp;|&nbsp; Doctor: {p['doctor']}
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================== #
# PREDICTION RESULT CARD
# ================================================================== #
level = predict.confidence_level(result.confidence)
info = utils.get_tumor_info(result.predicted_class)

if result.predicted_class == "notumor":
    st.markdown(
        f"""<div class="status-success">
        <h3>✅ No Tumor Detected</h3>
        <p>Confidence: <b>{utils.fmt_pct(result.confidence)}</b> &nbsp;|&nbsp;
        Processing time: <b>{utils.fmt_ms(result.processing_time_ms)}</b> &nbsp;|&nbsp;
        Device: <b>{result.device}</b></p>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""<div class="status-warning">
        <h3>{info['icon']} {result.predicted_label} Detected</h3>
        <p>Confidence: <b>{utils.fmt_pct(result.confidence)}</b> &nbsp;|&nbsp;
        Processing time: <b>{utils.fmt_ms(result.processing_time_ms)}</b> &nbsp;|&nbsp;
        Device: <b>{result.device}</b></p>
        </div>""",
        unsafe_allow_html=True,
    )


# ================================================================== #
# CLASS PROBABILITIES CHART
# ================================================================== #
st.markdown("### 📊 Class Probabilities")
prob_cols = st.columns(1)[0]
sorted_probs = sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)
for cls, prob in sorted_probs:
    label = predict.DISPLAY_NAMES.get(cls, cls.title())
    is_pred = cls == result.predicted_class
    bar_color = "#2563EB" if is_pred else "#93C5FD"
    st.markdown(
        f"""
        <div style="margin-bottom:0.5rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
                <span>{'⭐ ' if is_pred else ''}<b>{label}</b></span><span>{utils.fmt_pct(prob)}</span>
            </div>
            <div style="background:#E5E7EB; border-radius:8px; height:12px; overflow:hidden;">
                <div style="width:{prob*100:.1f}%; background:{bar_color}; height:100%; border-radius:8px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================== #
# CONFIDENCE GAUGE
# ================================================================== #
st.markdown("### 🎯 Confidence Gauge")
gauge_color = utils.confidence_color(level)
pct = result.confidence * 100
circumference = 2 * 3.14159 * 54
offset = circumference * (1 - result.confidence)
gauge_svg = f"""
<svg width="160" height="160" viewBox="0 0 120 120">
    <circle cx="60" cy="60" r="54" fill="none" stroke="#E5E7EB" stroke-width="10"/>
    <circle cx="60" cy="60" r="54" fill="none" stroke="{gauge_color}" stroke-width="10"
        stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
        stroke-linecap="round" transform="rotate(-90 60 60)"/>
    <text x="60" y="56" text-anchor="middle" font-size="20" font-weight="700" fill="{gauge_color}">{pct:.0f}%</text>
    <text x="60" y="74" text-anchor="middle" font-size="11" fill="#6B7280">{level}</text>
</svg>
"""
st.markdown(f'<div style="text-align:center;">{gauge_svg}</div>', unsafe_allow_html=True)


# ================================================================== #
# GRAD-CAM
# ================================================================== #
gc = st.session_state["gradcam_result"]
if gc:
    st.markdown("### 🔥 Grad-CAM Explainability")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(gc["original"], caption="Original MRI", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(gc["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(gc["overlay"], caption="Overlay Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🤖 How did the AI make this prediction?"):
        st.markdown(
            f"""
            The CNN scans the MRI in a series of layers, each learning to detect
            increasingly complex patterns — starting from simple edges and
            textures, and building up to shapes resembling tumor tissue.

            **Grad-CAM** works backward from the model's final prediction to find
            which regions of the image most strongly pushed the model toward
            the **{result.predicted_label}** class. Warmer colors (red/yellow) in the
            heatmap mark regions the model relied on most; cooler colors (blue)
            mark regions that had little influence.

            The highlighted regions above are the areas the model "looked at"
            most closely when reaching this prediction — useful for sanity-checking
            that the model is focusing on plausible anatomy rather than image
            artifacts.
            """
        )


# ================================================================== #
# SMART TUMOR INFORMATION PANEL
# ================================================================== #
st.markdown("### 🩺 Tumor Information")
risk_badge_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[info["risk_level"]]
if result.predicted_class == "notumor":
    st.markdown(
        f"""<div class="status-success">
        <h4>{info['icon']} {info['medical_name']}</h4>
        <p>{info['description']}</p>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="card" style="border-left:6px solid {info['color']};">
            <h4>{info['icon']} {info['medical_name']} &nbsp; <span class="badge {risk_badge_class}">{info['risk_level']} Risk</span></h4>
            <p>{info['description']}</p>
            <p><b>Common Location:</b> {info['location']}</p>
            <p><b>Characteristics:</b> {info['characteristics']}</p>
            <p><b>Common Symptoms:</b> {info['symptoms']}</p>
            <p><b>Typical Treatment Options:</b> {info['treatment']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """<div class="status-info" style="font-size:0.85rem; margin-top:0.5rem;">
    ⚠️ This AI prediction is intended for educational purposes only and should not be
    considered a medical diagnosis. Always consult a qualified radiologist or
    healthcare professional.</div>""",
    unsafe_allow_html=True,
)


# ================================================================== #
# RESULT DASHBOARD
# ================================================================== #
st.markdown("### 📋 Result Dashboard")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Prediction", result.predicted_label)
d2.metric("Confidence", utils.fmt_pct(result.confidence))
d3.metric("Risk Level", info["risk_level"])
d4.metric("Processing Time", utils.fmt_ms(result.processing_time_ms))
d5.metric("Device", result.device)

e1, e2, e3, e4, e5 = st.columns(5)
e1.metric("Image Resolution", f"{image.width}×{image.height}")
e2.metric("Image Size", meta.get("size", "—") if meta else "—")
e3.metric("Model Accuracy", MODEL_ACCURACY or "N/A")
e4.metric("Architecture", "CNN (Sequential)")
e5.metric("Classes", str(report.NUM_CLASSES))


# ================================================================== #
# COMPARE IMAGES
# ================================================================== #
st.markdown("### 🖼️ Compare Images")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Original MRI", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(preprocessed_preview, caption="Preprocessed MRI", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if gc:
        st.image(gc["overlay"], caption="Grad-CAM Overlay", use_container_width=True)
    else:
        st.info("Grad-CAM overlay unavailable.")
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================== #
# MEDICAL REPORT + DOWNLOADS
# ================================================================== #
st.markdown("### 📄 Medical Report")
report_text = report.build_report_text(
    patient=p,
    prediction_label=result.predicted_label,
    confidence=result.confidence,
    risk_level=info["risk_level"],
    model_accuracy=MODEL_ACCURACY,
)
st.markdown(f'<div class="card"><pre style="white-space:pre-wrap; font-family:monospace;">{report_text}</pre></div>', unsafe_allow_html=True)

dl1, dl2, dl3 = st.columns(3)
with dl1:
    pdf_bytes = report.build_report_pdf(
        patient=p, prediction_label=result.predicted_label, confidence=result.confidence,
        risk_level=info["risk_level"], overlay_image=gc["overlay"] if gc else None,
        model_accuracy=MODEL_ACCURACY,
    )
    if pdf_bytes:
        st.download_button("⬇️ Download Report (PDF)", data=pdf_bytes,
                            file_name=f"{p['patient_id']}_report.pdf", mime="application/pdf",
                            use_container_width=True)
    else:
        st.download_button("⬇️ Download Report (TXT)", data=report_text,
                            file_name=f"{p['patient_id']}_report.txt", mime="text/plain",
                            use_container_width=True)
with dl2:
    st.download_button("⬇️ Download Prediction Result (JSON)",
                        data=str({"prediction": result.predicted_label,
                                  "confidence": result.confidence,
                                  "probabilities": result.probabilities}),
                        file_name=f"{p['patient_id']}_prediction.json", mime="application/json",
                        use_container_width=True)
with dl3:
    if gc:
        import io as _io
        buf = _io.BytesIO()
        gc["overlay"].save(buf, format="PNG")
        st.download_button("⬇️ Download Grad-CAM Image", data=buf.getvalue(),
                            file_name=f"{p['patient_id']}_gradcam.png", mime="image/png",
                            use_container_width=True)
    else:
        st.button("⬇️ Grad-CAM Image (unavailable)", disabled=True, use_container_width=True)


# ================================================================== #
# SESSION HISTORY
# ================================================================== #
if st.session_state["history"]:
    st.markdown("### 🕘 Session History")
    st.dataframe(
        [{"Patient Name": h["name"], "Prediction": h["prediction"],
          "Confidence": utils.fmt_pct(h["confidence"]), "Date & Time": h["datetime"]}
         for h in reversed(st.session_state["history"])],
        use_container_width=True, hide_index=True,
    )

