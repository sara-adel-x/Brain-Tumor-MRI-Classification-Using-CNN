"""
predict.py
----------
Model loading and single-image inference for the Brain Tumor MRI
Classification System.

This module wraps the trained CNN (best_brain_tumor_model.h5) and exposes
a clean, typed API that the Streamlit UI (app.py) can call without ever
touching TensorFlow/Keras internals directly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from PIL import Image

# TensorFlow is imported lazily inside load_model() so that the rest of the
# app (page config, sidebar, forms, etc.) can render even in environments
# where TF is slow to import or the model file is missing.

# ------------------------------------------------------------------ #
# Constants — must match the training configuration in the notebook.
# ------------------------------------------------------------------ #
IMG_WIDTH: int = 150
IMG_HEIGHT: int = 150
TARGET_SIZE: Tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT)

# CLASS_NAMES = sorted(os.listdir(train_path)) in the notebook.
# The Kaggle "brain-tumor-mri-dataset" folder names sort to this order.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Human-friendly display labels used throughout the UI / reports.
DISPLAY_NAMES: Dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

DEFAULT_MODEL_PATH = "best_brain_tumor_model.h5"
MODEL_ACCURACY_PLACEHOLDER = None  # populated from a metrics file if present


@dataclass
class PredictionResult:
    """Structured result of a single-image prediction."""

    predicted_class: str            # raw class key, e.g. "glioma"
    predicted_label: str            # display label, e.g. "Glioma"
    confidence: float               # 0..1
    probabilities: Dict[str, float] # raw class key -> probability
    processing_time_ms: float
    device: str
    preprocessed_array: np.ndarray = field(repr=False)  # (1, H, W, 3), for Grad-CAM reuse


class ModelLoadError(RuntimeError):
    """Raised when the CNN model file cannot be found or loaded."""


def get_inference_device() -> str:
    """Return a human-readable description of the device TensorFlow will use."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return f"GPU ({gpus[0].name})" if gpus else "CPU"
    except Exception:
        return "CPU"


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained Keras model from disk.

    Raises:
        ModelLoadError: if the file is missing or fails to load, with a
        message suitable for display directly in the Streamlit UI.
    """
    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"Model file not found at '{model_path}'. Place the trained "
            f"'best_brain_tumor_model.h5' file in the project root, or "
            f"update MODEL_PATH."
        )
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as exc:  # pragma: no cover - defensive
        raise ModelLoadError(f"Failed to load model: {exc}") from exc


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Preprocess a PIL image exactly as done during training:
    convert to RGB, resize, rescale to [0, 1], add batch dimension.
    """
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype("float32")


def predict_image(model, image: Image.Image, class_names=CLASS_NAMES) -> PredictionResult:
    """
    Run inference on a single MRI image and return a structured result.
    """
    start = time.perf_counter()
    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    predicted_index = int(np.argmax(preds))
    predicted_class = class_names[predicted_index]
    confidence = float(preds[predicted_index])
    probabilities = {cls: float(p) for cls, p in zip(class_names, preds)}

    return PredictionResult(
        predicted_class=predicted_class,
        predicted_label=DISPLAY_NAMES.get(predicted_class, predicted_class.title()),
        confidence=confidence,
        probabilities=probabilities,
        processing_time_ms=elapsed_ms,
        device=get_inference_device(),
        preprocessed_array=img_array,
    )


def confidence_level(confidence: float) -> str:
    """Bucket a confidence score into 'High' / 'Medium' / 'Low' for the gauge."""
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.60:
        return "Medium"
    return "Low"
