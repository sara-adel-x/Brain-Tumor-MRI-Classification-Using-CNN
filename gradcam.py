"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation used
to explain the CNN's predictions by highlighting the MRI regions that most
influenced the predicted class.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM generation fails (e.g. no Conv2D layer found)."""


def get_last_conv_layer_name(model) -> str:
    """Return the name of the last Conv2D layer in the model (Grad-CAM target)."""
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise GradCAMError("No Conv2D layer found in the model — Grad-CAM is unavailable.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a Grad-CAM heatmap for a single preprocessed image.

    The model is split into two sub-models: (1) everything up to and
    including the last conv layer, and (2) the classifier head that follows.
    This two-model split is more robust than reusing `model.output` directly
    for models reloaded from disk via load_model(), where the original
    output tensor can otherwise become disconnected from the gradient tape.

    Args:
        img_array: preprocessed image, shape (1, H, W, 3)
        model: trained Keras model
        last_conv_layer_name: name of the conv layer to explain (auto-detected if None)
        pred_index: class index to explain (defaults to the model's top prediction)

    Returns:
        heatmap: 2D numpy array normalized to [0, 1]
        preds: the model's predicted probabilities for the image
    """
    import tensorflow as tf

    try:
        if last_conv_layer_name is None:
            last_conv_layer_name = get_last_conv_layer_name(model)

        last_conv_layer = model.get_layer(last_conv_layer_name)

        last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

        layer_names = [l.name for l in model.layers]
        last_conv_index = layer_names.index(last_conv_layer_name)

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer in model.layers[last_conv_index + 1:]:
            x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            conv_outputs = last_conv_layer_model(img_array)
            tape.watch(conv_outputs)
            predictions = classifier_model(conv_outputs)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            raise GradCAMError("Gradient computation failed for Grad-CAM.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), predictions.numpy()[0]

    except GradCAMError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise GradCAMError(f"Grad-CAM generation failed: {exc}") from exc


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    target_size: Tuple[int, int] = (150, 150),
    alpha: float = 0.4,
) -> Tuple[Image.Image, Image.Image]:
    """
    Overlay a Grad-CAM heatmap on top of the original image.

    Returns:
        original_img: the resized original image (RGB)
        overlayed_img: the heatmap blended over the original image
    """
    import matplotlib.pyplot as plt

    original_img = original_image.convert("RGB").resize(target_size)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(target_size)
    heatmap_resized = np.array(heatmap_img)

    colormap = plt.colormaps.get_cmap("jet")
    colored_heatmap = colormap(heatmap_resized / 255.0)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    original_array = np.array(original_img)
    overlayed_array = np.uint8(colored_heatmap * alpha + original_array * (1 - alpha))
    overlayed_img = Image.fromarray(overlayed_array)

    return original_img, overlayed_img


def heatmap_to_image(heatmap: np.ndarray, target_size: Tuple[int, int] = (150, 150)) -> Image.Image:
    """Render the raw heatmap alone (jet colormap) as a displayable PIL image."""
    import matplotlib.pyplot as plt

    colormap = plt.colormaps.get_cmap("jet")
    colored = colormap(heatmap)[:, :, :3]
    colored = np.uint8(colored * 255)
    img = Image.fromarray(colored).resize(target_size)
    return img


def generate_gradcam(model, image: Image.Image, img_array: np.ndarray, target_size=(150, 150)):
    """
    Convenience wrapper: run Grad-CAM end to end for a given preprocessed
    image array and return (original_img, heatmap_img, overlayed_img, heatmap).
    Raises GradCAMError on failure so the caller can show an elegant warning.
    """
    heatmap, _ = make_gradcam_heatmap(img_array, model)
    original_img, overlayed_img = overlay_heatmap(image, heatmap, target_size=target_size)
    heatmap_img = heatmap_to_image(heatmap, target_size=target_size)
    return original_img, heatmap_img, overlayed_img, heatmap
