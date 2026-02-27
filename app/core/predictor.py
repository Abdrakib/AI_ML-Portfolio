"""Inference pipeline for brain tumor classification."""
import io
import base64
import logging
from typing import Any

import numpy as np
from tensorflow import keras

from app.core.model_loader import get_model, get_label_maps
from app.core.gradcam import get_gradcam_overlay_from_bytes

logger = logging.getLogger(__name__)


def _preprocess_image(img_bytes: bytes) -> np.ndarray:
    """
    Load and preprocess image for model input.
    Match notebook: load_img -> img_to_array (float32, 0-255), resize 224x224.
    Model applies augmentation + preprocess_input internally.
    """
    img = keras.utils.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    arr = keras.utils.img_to_array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def predict_with_gradcam(img_bytes: bytes) -> dict[str, Any]:
    """
    Run inference and Grad-CAM overlay.
    Returns: pred_label, confidence, gradcam_overlay_b64
    """
    model = get_model()
    _, idx_to_label = get_label_maps()
    classes = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    num_classes = len(classes)

    # Preprocess for model (0-255 float32, 224x224)
    img_batch = _preprocess_image(img_bytes)

    # Prediction
    probs = model.predict(img_batch, verbose=0)

    if num_classes == 2:
        p1 = float(probs.reshape(-1)[0])
        pred_idx = 1 if p1 >= 0.5 else 0
        confidence = float(p1 if pred_idx == 1 else (1.0 - p1))
        probs_vec = np.array([1.0 - p1, p1])
    else:
        probs_vec = probs.reshape(-1)
        pred_idx = int(np.argmax(probs_vec))
        confidence = float(probs_vec[pred_idx])

    pred_label = classes[pred_idx]

    # Grad-CAM overlay (same preprocessing as notebook)
    overlay_bytes = get_gradcam_overlay_from_bytes(img_bytes, model, alpha=0.35, format="png")
    gradcam_b64 = base64.b64encode(overlay_bytes).decode("utf-8")

    return {
        "pred_label": pred_label,
        "confidence": round(confidence, 4),
        "probs": {classes[i]: float(probs_vec[i]) for i in range(num_classes)},
        "gradcam_overlay_b64": gradcam_b64,
    }


def get_gradcam_overlay_bytes(img_bytes: bytes) -> bytes:
    """Return Grad-CAM overlay image as PNG bytes (no prediction metadata)."""
    model = get_model()
    return get_gradcam_overlay_from_bytes(img_bytes, model, alpha=0.35, format="png")
