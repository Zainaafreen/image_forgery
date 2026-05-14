# detector/utils.py
import os
import numpy as np
import logging
import cv2
import urllib.request

import tensorflow as tf
from keras.models import load_model
from keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D

logger = logging.getLogger(__name__)


class CustomDepthwiseConv2D(BaseDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_project.h5")

_model = None


def load_custom_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found → {MODEL_PATH}")
        logger.info(f"Loading model: {MODEL_PATH}")
        _model = load_model(
            MODEL_PATH,
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
            compile=False
        )
        logger.info("Model loaded OK")
    return _model


def _read_image_from_url(url: str) -> np.ndarray:
    """
    Download image bytes directly into memory and decode with OpenCV.
    Avoids temp-file race conditions and file-locking issues entirely.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        raw_bytes = response.read()

    if not raw_bytes:
        raise ValueError(f"Downloaded 0 bytes from URL: {url}")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(
            f"cv2.imdecode failed — the downloaded content is not a valid image. "
            f"URL: {url} | Bytes received: {len(raw_bytes)}"
        )

    return img


def _read_image_from_path(path: str) -> np.ndarray:
    """
    Read image from a local filesystem path with clear error messages.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image file not found at path: {path}. "
            "On Railway, local files are lost on redeploy — use Cloudinary URLs instead."
        )

    img = cv2.imread(path)

    if img is None:
        raise ValueError(
            f"cv2.imread returned None for path: {path}. "
            "File may be corrupt, empty, or an unsupported format."
        )

    return img


def predict_image(img_path_or_url: str) -> dict:
    model = load_custom_model()

    logger.info("predict_image called with: %s", img_path_or_url)

    # Load image — URL (Cloudinary) or local path
    if img_path_or_url.startswith("http://") or img_path_or_url.startswith("https://"):
        img = _read_image_from_url(img_path_or_url)
    else:
        img = _read_image_from_path(img_path_or_url)

    # Preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]

    confidence = {
        "Fake": round(float(preds[0]) * 100, 2),
        "Real": round(float(preds[1]) * 100, 2),
    }

    if confidence["Real"] > confidence["Fake"]:
        result = "Real"
        conf = confidence["Real"]
    else:
        result = "Fake"
        conf = confidence["Fake"]

    logger.info(
        "Prediction complete | File: %s | Result: %s (%.2f%%) | Probs: %s",
        os.path.basename(img_path_or_url),
        result,
        conf,
        confidence,
    )

    return {
        "result": result,
        "confidence": conf,
        "details": confidence,
    }