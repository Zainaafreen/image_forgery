# detector/utils.py
import os
import numpy as np
import logging
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D, InputLayer as BaseInputLayer

logger = logging.getLogger(__name__)

class CustomDepthwiseConv2D(BaseDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

class CustomInputLayer(BaseInputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(*args, **kwargs)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_project.h5")

_model = None

def load_custom_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found → {MODEL_PATH}")
        logger.debug(f"Loading model: {MODEL_PATH}")
        _model = load_model(
            MODEL_PATH,
            custom_objects={
                'DepthwiseConv2D': CustomDepthwiseConv2D,
                'InputLayer': CustomInputLayer,
            },
            compile=False
        )
        logger.debug("Model loaded OK")
    return _model

def predict_image(img_path):
    model = load_custom_model()

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]  # [p0, p1]

    confidence = {
        "Fake": round(float(preds[0]) * 100, 2),
        "Real": round(float(preds[1]) * 100, 2)
    }

    if confidence["Real"] > confidence["Fake"]:
        result = "Real"
        conf = confidence["Real"]
    else:
        result = "Fake"
        conf = confidence["Fake"]

    logger.debug("\n" + "="*50)
    logger.debug("Image: %s", os.path.basename(img_path))
    logger.debug("Raw probs [Fake, Real]: %s", [confidence["Fake"], confidence["Real"]])
    logger.debug("Predicted: %s (%.2f%%)", result, conf)
    logger.debug("="*50 + "\n")

    return {
        "result": result,
        "confidence": conf,
        "details": confidence
    }