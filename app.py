import io
import json
import os

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.onnx")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.json")

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

CLASSES = META["classes"]
IMG_SIZE = int(META.get("image_size", 224))
CONF_THRESHOLD = float(META.get("default_confidence_threshold", 0.8))
MARGIN_THRESHOLD = float(META.get("default_margin_threshold", 0.15))

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)
INPUT_NAME = session.get_inputs()[0].name


def letterbox(image, size=224, fill=(114, 114, 114)):
    """Aspect-preserving resize followed by centered square padding."""
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions")

    scale = min(size / w, size / h)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), fill)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(image, (left, top))
    return canvas


def preprocess(image):
    image = ImageOps.exif_transpose(image)
    image = letterbox(image, IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def predict_image(image):
    tensor = preprocess(image)
    outputs = session.run(None, {INPUT_NAME: tensor})
    raw = np.asarray(outputs[0])

    # YOLOv8 classification exports normally return [batch, classes].
    scores = raw.reshape(raw.shape[0], -1)[0]

    # If the exported graph already returns probabilities, keep them;
    # otherwise convert logits to probabilities.
    if np.all(scores >= 0) and np.isclose(float(scores.sum()), 1.0, atol=1e-3):
        probs = scores
    else:
        probs = softmax(scores)

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    order = np.argsort(probs)[::-1]
    margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else 1.0

    predicted = CLASSES[idx] if idx < len(CLASSES) else str(idx)
    accepted = confidence >= CONF_THRESHOLD and margin >= MARGIN_THRESHOLD

    return {
        "class": predicted if accepted else "uncertain",
        "predicted_class": predicted,
        "confidence": round(confidence, 6),
        "margin": round(margin, 6),
        "accepted": bool(accepted),
        "thresholds": {
            "confidence": CONF_THRESHOLD,
            "margin": MARGIN_THRESHOLD,
        },
        "probabilities": {
            CLASSES[i] if i < len(CLASSES) else str(i): round(float(probs[i]), 6)
            for i in range(len(probs))
        },
    }


app = Flask(__name__)
CORS(app)


@app.get("/")
def root():
    return jsonify({
        "service": "GreenArk YOLOv8 Waste Classification API",
        "status": "online",
        "model": "YOLOv8n-cls ONNX",
        "classes": CLASSES,
        "input_size": IMG_SIZE,
        "endpoints": ["/health", "/predict"],
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model": "best.onnx",
        "classes": CLASSES,
        "input_size": IMG_SIZE,
    })


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image supplied. Use multipart/form-data with field name 'file'."
        }), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename."}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        result = predict_image(image)
        return jsonify({"success": True, **result})
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Inference failed: {type(exc).__name__}: {exc}"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
