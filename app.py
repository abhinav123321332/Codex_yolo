```python
import io
import json
import os

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.onnx")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.json")


# ============================================================
# LOAD MODEL METADATA
# ============================================================

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

CLASSES = META["classes"]

IMG_SIZE = int(
    META.get("image_size", 224)
)

CONF_THRESHOLD = float(
    META.get(
        "default_confidence_threshold",
        0.8
    )
)

MARGIN_THRESHOLD = float(
    META.get(
        "default_margin_threshold",
        0.15
    )
)


# ============================================================
# LOAD ONNX MODEL
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def letterbox(
    image,
    size=224,
    fill=(114, 114, 114)
):

    image = image.convert("RGB")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid image dimensions"
        )

    scale = min(
        size / width,
        size / height
    )

    new_width = max(
        1,
        round(width * scale)
    )

    new_height = max(
        1,
        round(height * scale)
    )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (size, size),
        fill
    )

    left = (
        size - new_width
    ) // 2

    top = (
        size - new_height
    ) // 2

    canvas.paste(
        image,
        (left, top)
    )

    return canvas


def preprocess(image):

    image = ImageOps.exif_transpose(
        image
    )

    image = letterbox(
        image,
        IMG_SIZE
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = array / 255.0

    # HWC → CHW
    array = np.transpose(
        array,
        (2, 0, 1)
    )

    # Add batch dimension
    array = np.expand_dims(
        array,
        axis=0
    )

    return array.astype(
        np.float32
    )


# ============================================================
# SOFTMAX
# ============================================================

def softmax(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = (
        values
        - np.max(
            values,
            axis=-1,
            keepdims=True
        )
    )

    exponential = np.exp(
        values
    )

    return (
        exponential
        /
        np.sum(
            exponential,
            axis=-1,
            keepdims=True
        )
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def predict_image(image):

    tensor = preprocess(
        image
    )

    outputs = session.run(
        None,
        {
            INPUT_NAME: tensor
        }
    )

    raw_output = np.asarray(
        outputs[0]
    )

    scores = raw_output.reshape(
        raw_output.shape[0],
        -1
    )[0]

    # Check whether the ONNX output is
    # already normalized probabilities.
    if (
        np.all(scores >= 0)
        and np.isclose(
            float(scores.sum()),
            1.0,
            atol=1e-3
        )
    ):
        probabilities = scores

    else:
        probabilities = softmax(
            scores
        )

    # Highest probability class
    class_index = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[class_index]
    )

    # Sort probabilities
    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    # Confidence difference between
    # first and second class
    if len(sorted_indices) > 1:

        margin = float(
            probabilities[
                sorted_indices[0]
            ]
            -
            probabilities[
                sorted_indices[1]
            ]
        )

    else:

        margin = 1.0

    if class_index < len(CLASSES):

        predicted_class = (
            CLASSES[class_index]
        )

    else:

        predicted_class = str(
            class_index
        )

    accepted = (
        confidence >= CONF_THRESHOLD
        and
        margin >= MARGIN_THRESHOLD
    )

    final_class = (
        predicted_class
        if accepted
        else "uncertain"
    )

    probability_result = {}

    for index, probability in enumerate(
        probabilities
    ):

        if index < len(CLASSES):

            class_name = CLASSES[index]

        else:

            class_name = str(index)

        probability_result[
            class_name
        ] = round(
            float(probability),
            6
        )

    return {

        "class":
            final_class,

        "predicted_class":
            predicted_class,

        "confidence":
            round(
                confidence,
                6
            ),

        "margin":
            round(
                margin,
                6
            ),

        "accepted":
            bool(
                accepted
            ),

        "thresholds": {

            "confidence":
                CONF_THRESHOLD,

            "margin":
                MARGIN_THRESHOLD

        },

        "probabilities":
            probability_result

    }


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__
)

CORS(
    app
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return jsonify({

        "service":
            "GreenArk YOLOv8 Waste Classification API",

        "status":
            "online",

        "model":
            "YOLOv8n-cls ONNX",

        "classes":
            CLASSES,

        "input_size":
            IMG_SIZE,

        "endpoints": [

            "/",

            "/health",

            "/predict",

            "/api/upload"

        ]

    })


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "model_loaded":
            True,

        "model":
            "best.onnx",

        "classes":
            CLASSES,

        "input_size":
            IMG_SIZE

    })


# ============================================================
# READ UPLOADED IMAGE
# ============================================================

def get_uploaded_image():

    if "file" not in request.files:

        raise ValueError(
            "No image supplied. "
            "Send the image using "
            "multipart/form-data "
            "with field name 'file'."
        )

    uploaded_file = (
        request.files["file"]
    )

    if not uploaded_file.filename:

        raise ValueError(
            "Empty filename."
        )

    image_data = (
        uploaded_file.read()
    )

    if not image_data:

        raise ValueError(
            "Uploaded file is empty."
        )

    image = Image.open(
        io.BytesIO(
            image_data
        )
    )

    return image


# ============================================================
# /predict
# ============================================================

@app.post("/predict")
def predict():

    try:

        image = (
            get_uploaded_image()
        )

        result = (
            predict_image(
                image
            )
        )

        return jsonify({

            "success":
                True,

            **result

        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                (
                    "Inference failed: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        }), 500


# ============================================================
# /api/upload
#
# This endpoint is provided for your
# existing GreenArk frontend.
# ============================================================

@app.post("/api/upload")
def api_upload():

    try:

        image = (
            get_uploaded_image()
        )

        result = (
            predict_image(
                image
            )
        )

        return jsonify({

            "success":
                True,

            # Main classification
            "class":
                result["class"],

            "predicted_class":
                result[
                    "predicted_class"
                ],

            # Confidence
            "confidence":
                result[
                    "confidence"
                ],

            # Margin
            "margin":
                result[
                    "margin"
                ],

            # Whether prediction
            # passed thresholds
            "accepted":
                result[
                    "accepted"
                ],

            # All class probabilities
            "probabilities":
                result[
                    "probabilities"
                ],

            # Compatibility fields
            "prediction":
                result["class"],

            "label":
                result["class"],

            "category":
                result["class"]

        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                (
                    "Classification failed: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
```
