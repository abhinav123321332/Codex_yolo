"""Shared, explicit RGB preprocessing for training and deployment."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps

CLASSES = ["metal", "other", "plastic"]
IMAGE_SIZE = 224


def square_rgb(image, size=IMAGE_SIZE):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                        color=(114, 114, 114), centering=(0.5, 0.5))


def image_array(image, size=IMAGE_SIZE):
    return np.asarray(square_rgb(image, size), dtype=np.float32).transpose(2, 0, 1) / 255.0


def read_manifest(data):
    return [json.loads(line) for line in (Path(data) / "manifest.jsonl").read_text().splitlines()]


def classification_metrics(truth, prediction, names=CLASSES):
    cm = np.zeros((len(names), len(names)), dtype=np.int64)
    for a, b in zip(truth, prediction):
        cm[int(a), int(b)] += 1
    rows = {}
    for k, name in enumerate(names):
        tp = int(cm[k, k]); support = int(cm[k].sum()); predicted = int(cm[:, k].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        rows[name] = {"precision": precision, "recall": recall,
                      "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                      "support": support}
    return {"accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0,
            "macro_f1": float(np.mean([r["f1"] for r in rows.values()])),
            "per_class": rows, "confusion_matrix": cm.tolist(),
            "class_order": names, "sample_count": int(cm.sum())}
