"""Classify one centered item using the same preprocessing as training."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from common import image_array


class WasteClassifier:
    def __init__(self, weights, confidence=.8, margin=.15, threads=4):
        weights = Path(weights)
        if not 0 <= confidence <= 1 or not 0 <= margin <= 1: raise ValueError("Thresholds must be in [0,1]")
        metadata = json.loads((weights.parent / "metadata.json").read_text())
        self.names = metadata["classes"]; self.size = metadata["image_size"]
        self.confidence, self.margin = confidence, margin
        self.backend = weights.suffix.lower()
        if self.backend == ".onnx":
            import onnxruntime as ort
            options = ort.SessionOptions(); options.intra_op_num_threads = threads
            self.session = ort.InferenceSession(str(weights), sess_options=options, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        elif self.backend == ".pt":
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(threads)
            wrapper = YOLO(str(weights)); self.model = wrapper.model.float().cpu().eval()
            if [wrapper.names[i] for i in range(len(wrapper.names))] != self.names:
                raise ValueError("Metadata classes do not match the checkpoint")
        else: raise ValueError("Use a .pt or .onnx model")

    def probabilities(self, image):
        array = np.ascontiguousarray(image_array(image, self.size)[None])
        if self.backend == ".onnx":
            return self.session.run(None, {self.input_name: array})[0][0]
        import torch
        with torch.inference_mode():
            result = self.model(torch.from_numpy(array))
        return (result[0] if isinstance(result, tuple) else result)[0].numpy()

    def predict(self, image):
        probabilities = self.probabilities(image); ranking = np.argsort(probabilities)[::-1]
        first, second = int(ranking[0]), int(ranking[1]); confidence = float(probabilities[first])
        margin = confidence - float(probabilities[second]); label = self.names[first]
        decision = label if confidence >= self.confidence and margin >= self.margin else "uncertain"
        return {"predicted_class": label, "confidence": confidence, "margin": margin,
                "decision": decision, "probabilities": {name: float(probabilities[i]) for i, name in enumerate(self.names)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "models/best.onnx"))
    parser.add_argument("--confidence", type=float, default=.8)
    parser.add_argument("--margin", type=float, default=.15)
    args = parser.parse_args()
    classifier = WasteClassifier(args.weights, args.confidence, args.margin)
    with Image.open(args.image) as image: print(json.dumps(classifier.predict(image), indent=2))


if __name__ == "__main__": main()
