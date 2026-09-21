"""Export and compare the ONNX probabilities against the PyTorch checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from ultralytics import YOLO
from common import image_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--images", nargs="+", required=True, help="Real image files used for numerical verification")
    parser.add_argument("--report", default="reports/export_verification.json")
    args = parser.parse_args(); torch.set_num_threads(4)
    model = YOLO(args.weights)
    exported = Path(model.export(format="onnx", imgsz=224, batch=1, dynamic=True, simplify=False, opset=17, device="cpu"))
    onnx.checker.check_model(onnx.load(exported))
    reference = YOLO(args.weights).model.float().cpu().eval()
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(exported), sess_options=options, providers=["CPUExecutionProvider"])
    arrays = []
    for path in args.images:
        with Image.open(path) as image: arrays.append(image_array(image))
    array = np.stack(arrays).astype(np.float32)
    with torch.inference_mode(): expected = reference(torch.from_numpy(array))[0].numpy()
    actual = session.run(None, {session.get_inputs()[0].name: array})[0]
    delta = float(np.max(np.abs(expected - actual)))
    matches = bool(np.array_equal(expected.argmax(1), actual.argmax(1)))
    if not np.allclose(expected, actual, atol=.0002, rtol=.0002) or not matches:
        raise RuntimeError(f"ONNX comparison failed: max delta={delta}, classes match={matches}")
    report = {"status": "passed", "checked_images": len(arrays), "maximum_absolute_probability_difference": delta,
              "top1_classes_match": matches, "onnx_structural_check": "passed", "opset": 17,
              "dynamic_batch": True, "input": "float32 RGB NCHW; 224x224 padding; divide by 255",
              "output": "softmax probabilities in metadata class order"}
    path = Path(args.report); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
