"""Evaluate a frozen checkpoint on the untouched test split and save per-image results."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from common import CLASSES, read_manifest, classification_metrics
from train import Images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    rows = [r for r in read_manifest(args.data) if r["split"] == "test"]
    if not rows: raise ValueError("No test split")
    wrapper = YOLO(args.weights); model = wrapper.model.float().cpu().eval()
    if [wrapper.names[i] for i in range(len(CLASSES))] != CLASSES: raise ValueError("Checkpoint class order mismatch")
    loader = DataLoader(Images(args.data, rows), batch_size=args.batch, shuffle=False, num_workers=0)
    probabilities = []; truth = []; start = time.monotonic()
    with torch.inference_mode():
        for images, labels in loader:
            result = model(images); result = result[0] if isinstance(result, tuple) else result
            probabilities.append(result.numpy()); truth.append(labels.numpy())
    probabilities = np.concatenate(probabilities); truth = np.concatenate(truth); predicted = probabilities.argmax(1)
    ordered = np.sort(probabilities, axis=1); confident = (ordered[:, -1] >= .8) & (ordered[:, -1] - ordered[:, -2] >= .15)
    report = classification_metrics(truth, predicted)
    report.update({"split": "test", "evaluation_seconds": time.monotonic() - start,
                   "threshold": .8, "margin": .15, "thresholds_calibrated": False,
                   "confident_coverage": float(confident.mean()),
                   "confident_accuracy": float((truth[confident] == predicted[confident]).mean()) if confident.any() else None})
    report["by_source"] = {}
    for source in sorted({r["source"] for r in rows}):
        mask = np.array([r["source"] == source for r in rows])
        report["by_source"][source] = classification_metrics(truth[mask], predicted[mask])
    # Cluster bootstrap uncertainty respects the prepared split groups.
    groups = sorted({r["group"] for r in rows}); group_array = np.array([r["group"] for r in rows])
    correct = truth == predicted
    group_correct = np.array([correct[group_array == g].sum() for g in groups])
    group_sizes = np.array([(group_array == g).sum() for g in groups])
    rng = np.random.default_rng(42); bootstrap = []
    for _ in range(2000):
        selected = rng.integers(0, len(groups), size=len(groups))
        bootstrap.append(float(group_correct[selected].sum() / group_sizes[selected].sum()))
    report["accuracy_95pct_group_bootstrap_interval"] = np.quantile(bootstrap, [.025, .975]).tolist()
    report["test_group_count"] = len(groups)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2))
    with (output / "test_predictions.csv").open("w", newline="") as file:
        fields = ["path", "source", "group", "true_label", "predicted_label", "confidence", "decision"] + ["p_"+n for n in CLASSES]
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"path": row["path"], "source": row["source"], "group": row["group"],
                "true_label": CLASSES[truth[i]], "predicted_label": CLASSES[predicted[i]],
                "confidence": float(probabilities[i].max()), "decision": CLASSES[predicted[i]] if confident[i] else "uncertain",
                **{"p_"+n: float(probabilities[i, k]) for k, n in enumerate(CLASSES)}})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
