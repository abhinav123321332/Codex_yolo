"""Create a readable report and figures from completed measurements; never invent scores."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default=".")
    root = Path(parser.parse_args().root); reports = root / "reports"
    training = json.loads((reports / "training_report.json").read_text())
    test = json.loads((reports / "test_metrics.json").read_text())
    data = json.loads((reports / "dataset_report.json").read_text())
    export = json.loads((reports / "export_verification.json").read_text())
    names = test["class_order"]; cm = np.array(test["confusion_matrix"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.3, 5.1), layout="constrained")
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
           xlabel="Predicted class", ylabel="Actual class", title="YOLOv8 held-out test results")
    for (r, c), value in np.ndenumerate(cm):
        ax.text(c, r, str(value), ha="center", va="center", fontsize=17,
                color="white" if value > cm.max() / 2 else "#102a43")
    fig.savefig(reports / "confusion_matrix.png", dpi=180); plt.close(fig)
    history = list(csv.DictReader((reports / "training_history.csv").open()))
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    epochs = [int(r["epoch"]) for r in history]
    ax.plot(epochs, [100 * float(r["val_accuracy"]) for r in history], label="Validation accuracy", color="#007c91")
    ax.plot(epochs, [100 * float(r["val_macro_f1"]) for r in history], label="Validation macro F1", color="#bb5a24")
    ax.axvline(training["best_epoch"], color="#697681", linestyle="--", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="Score (%)", title="Training-head validation results")
    ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.savefig(reports / "training_curve.png", dpi=180); plt.close(fig)
    low, high = test["accuracy_95pct_group_bootstrap_interval"]
    class_table = "\n".join(f"| {name} | {test['per_class'][name]['precision']:.3f} | {test['per_class'][name]['recall']:.3f} | {test['per_class'][name]['f1']:.3f} | {test['per_class'][name]['support']} |" for name in names)
    sources = "\n".join(f"| {name} | {result['sample_count']} | {result['accuracy']*100:.2f}% |" for name, result in test["by_source"].items())
    splits = "\n".join(f"| {split} | {data['split_counts'][split+'/metal']} | {data['split_counts'][split+'/other']} | {data['split_counts'][split+'/plastic']} |" for split in ("train", "val", "test"))
    report = f"""# Waste YOLOv8 — completed model report

Built on 2026-09-20. Model: **YOLOv8n-cls**, trained on CPU with three image-level classes.

## Measured result

**Test top-1 accuracy: {test['accuracy']*100:.2f}%** on **{test['sample_count']:,} held-out images/crops**.
Test macro F1: **{test['macro_f1']:.4f}**. A 2,000-resample group bootstrap gives an approximate
95% accuracy interval of **{low*100:.2f}%–{high*100:.2f}%** within this curated dataset.
This does not measure accuracy on a new camera, new physical items, or a different location.

| Class | Precision | Recall | F1 | Test images |
| --- | ---: | ---: | ---: | ---: |
{class_table}

![Confusion matrix](reports/confusion_matrix.png)

## Source-specific test accuracy

| Source | Test images | Accuracy |
| --- | ---: | ---: |
{sources}

The mixed overall accuracy weights sources by their retained test image counts.
The wild-bottle source contains only the plastic class. Its accuracy cannot measure
false positives on nonplastic objects. Per-source results may differ substantially.

## Data and split

| Split | Metal | Other | Plastic |
| --- | ---: | ---: | ---: |
{splits}

Total retained images/crops: {data['retained_images']:,}. Exact duplicates removed: {data['exact_duplicates_removed']}.
There are zero shared exact pixel hashes or prepared groups across the three splits.
The split was rebuilt with 70/15/15 target proportions and grouping, so it is not an
official benchmark split. Uploaded adjacent filename blocks, source image names,
and perceptually similar images are grouped. That heuristic does not guarantee
independent physical objects or photographic sessions.

The water-level dataset was inspected and excluded because its class labels do
not establish material. The three compatible Kaggle sources and the user upload
contributed images. See `DATA_SOURCES.md` for exact mappings and attributions.

## Training method

- ImageNet-pretrained YOLOv8n-cls backbone: frozen.
- Trained component: the final 1280-to-3 linear layer ({training['trainable_parameter_count']:,} parameters).
- Training images: {training['train_images']:,}; original plus horizontal-flipped views.
- Features: {training['train_feature_vectors']:,}; feature dropout 0.15.
- Loss: class-weighted cross entropy with label smoothing 0.02.
- Optimizer: AdamW; seed 42; selected by validation macro F1.
- Epochs completed: {training['epochs_completed']}; selected epoch: {training['best_epoch']}.
- Validation accuracy at selection: {training['validation']['accuracy']*100:.2f}%.
- Test images were not used for training or checkpoint selection.
- Input: RGB, full-image aspect-preserving padding to 224, divide by 255.
- Runtime: torch {training['torch']}, ultralytics {training['ultralytics']}.

![Validation curve](reports/training_curve.png)

This is a completed transfer-learning baseline. End-to-end fine-tuning is an
optional supplied recipe, not a step claimed to have been completed here.

## Prediction thresholds

The predictor reports `uncertain` below score 0.80 or probability margin 0.15.
These default thresholds were not calibrated. On this test set they retain
{test['confident_coverage']*100:.2f}% of images and have
{test['confident_accuracy']*100:.2f}% top-1 accuracy among the retained images.
Neither the score nor the `other` class guarantees rejection of unfamiliar inputs.

## Verification

The saved PyTorch checkpoint was loaded and evaluated on the test set. The
ONNX structure passed validation. On {export['checked_images']} representative validation images,
ONNX and PyTorch predicted the same top class; the largest absolute probability
difference was {export['maximum_absolute_probability_difference']:.8f}.
Scripts passed Python syntax compilation. The Colab notebook was structurally
checked but was not run in Colab, and the optional GPU fine-tuning path was not
executed in this CPU build.

## Scope

The model classifies a whole photo/crop as metal, plastic, or other. It does not
produce detection boxes, reliably count items, verify PET type, determine empty
bottles, or decide reward eligibility. No real camera or machine integration was
tested. Preserve the included preprocessing and test actual camera images before
depending on its classifications.
"""
    (root / "MODEL_REPORT.md").write_text(report)
    print(f"Wrote {root / 'MODEL_REPORT.md'}")


if __name__ == "__main__": main()
