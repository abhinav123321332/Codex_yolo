# Waste YOLOv8 — trained material classifier

This package contains a **trained YOLOv8n-cls model** for `metal`, `other`, and
`plastic`. It assigns one label to a photo or item crop. It does not draw boxes,
count multiple objects, distinguish PET resin, or exclusively recognize cans
and bottles. See `MODEL_REPORT.md` for the actual measured results.

The supplied checkpoint uses ImageNet-pretrained YOLOv8n features with a newly
trained three-class final layer. This is a frozen-backbone transfer-learning
baseline. Optional end-to-end GPU fine-tuning code is included separately.

## Use the model now

Unzip this package, open a terminal in the `Waste_YOLOv8` folder, and install the
lightweight inference requirements using Python 3.10–3.12:

```bash
python -m pip install -r requirements-inference.txt
python scripts/predict.py path/to/photo.jpg
```

The default command uses `models/best.onnx`, works on a computer CPU, and prints
JSON containing the class, score, all class probabilities, and decision.
`uncertain` is returned when the score is below 0.80 or the top two probabilities
differ by less than 0.15. Those are starting thresholds, not calibrated certainty
or guaranteed rejection of unfamiliar objects.

For the PyTorch checkpoint:

```bash
python -m pip install -r requirements.txt
python scripts/predict.py path/to/photo.jpg --weights models/best.pt
```

The output class order is **0 = metal, 1 = other, 2 = plastic**. Read
`models/metadata.json` instead of guessing class numbers in another application.
Keep this metadata beside the model files.

## Important preprocessing

Use the supplied prediction function or reproduce these steps exactly:

1. Apply EXIF orientation and convert to RGB.
2. Preserve the aspect ratio and pad to 224 × 224 with RGB (114, 114, 114), using
   bilinear resizing and centered padding.
3. Convert to float32, divide by 255, and arrange NCHW (`1 × 3 × 224 × 224`).
4. Read the three output softmax probabilities in the metadata class order.

Do not feed a rectangular camera image directly through Ultralytics' default
center-crop preprocessing and assume it is identical to this package. The
supplied script preserves the full item. Use one centered item or an independently
detected item crop per prediction.

## Files

| File | Purpose |
| --- | --- |
| `models/best.pt` | Trained YOLOv8 classification checkpoint |
| `models/best.onnx` | Exported CPU inference model |
| `models/metadata.json` | Class order, input size, preprocessing, thresholds |
| `MODEL_REPORT.md` | Measured test results, method, and limitations |
| `reports/test_metrics.json` | Overall and per-source test measurements |
| `reports/test_predictions.csv` | Every test image's prediction |
| `reports/training_history.csv` | Validation results for every completed epoch |
| `reports/dataset_report.json` | Retained counts, exclusions, split details |
| `reports/manifest.jsonl` | Exact source/crop/group/split record for every image |
| `reports/export_verification.json` | Numerical comparison of PT and ONNX |
| `scripts/` | Data preparation, training, prediction, evaluation, export |
| `Waste_YOLOv8_Colab.ipynb` | Colab use and optional further training |

## Train again

The companion `Waste_YOLOv8_Prepared_Data.zip` contains the exact ready-to-use
`data/` folder. Extract it into this folder. It is only needed for training or
re-evaluation; ordinary prediction needs only this model package.

To reproduce the training method in a new output folder:

```bash
python scripts/train.py --data data --out retrained --weights yolov8n-cls.pt
python scripts/evaluate.py --data data --weights retrained/models/best.pt --out retrained/reports
```

The first run downloads the official pretrained YOLOv8n-cls checkpoint. CPU-only
PyTorch can be installed first to avoid downloading CUDA libraries on a computer
without a GPU:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For optional end-to-end GPU refinement:

```bash
python scripts/finetune.py --data data --weights models/best.pt --out finetuned --epochs 60 --device 0
python scripts/evaluate.py --data data --weights finetuned/models/best.pt --out finetuned/reports
```

The fine-tuning path and Colab notebook are provided for further work; the
shipped model's measured results refer only to the completed CPU training run.
Do not tune choices against the test scores. Add a new camera test set for later
experiments and retain all photos of the same physical item in one split.

## Rebuild the dataset from the original sources

```bash
python scripts/download_data.py --out downloads
python scripts/prepare_data.py --uploaded "Plastic and Cans.zip" --downloads downloads --out data
```

The downloader requests the exact inspected Kaggle versions. If Kaggle requires
sign-in in your environment, download those versions from the linked dataset
pages and save them under the names described in `download_data.py`.
Preparation requires a new/empty output folder and will not overwrite an
existing dataset.

The water-level dataset was downloaded and inspected but excluded because its
labels do not establish container material. See `DATA_SOURCES.md` for all source
links, versions, attributions, mappings, and transformations.

## Before using a camera

Evaluate photos from the actual camera, lighting, background, item distance,
and viewing angle. Include empty views, glass bottles, mixed materials, hands,
crushed items, and objects that the application should reject. This model has
no trained empty-view class and a high score alone is not proof of a valid deposit.
Real camera performance has not been measured in this build.

An ESP32-CAM can send images to a computer running the predictor; the included
PyTorch and ONNX models are not ESP32 firmware. No server was deployed and no
machine or reward service was connected as part of this build.

## Licensing and provenance

Dataset attributions and source license metadata are in `DATA_SOURCES.md` and
`reports/source_metadata.json`. Ultralytics models/code and the accompanying
custom scripts follow AGPL-3.0 unless a separately applicable license is obtained.
