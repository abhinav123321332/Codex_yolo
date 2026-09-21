"""CPU-friendly transfer learning: freeze YOLOv8n features, train its final linear head.

The test split is never loaded here. Use finetune.py for optional end-to-end GPU training.
"""
import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO, __version__ as ultralytics_version
from common import CLASSES, IMAGE_SIZE, image_array, read_manifest, classification_metrics


class Images(Dataset):
    def __init__(self, root, rows, flip=False): self.root, self.rows, self.flip = Path(root), rows, flip
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            array = image_array(image)
        if self.flip: array = array[:, :, ::-1].copy()
        return torch.from_numpy(array), row["class_id"]


def features(model, root, rows, batch, workers, flip=False):
    loader = DataLoader(Images(root, rows, flip), batch_size=batch, shuffle=False,
                        num_workers=workers, persistent_workers=workers > 0)
    captured = []; output = []; labels = []
    hook = model.model[-1].linear.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for step, (images, targets) in enumerate(loader):
                model(images)
                output.append(captured.pop().cpu()); labels.append(targets)
                if step % 50 == 0:
                    print(f"Features {'flipped' if flip else 'original'}: {min((step+1)*batch,len(rows))}/{len(rows)}; {time.monotonic()-start:.1f}s", flush=True)
    finally:
        hook.remove()
    return torch.cat(output), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--weights", default="yolov8n-cls.pt")
    parser.add_argument("--out", default=".")
    parser.add_argument("--cache", default="feature_cache")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    output = Path(args.out); (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.data)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not train_rows or not val_rows: raise ValueError("Both train and val splits are required")
    wrapper = YOLO(args.weights, task="classify"); model = wrapper.model.float().cpu().eval()
    if wrapper.task != "classify": raise ValueError("Expected a YOLOv8 classification checkpoint")
    if len(model.model) != 10 or model.model[-1].linear.in_features != 1280:
        raise ValueError("This reproducible recipe expects the YOLOv8n-cls architecture")
    for parameter in model.parameters(): parameter.requires_grad_(False)
    initial_weight_hash = hashlib.sha256(Path(wrapper.ckpt_path).read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256((Path(args.data) / "manifest.jsonl").read_bytes()).hexdigest()
    cache_key = hashlib.sha256((manifest_hash + initial_weight_hash + "rgb-pad224-v1").encode()).hexdigest()[:20]
    cache_path = cache / (cache_key + ".pt")
    if cache_path.is_file():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_x, train_y, val_x, val_y = [saved[k] for k in ("train_x", "train_y", "val_x", "val_y")]
        print("Reused matching feature cache", flush=True)
    else:
        x1, y1 = features(model, args.data, train_rows, args.batch, args.workers)
        x2, y2 = features(model, args.data, train_rows, args.batch, args.workers, flip=True)
        val_x, val_y = features(model, args.data, val_rows, args.batch, args.workers)
        train_x, train_y = torch.cat((x1, x2)), torch.cat((y1, y2))
        torch.save(dict(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y), cache_path)
    # A real trained YOLO classification head; the pretrained backbone stays frozen.
    head = nn.Linear(train_x.shape[1], len(CLASSES))
    class_counts = torch.bincount(train_y, minlength=len(CLASSES)).float()
    if (class_counts == 0).any(): raise ValueError("A training class has no samples")
    class_weights = class_counts.sum() / (len(CLASSES) * class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.0001)
    history = []; best_score = -1.; best_epoch = 0; best_state = None; start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        head.train(); permutation = torch.randperm(len(train_y)); losses = []
        for indices in permutation.split(256):
            batch_x = nn.functional.dropout(train_x[indices], p=.15, training=True)
            loss = criterion(head(batch_x), train_y[indices])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():
            logits = head(val_x); val_loss = float(criterion(logits, val_y))
            metrics = classification_metrics(val_y.numpy(), logits.argmax(1).numpy())
        score = metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                        "val_accuracy": metrics["accuracy"], "val_macro_f1": score})
        if score > best_score + 1e-7:
            best_score, best_epoch, best_state = score, epoch, deepcopy(head.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch}: val_accuracy={metrics['accuracy']:.4f}, val_macro_f1={score:.4f}, best_epoch={best_epoch}", flush=True)
        scheduler.step()
        if epoch - best_epoch >= args.patience: break
    head.load_state_dict(best_state); head.eval(); model.model[-1].linear = head
    model.names = dict(enumerate(CLASSES)); model.yaml["nc"] = len(CLASSES)
    model.args.update({"imgsz": IMAGE_SIZE, "task": "classify", "data": str(Path(args.data).resolve())})
    wrapper.ckpt = {"train_args": dict(model.args), "training_method": "frozen YOLOv8n-cls features; fitted linear head",
                    "best_epoch": best_epoch, "manifest_sha256": manifest_hash}
    wrapper.save(output / "models" / "best.pt")
    with torch.inference_mode():
        validation_probabilities = head(val_x).softmax(1).numpy()
    report = {"status": "trained", "architecture": "YOLOv8n-cls", "ultralytics": ultralytics_version,
              "torch": torch.__version__, "device": "cpu", "image_size": IMAGE_SIZE,
              "method": "Transfer learning: ImageNet pretrained backbone frozen; final 1280-to-3 linear classifier trained.",
              "augmentation": "Original and horizontal-flipped training images; 15% feature dropout",
              "preprocessing": "EXIF transpose, RGB, aspect-preserving 224x224 padding (114,114,114), divide by 255",
              "train_images": len(train_rows), "train_feature_vectors": len(train_y), "validation_images": len(val_rows),
              "test_used_for_training_or_selection": False, "seed": args.seed, "epochs_completed": len(history),
              "best_epoch": best_epoch, "selection_metric": "validation macro F1", "head_training_seconds": time.monotonic()-start,
              "initial_weights_sha256": initial_weight_hash, "manifest_sha256": manifest_hash,
              "validation": classification_metrics(val_y.numpy(), validation_probabilities.argmax(1)),
              "trainable_parameter_count": sum(p.numel() for p in head.parameters())}
    (output / "reports" / "training_report.json").write_text(json.dumps(report, indent=2))
    with (output / "reports" / "training_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    (output / "models" / "metadata.json").write_text(json.dumps({"architecture": "YOLOv8n-cls", "task": "classify",
        "classes": CLASSES, "image_size": IMAGE_SIZE, "preprocessing": report["preprocessing"],
        "default_confidence_threshold": .8, "default_margin_threshold": .15,
        "thresholds_calibrated": False, "training_method": report["method"]}, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
