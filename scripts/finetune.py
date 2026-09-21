"""Optional end-to-end refinement on a GPU. Does not overwrite the shipped model."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--out", default="finetuned")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); output = Path(args.out).resolve()
    if output.exists(): raise SystemExit("Choose a new --out folder to preserve earlier results")
    data = Path(args.data).resolve()
    if not (data / "train").is_dir() or not (data / "val").is_dir(): raise SystemExit("Prepared train/val folders are required")
    model = YOLO(args.weights)
    model.train(data=str(data), epochs=args.epochs, imgsz=224, batch=args.batch,
                device=args.device, workers=2, optimizer="AdamW", lr0=.0003,
                weight_decay=.0005, patience=12, cos_lr=True, seed=42,
                scale=.1, fliplr=.5, erasing=.05, auto_augment=None,
                project=str(output), name="run", plots=True, exist_ok=False)
    destination = output / "models"; destination.mkdir()
    shutil.copy2(model.trainer.best, destination / "best.pt")
    metadata = json.loads((Path(args.weights).parent / "metadata.json").read_text())
    metadata["training_method"] = "End-to-end fine tuning of the supplied transfer-learning model"
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Fine-tuned checkpoint: {destination / 'best.pt'}")
    print("Evaluate the new checkpoint once on a reserved test set. Its metrics are separate from the shipped baseline.")


if __name__ == "__main__": main()
