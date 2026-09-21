"""Map compatible labels, crop annotated bottles, deduplicate, and split by group.

No detection boxes are invented. The water-level dataset is intentionally excluded
because its class names do not identify container material.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageOps
from common import CLASSES, square_rgb

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAPPING = {
    "metal": "metal", "metal waste": "metal", "plastic": "plastic", "plastic waste": "plastic",
    **{name: "other" for name in ["battery", "biological", "cardboard", "clothes", "glass", "paper",
        "shoes", "trash", "electronic waste", "general waste", "glass waste", "hazardous waste",
        "medicinal waste", "organic waste", "paper and cardboard waste"]},
}
DCT = np.cos(np.pi * (2 * np.arange(32)[None, :] + 1) * np.arange(8)[:, None] / 64)


class UnionFind:
    def __init__(self, n): self.parents = list(range(n))
    def find(self, k):
        while self.parents[k] != k:
            self.parents[k] = self.parents[self.parents[k]]; k = self.parents[k]
        return k
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parents[max(a, b)] = min(a, b)


class BKTree:
    """Hamming-distance index; used only to group similar images across splits."""
    def __init__(self): self.root = None
    def add(self, value, index):
        if self.root is None: self.root = [value, [index], {}]; return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if not distance: node[1].append(index); return
            if distance not in node[2]: node[2][distance] = [value, [index], {}]; return
            node = node[2][distance]
    def query(self, value, radius=3):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop(); distance = (value ^ node[0]).bit_count()
            if distance <= radius: yield from node[1]
            stack.extend(child for d, child in node[2].items() if distance - radius <= d <= distance + radius)


def phash(image):
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    values = (DCT @ pixels @ DCT.T).flatten()
    bits = values > np.median(values[1:]); bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def base_name(path):
    stem = Path(path).stem.split(".rf.")[0]
    return re.sub(r"_(jpg|jpeg|png)$", "", stem, flags=re.I)


def process_image(task):
    source, archive, member, label, group, crop, original_split = task
    try:
        with Image.open(io.BytesIO(archive.read(member))) as opened:
            opened.load()
            # YOLO coordinates describe stored pixel orientation, before EXIF transforms.
            if crop is not None:
                image = opened.convert("RGB")
                xc, yc, width, height = crop; iw, ih = image.size
                # Include 5% of the box width/height on each side as context.
                box = (max(0, math.floor((xc - width * .55) * iw)),
                       max(0, math.floor((yc - height * .55) * ih)),
                       min(iw, math.ceil((xc + width * .55) * iw)),
                       min(ih, math.ceil((yc + height * .55) * ih)))
                if box[2] - box[0] < 24 or box[3] - box[1] < 24:
                    return None, {"source": source, "member": member, "reason": "crop smaller than 24 pixels"}
                image = image.crop(box)
            else:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            aspect = image.width / image.height
            image = square_rgb(image)
            raw_hash = hashlib.sha256(image.tobytes()).hexdigest()
            small = np.asarray(image.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.int16)
            buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=95)
            record = {"source": source, "member": member, "label": label,
                      "origin_group": group, "original_split": original_split,
                      "crop_xywh_normalized": crop, "pixel_sha256": raw_hash,
                      "phash": f"{phash(image):016x}", "aspect": aspect}
            return (record, buffer.getvalue(), small), None
    except Exception as error:
        return None, {"source": source, "member": member, "reason": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uploaded", required=True, help="Path to Plastic and Cans.zip")
    parser.add_argument("--downloads", default="downloads")
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(); destination = Path(args.out)
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"Output folder is not empty: {destination}. Choose a fresh --out directory.")
    staging = destination / "staging"; staging.mkdir(parents=True, exist_ok=True)
    downloads = Path(args.downloads)
    specifications = [("uploaded", Path(args.uploaded)),
        ("wild_bottles", downloads / "siddharthkumarsah.zip"),
        ("garbage_v12", downloads / "sumn2u.zip"),
        ("wastesnap_v2", downloads / "mayankbansal2205.zip")]
    archives = []; tasks = []; errors = []; source_counts = Counter()
    for source, path in specifications:
        if not path.is_file(): raise FileNotFoundError(path)
        archive = zipfile.ZipFile(path); archives.append(archive)
        for member in sorted(archive.namelist()):
            if Path(member).suffix.lower() not in EXTENSIONS: continue
            parts = member.split("/")
            if source == "garbage_v12" and parts[0] != "original": continue
            split = next((p for p in parts if p in ("train", "val", "valid", "test")), "unsplit")
            if source == "wild_bottles":
                label_path = str(Path(member).parent.parent / "labels" / (Path(member).stem + ".txt"))
                if label_path not in archive.namelist(): raise ValueError(f"Missing annotation: {label_path}")
                for line in archive.read(label_path).decode().strip().splitlines():
                    values = line.split()
                    if not values or values[0] != "0": raise ValueError(f"Unexpected annotation: {label_path}")
                    coordinates = [float(x) for x in values[1:]]
                    if len(values) == 5:
                        box = coordinates
                    elif len(values) >= 7 and len(values) % 2 == 1:
                        # This source mixes YOLO boxes and segmentation polygons.
                        if not all(math.isfinite(x) and 0 <= x <= 1 for x in coordinates):
                            errors.append({"source": source, "member": member, "reason": "invalid polygon coordinates"}); continue
                        xs, ys = coordinates[::2], coordinates[1::2]
                        box = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                               max(xs) - min(xs), max(ys) - min(ys)]
                    else:
                        errors.append({"source": source, "member": member, "reason": "malformed annotation"}); continue
                    if not all(math.isfinite(x) for x in box) or not all(0 <= x <= 1 for x in box) or min(box[2:]) <= 0:
                        errors.append({"source": source, "member": member, "reason": "invalid bounding box"}); continue
                    tasks.append((source, archive, member, "plastic", source + ":" + base_name(member), box, split))
                    source_counts[source] += 1
            else:
                category = parts[-2].lower()
                if category not in MAPPING: raise ValueError(f"Unmapped category: {category}")
                label = MAPPING[category]
                group = source + ":" + category + ":" + base_name(member)
                if source == "uploaded":
                    # Adjacent numbered views can show the same item: keep blocks of 100 together.
                    match = re.match(r"(?:metal|plastic)(\d+)", base_name(member))
                    if match: group = source + ":" + category + ":capture_block_" + str(int(match[1]) // 100)
                tasks.append((source, archive, member, label, group, None, split)); source_counts[source] += 1
    print("Candidate images/crops:", dict(source_counts), flush=True)
    records = []; thumbnails = []; exact = {}; conflicts = set(); duplicate_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for number, (result, error) in enumerate(executor.map(process_image, tasks), 1):
            if error: errors.append(error); continue
            record, content, thumbnail = result
            key = record["pixel_sha256"]
            if key in exact:
                previous = records[exact[key]]
                previous["alias_groups"].append(record["origin_group"])
                if previous["label"] != record["label"]: conflicts.add(key)
                duplicate_count += 1
            else:
                index = len(records); exact[key] = index
                record["alias_groups"] = [record["origin_group"]]
                record["id"] = key[:24]
                (staging / (record["id"] + ".jpg")).write_bytes(content)
                records.append(record); thumbnails.append(thumbnail)
            if number % 2000 == 0: print(f"Processed {number}/{len(tasks)}", flush=True)
    for archive in archives: archive.close()
    union = UnionFind(len(records)); named_groups = {}; tree = BKTree(); near_links = 0
    for i, record in enumerate(records):
        for group in record["alias_groups"]:
            if group in named_groups: union.union(i, named_groups[group])
            else: named_groups[group] = i
        value = int(record["phash"], 16)
        for j in tree.query(value, radius=3):
            # pHash alone can confuse two different objects on the same blank background.
            if np.mean(np.abs(thumbnails[i] - thumbnails[j])) <= 18:
                union.union(i, j); near_links += 1
        tree.add(value, i)
    grouped = defaultdict(list)
    for i, record in enumerate(records): grouped[union.find(i)].append(i)
    excluded_groups = {union.find(exact[key]) for key in conflicts}
    strata = defaultdict(list)
    for group, members in grouped.items():
        if group in excluded_groups: continue
        primary = Counter((records[i]["label"], records[i]["source"]) for i in members).most_common(1)[0][0]
        strata[primary].append(group)
    assignments = {}; rng = random.Random(args.seed)
    for stratum, groups in sorted(strata.items()):
        rng.shuffle(groups)
        total = sum(len(grouped[g]) for g in groups)
        targets = {"train": total * .70, "val": total * .15, "test": total * .15}; used = Counter()
        # Allocate whole groups by largest remaining proportional deficit.
        for group in groups:
            split = max(targets, key=lambda s: (targets[s] - used[s]) / max(targets[s], 1))
            assignments[group] = split; used[split] += len(grouped[group])
    counts = Counter(); by_source = Counter(); manifest = []; excluded_count = 0
    for i, record in enumerate(records):
        group = union.find(i)
        if group in excluded_groups:
            excluded_count += 1; continue
        split = assignments[group]; label = record["label"]
        relative = Path(split) / label / (record["id"] + ".jpg")
        target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging / (record["id"] + ".jpg"), target)
        record.update({"path": relative.as_posix(), "split": split, "group": str(group), "class_id": CLASSES.index(label)})
        record.pop("alias_groups"); record.pop("aspect")
        manifest.append(record); counts[split + "/" + label] += 1; by_source[record["source"] + "/" + split] += 1
    for split in ("train", "val", "test"):
        for name in CLASSES:
            if counts[split + "/" + name] < 1: raise ValueError(f"Empty {split}/{name}")
    shutil.rmtree(staging)
    (destination / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in manifest))
    report = {"classes": CLASSES, "seed": args.seed, "candidate_counts": dict(source_counts),
        "retained_images": len(manifest), "split_counts": dict(sorted(counts.items())),
        "source_split_counts": dict(sorted(by_source.items())), "exact_duplicates_removed": duplicate_count,
        "conflicting_exact_hashes": len(conflicts), "images_excluded_in_conflicting_groups": excluded_count,
        "perceptual_group_links": near_links, "groups": len(assignments),
        "errors_or_tiny_crops": errors,
        "water_level_dataset": {"used": False, "reason": "Water-level labels do not establish container material."},
        "split_method": "Fresh 70/15/15 target split, source/class stratified, whole groups; supplied splits rebuilt.",
        "grouping": "Exact normalized pixels; source base names; uploaded consecutive blocks of 100; pHash Hamming<=3 plus RGB MAE<=18.",
        "limitations": "Heuristic grouping cannot guarantee unseen physical items or capture sessions across splits."}
    (destination / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "errors_or_tiny_crops"}, indent=2), flush=True)


if __name__ == "__main__": main()
