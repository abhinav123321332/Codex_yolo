"""Download the exact public Kaggle versions inspected for this build."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCES = [
    ("siddharthkumarsah/plastic-bottles-image-dataset", 2),
    ("chethuhn/water-bottle-dataset", 1),
    ("sumn2u/garbage-classification-v2", 12),
    ("mayankbansal2205/multi-class-waste-image-classification-dataset", 2),
]


def download(item, destination):
    slug, version = item
    path = destination / (slug.split("/")[0] + ".zip")
    if path.exists() and zipfile.is_zipfile(path):
        return {"source": slug, "version": version, "path": str(path), "status": "already present"}
    request = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/datasets/download/{slug}?datasetVersionNumber={version}",
        headers={"User-Agent": "Waste-YOLOv8-dataset-preparation"})
    temporary = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as file:
            while chunk := response.read(4 * 1024 * 1024):
                file.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Kaggle did not return a ZIP archive. Download this version from its dataset page.")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"source": slug, "version": version, "path": str(path), "status": "downloaded"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="downloads")
    args = parser.parse_args()
    destination = Path(args.out); destination.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download, item, destination): item for item in SOURCES}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Stop on errors; never silently omit a requested source.
            print(json.dumps(result), flush=True); results.append(result)
    (destination / "download_manifest.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
