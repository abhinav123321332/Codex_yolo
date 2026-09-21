# Dataset sources and label decisions

The archives were inspected and processed on 2026-09-20. Counts below come from
the actual files, not advertised dataset totals. The dataset manifest records
the source member, crop coordinates, grouping, and final split of every example.

| Source | Exact version | What was used |
| --- | --- | --- |
| User upload, `Plastic and Cans.zip` | Uploaded file | All 3,849 metal/plastic images before cross-source cleaning. No object-location labels. |
| [Plastic Bottles in the Wild](https://www.kaggle.com/datasets/siddharthkumarsah/plastic-bottles-image-dataset/versions/2) | 2 | 3,999 images with 5,528 annotation rows; valid box/polygon bounds converted to plastic crops. Tiny and invalid crops excluded. |
| [Water Bottle Image Classification Dataset](https://www.kaggle.com/datasets/chethuhn/water-bottle-dataset) | 1 | Downloaded and inspected (486 images), excluded from material training: full/half/overflowing are water-level labels. They do not establish whether a container is plastic or glass. |
| [Garbage Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) | 12 | 12,259 original images. Resized copies in `standardized_256` and `standardized_384` were not counted again. Metal/plastic kept; eight other waste categories mapped to `other`. |
| [Multi-Class Waste Image Classification](https://www.kaggle.com/datasets/mayankbansal2205/multi-class-waste-image-classification-dataset) | 2 | 770 images. Metal waste to `metal`, plastic waste to `plastic`, remaining seven classes to `other`. |

These labels support broad material classification. They do not consistently
distinguish metal cans from every other metal object, identify PET resin, or
verify that a bottle is empty or eligible for a reward. The broad `other` class
does not guarantee rejection of every unseen object.

The source metadata lists CC BY 4.0 for Plastic Bottles in the Wild and the
Multi-Class Waste dataset, CC0 for the water-level dataset, and MIT for the Garbage
Dataset. The original authors retain their respective rights. The upload has no
license file in the ZIP; retain its original provenance when sharing data.

Preparation changes: polygon-to-rectangle crop bounds where needed, 5% crop
context per side, RGB conversion, aspect-preserving resize and padding to 224,
JPEG quality 95, exact duplicate removal, grouping, and a fresh split. No box
annotations were fabricated from image-level labels.

Exact normalized-image duplicates and related groups have zero cross-split
overlap. Perceptual grouping and blocks of adjacent uploaded filenames reduce
leakage but cannot prove that every physical item or photographic session is
unique to one split. Test results measure this curated dataset; an independent
camera dataset is still needed to measure real deployment performance.

Implementation references:

- [Official YOLOv8 models](https://docs.ultralytics.com/models/yolov8/)
- [Official classification task guide](https://docs.ultralytics.com/tasks/classify/)
- [Official classification dataset structure](https://docs.ultralytics.com/datasets/classify/)

YOLOv8 code and model licensing follows Ultralytics' AGPL-3.0 / applicable
commercial terms. The custom scripts in this package are supplied under AGPL-3.0
to match that dependency; dataset licenses remain separate.
