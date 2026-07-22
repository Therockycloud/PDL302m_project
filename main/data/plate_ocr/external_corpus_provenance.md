# External detector-corpus provenance

The ignored source corpus is expected at `main/data/raw/plate_det`. Its three
COCO exports identify themselves as Roboflow exports created 2022-01-13,
metadata version `1`, category `license_plate`, and license `CC BY 4.0`. The
embedded URL is literally
`https://public.roboflow.ai/object-detection/undefined`, so an exact Roboflow
project/version URL is not recoverable from the files and is not claimed here.

The project proposal lists Vietnamese plate sources from Kaggle, Roboflow
Universe, and Hugging Face, but does not map individual detector files to one
of those sources. Consequently those links are contextual leads, not asserted
origins for this exact export.

Local inventory used for the review:

- COCO records: train 6,176 images / 6,413 boxes; valid 1,765 / 1,840;
  test 882 / 902.
- YOLO image/label pairs used by candidate review: train 6,176; valid 1,765.
- Test export layout: 882 images beside its COCO file and no YOLO text labels.
- Final held-out subset: 102 crops from 102 distinct source-file SHA-256 values.

Restore an authorized saved copy and verify the exact annotation exports:

```sh
rsync -a --delete "$PLATE_DET_CORPUS"/ main/data/raw/plate_det/
shasum -a 256 -c main/data/plate_ocr/plate_det_coco.sha256
```

The expanded manifest remains usable without this external corpus because all
reviewed crops are committed. Provenance auditing always verifies committed
crop hashes; it additionally verifies source hashes and reconstructs crops
when the ignored corpus is present, otherwise reports the source unavailable.
