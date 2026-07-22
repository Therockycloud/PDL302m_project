# Crawl Vietnamese Car Plate Corpora — Continuation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish interrupted public-corpus ingest so we have a reserved-disjoint Vietnamese **car** plate review pack with honest unique-label counts (soft target ≥200; report honestly if <50).

**Architecture:** Reuse existing YOLO-label crop → Apple Vision draft → near-reserved filter → blind contact-sheet review pipeline. Prefer completing already-downloaded artifacts before new crawls. Do not retrain or touch locked test sets.

**Tech Stack:** Python (`.venv` for Vision; Docker backend for OpenCV/detector), existing scripts under `main/scripts/`, pytest.

---

## Current state (do not redo)

| Source | Status | Usable car unique (approx) |
|--------|--------|----------------------------|
| `plate_det` train/valid/test | mined | ~7–28 after reserved (sparse) |
| `license_plates_kaggle` (winter2897 + mrzaizai2k) | imported + audited | **29** unique (heavy overlap with plate_det) |
| `license_plates_crawl_hf/_downloads/dataset.zip` | **959MB zip present, not extracted** | unknown — priority |
| `license_plates_crawl_trungdinh/...part` | **incomplete gdown** (~193MB) | Drive id `1xchPXf7a1r466ngow_W_9bittRqQEf_T` (trungdinh22; may overlap Mi AI) |

Locked reserved manifests: `real_validation.csv`, `expanded_real_test.csv`, `frozen_regression.csv`.

Env: broken Homebrew `python3` — use `.venv/bin/python` or `docker compose exec -T -w /app/main backend ...`. No `~/.kaggle/kaggle.json`.

---

## File Map

- Extend: `main/scripts/download_external_car_plates.py` (or add `download_crawl_car_plates.py`) for HF extract + trungdinh resume + PROVENANCE
- Extend: `main/scripts/export_external_car_train_review.py` / shared pipeline to apply `near_reserved_filter`
- Create: `main/data/raw/license_plates_crawl_hf/PROVENANCE.md`
- Create: `main/data/raw/license_plates_crawl_trungdinh/PROVENANCE.md` (if download completes)
- Create: review under `main/data/plate_ocr/review/crawl_hf_car_audit/` (and optional `crawl_trungdinh_car_audit/`)
- Tests: extend `main/tests/test_download_external_car_plates.py` / near-reserved coverage

---

### Task 1: Extract HF zip + PROVENANCE

**Files:** extract into `main/data/raw/license_plates_crawl_hf/`; write `PROVENANCE.md`; small script/CLI hook + test if new code.

- [ ] Identify dataset origin from zip (`dataset.yaml` classes `BSD`/`BSV`, paths `TongHop\YOLODataset`) and HF cache if possible; document URL + license honestly (unknown → say unknown, do not invent).
- [ ] Extract zip to `images/{train,val,test}` + `labels/...` layout usable by existing YOLO crop pipeline.
- [ ] Inventory: image/label counts, sample resolutions, class histogram.
- [ ] Commit scripts/tests/PROVENANCE only (not blobs).

### Task 2: Resume trungdinh Drive download

**Files:** `license_plates_crawl_trungdinh/`

- [ ] Resume/re-download Google Drive file `1xchPXf7a1r466ngow_W_9bittRqQEf_T` via `gdown` into `_downloads/`, extract YOLO layout.
- [ ] PROVENANCE.md noting trungdinh22 repo + Mi AI/winter2897 overlap risk.
- [ ] If download fails, document exact blocker; do not block Task 3 on HF.

### Task 3: Car review export with near-reserved filter

**Files:** reuse/adapt `export_external_car_train_review.py` + `near_reserved_filter.py`

- [ ] Wire **near-reserved** into external/crawl export path (exact audits already miss this).
- [ ] Run on HF corpus → `main/data/plate_ocr/review/crawl_hf_car_audit/` (candidates, crops, blind sheets ≤250 cells, sheet_map, stats.json).
- [ ] Optionally run on trungdinh if ready.
- [ ] Report: kept, unique_labels, drop histogram, leakage zeros (label/source/crop + near-reserved).
- [ ] Commit script/test changes only.

### Task 4: If unique_labels < 50 after HF (+ trungdinh)

- [ ] Try one more public source (prefer Roboflow Universe public zip or GitHub release; no social scrape).
- [ ] Or Wikimedia Commons capped crawl if scripts already exist.
- [ ] Otherwise stop and report gap + next user actions (Kaggle creds / site CCTV).

### Out of scope

- Retrain CTC, deployment gate on locked sets, runtime switch to ONNX, commit large blobs, push.

---

## Verify

```bash
.venv/bin/python -m pytest -q main/tests/test_download_external_car_plates.py main/tests/test_near_reserved_filter.py
# after export:
python3 -c 'import json;print(json.load(open("main/data/plate_ocr/review/crawl_hf_car_audit/stats.json"))["unique_labels"])'
```
