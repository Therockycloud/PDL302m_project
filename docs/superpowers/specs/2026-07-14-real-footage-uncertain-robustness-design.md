# Real-Footage UNCERTAIN Robustness + Demo Case

**Date:** 2026-07-14  
**Status:** Approved (approach C)

## Goal

Reduce premature `UNCERTAIN` on real parking footage when OCR already has useful evidence, and add a second real-world demo video to the dashboard. Keep the barrier closed when no plate can be localized.

## Non-goals

- Retrain detector / OCR weights.
- OCR the whole vehicle crop when the plate detector misses (must not resurrect VF3 badge false reads).
- Change Product-cam media-clock architecture.

## Decision / lock changes

Hard lock (unchanged shape, milder defaults):

- `lock_conf`: `0.60` → `0.50`
- `lock_repeat`: stays `2`
- `collect_frames`: `5` → `10`

Soft expire (new, only when the evidence window is exhausted without a hard lock):

1. If the same plate appears ≥2 times with `plate_conf >= soft_conf` (`0.40`) → lock that plurality plate.
2. Else if a single read has `plate_conf >= single_lock_conf` (`0.85`) → lock that read.
3. Else → `UNCERTAIN` / `LOG`.

Hard lock still wins early when 2 reads meet `lock_conf`.

Empty evidence (no non-empty `plate_text`) still becomes boundary `UNCERTAIN` with the existing “No readable plate…” message and does not open the barrier.

## Trigger / ROI defaults

Widen defaults so distant / off-center real approaches still open the gate, while the existing reverse-park clip remains usable:

- `roi`: `[0.20, 0.20, 0.80, 1.0]`
- `min_area_ratio` / `approach_min_area`: `0.10`

## Demo video + UI

- Install source clip as `main/data/test/parking_case_real_v2.mp4`.
- Dashboard Upload Video mode: add **Play Real-World Case** checkbox beside the default parking video control (mutually exclusive selection).
- If the plate text for the new clip is reliably readable, register it in `main/data/database.csv` so a correct lock can demo `AUTHORIZED`.

## Follow-up refinements (implemented)

- Sliding evidence window (`max_collect_frames` = 50) plus `max_ready_samples` = 300 so dense API sampling (`sample_interval=1`) does not finalize on early empty OCR.
- Lock eligibility requires `validate_vietnamese_plate` plus a complete numeric suffix (reject partial OCR like `30K`).
- Demo plate `30K-439.36` registered in `database.csv` for the v2 clip.
- Bundled file path: `main/data/test/parking_case_real_v2.mp4` (gitignored like other `.mp4` assets).

## Verification

- Unit tests for soft expire, single high-conf expire lock, invalid partial plate non-lock, and empty-plate → UNCERTAIN.
- Existing hard-lock tests still pass with updated defaults or explicit kwargs.
- Automated test run after implementation (`pytest` for touched engine/UI helpers; headless video probe if models available).
- Headless target for `parking_case_real_v2.mp4`: lock full plate `30K43936` (AUTHORIZED when DB row present).
