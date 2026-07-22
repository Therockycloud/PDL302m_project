# Benchmark C — License-Plate OCR

Eval set: `data/test/ocr_eval` (16 hand-labelled real CCTV plate crops, incl. one 2-line motorbike plate). CER = Levenshtein / ground-truth length; latency = CPU ms/plate.

| method          |   exact_match |   mean_cer |   latency_ms |
|:----------------|--------------:|-----------:|-------------:|
| easyocr         |         0     |      0.278 |         31.7 |
| easyocr+enhance |         0.062 |      0.289 |         47.2 |
| ppocr           |         0.812 |      0.031 |        423.2 |
| ppocr+enhance   |         0.438 |      0.092 |        423.3 |

**Winner: `ppocr`** (exact-match 0.812, CER 0.031, 423.2 ms/plate).

## Conclusion

- **PaddleOCR (B2) is the decisive winner** — it lifts exact-match from **0 % → 81 %**
  and cuts CER from 0.28 → **0.03** versus the current EasyOCR. EasyOCR is
  effectively unusable on these CCTV plates.
- **B1 (deskew + CLAHE + upscale) does not help**: it nudges EasyOCR from 0 → 6 %
  exact, and it actively *hurts* PaddleOCR (0.81 → 0.44) — PP-OCR prefers the
  original colour crop, so the extra preprocessing is dropped.
- **Latency trade-off:** PP-OCR is ~13× slower per call (423 ms vs 32 ms). This is
  acceptable because OCR runs **once per parked vehicle** (gated by the parking
  trigger), not per frame. If tighter, PP-OCR can be run recognition-only (the
  YOLO stage already localises the plate) and/or with the mobile model — both cut
  latency substantially without losing accuracy.

**Decision: adopt PaddleOCR** (`ocr.engine: "ppocr"`), no enhancement. EasyOCR has
since been removed as a runtime fallback; deployed system is 100% PaddleOCR (hard
error if PaddleOCR is unavailable). EasyOCR remains available only for
training/evaluation.
