"""Benchmark C — license-plate OCR engines on a labelled plate-crop set.

Compares, on ``data/test/ocr_eval`` (16 hand-labelled real CCTV plate crops):
  * easyocr            — current pipeline OCR (EasyOCR readtext + sort/merge)
  * easyocr+enhance    — B1: deskew + CLAHE + upscale, then EasyOCR
  * ppocr              — B2: PaddleOCR (PP-OCRv4)
  * ppocr+enhance      — B2 + the same enhancement

Metrics: exact-match accuracy, mean CER (Levenshtein / GT length), mean
latency (ms/plate, CPU). Writes ``docs/benchmarks/ocr_benchmark.{csv,md}``.

Run from ``main/``::  python scripts/benchmark_ocr.py
"""

from __future__ import annotations

import csv
import re
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import Levenshtein

warnings.filterwarnings("ignore")
_MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MAIN))

EVAL_DIR = _MAIN / "data" / "test" / "ocr_eval"
OUT_DIR = _MAIN.parent / "docs" / "benchmarks"


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _enhance(bgr: np.ndarray) -> np.ndarray:
    """B1: deskew (clamped) + CLAHE + 2x upscale."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(th > 0))
    if len(coords) > 50:
        ang = cv2.minAreaRect(coords.astype(np.float32))[-1]
        ang = -(90 + ang) if ang < -45 else -ang
        if abs(ang) <= 20:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _load_eval():
    rows = list(csv.DictReader(open(EVAL_DIR / "labels.csv")))
    data = []
    for r in rows:
        img = cv2.imread(str(EVAL_DIR / f"{r['index']}.png"))
        if img is not None:
            data.append((img, _norm(r["plate"])))
    return data


def _score(name, predict_fn, data):
    cers, exact, t = [], 0, 0.0
    for img, gt in data:
        t0 = time.perf_counter()
        pred = _norm(predict_fn(img))
        t += time.perf_counter() - t0
        cers.append(Levenshtein.distance(pred, gt) / max(1, len(gt)))
        exact += int(pred == gt)
    n = len(data)
    return {"method": name, "exact_match": round(exact / n, 3),
            "mean_cer": round(float(np.mean(cers)), 3),
            "latency_ms": round(t / n * 1000, 1)}


def main() -> None:
    data = _load_eval()
    print(f"eval plates: {len(data)}", flush=True)

    from src.models.ocr import PlateOCR
    from paddleocr import PaddleOCR

    easy = PlateOCR()
    pp = PaddleOCR(lang="en", use_textline_orientation=False)

    def easy_read(img):
        return easy.read_plate(img)

    def pp_read(img):
        res = pp.predict(img)
        if res and hasattr(res[0], "get"):
            return "".join(res[0].get("rec_texts", []))
        return ""

    methods = [
        ("easyocr", easy_read),
        ("easyocr+enhance", lambda im: easy_read(_enhance(im))),
        ("ppocr", pp_read),
        ("ppocr+enhance", lambda im: pp_read(_enhance(im))),
    ]
    rows = []
    for name, fn in methods:
        r = _score(name, fn, data)
        rows.append(r)
        print(f"{name}: exact={r['exact_match']} CER={r['mean_cer']} lat={r['latency_ms']}ms", flush=True)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "ocr_benchmark.csv", index=False)
    best = df.sort_values(["exact_match", "mean_cer"], ascending=[False, True]).iloc[0]
    (OUT_DIR / "ocr_benchmark.md").write_text(
        "# Benchmark C — License-Plate OCR\n\n"
        f"Eval set: `data/test/ocr_eval` ({len(data)} hand-labelled real CCTV plate crops, "
        "incl. one 2-line motorbike plate). CER = Levenshtein / ground-truth length; "
        "latency = CPU ms/plate.\n\n" + df.to_markdown(index=False) + "\n\n"
        f"**Winner: `{best['method']}`** (exact-match {best['exact_match']}, "
        f"CER {best['mean_cer']}, {best['latency_ms']} ms/plate).\n",
        encoding="utf-8")
    print("\n" + df.to_markdown(index=False), flush=True)
    print(f"\nwinner: {best['method']}", flush=True)


if __name__ == "__main__":
    main()
