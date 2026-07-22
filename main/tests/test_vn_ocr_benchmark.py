"""Tests for the Vietnamese plate OCR benchmark and deployment gate."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.benchmark_vn_plate_ocr import (
    compute_cer,
    deployment_gate,
    run_benchmark,
)
from src.datasets.plate_ocr_dataset import CORE_FIELDS


def write_manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    for item in rows:
        image_path = tmp_path / str(item["image_path"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), np.full((24, 80, 3), 127, dtype=np.uint8))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CORE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def row(
    image_path: str,
    label: str,
    *,
    group_id: str,
) -> dict[str, object]:
    return {
        "image_path": image_path,
        "label": label,
        "source_type": "real",
        "group_id": group_id,
        "split": "test",
        "verified": True,
    }


def test_gate_requires_both_real_sets_at_90_percent():
    result = deployment_gate(frozen_exact=15 / 16, expanded_exact=0.89, cer=0.02)
    assert result.passed is False
    assert any("expanded exact match" in reason for reason in result.reasons)


def test_character_accuracy_cannot_replace_exact_match():
    result = deployment_gate(frozen_exact=14 / 16, expanded_exact=0.95, cer=0.001)
    assert result.passed is False
    assert any("frozen exact match" in reason for reason in result.reasons)


def test_gate_passes_when_all_thresholds_are_met():
    result = deployment_gate(
        frozen_exact=0.95,
        expanded_exact=0.92,
        cer=0.02,
        candidate_p95_ms=40.0,
        paddle_p95_ms=80.0,
    )
    assert result.passed is True
    assert result.reasons == []


def test_gate_requires_candidate_p95_lower_than_paddle_when_both_provided():
    result = deployment_gate(
        frozen_exact=0.95,
        expanded_exact=0.95,
        cer=0.01,
        candidate_p95_ms=90.0,
        paddle_p95_ms=80.0,
    )
    assert result.passed is False
    assert any("p95 latency" in reason for reason in result.reasons)


def test_compute_cer_uses_total_truth_characters():
    cer = compute_cer(["30A1234", "51B6789"], ["30A12345", "51B67890"])
    assert cer == pytest.approx(2 / 16)


class _FakeReader:
    def __init__(self, predictions: list[str], latency_ms: float = 5.0) -> None:
        self._predictions = list(predictions)
        self._latency_ms = latency_ms

    def read_plate(self, image: np.ndarray) -> dict[str, float | str]:
        del image
        time.sleep(self._latency_ms / 1000.0)
        if not self._predictions:
            return {"text": "", "ocr_conf": 0.0}
        text = self._predictions.pop(0)
        return {"text": text, "ocr_conf": 0.9}


def _reader_predictions(truths: list[str], warmup_runs: int) -> list[str]:
    prefix = [truths[0]] * (1 + max(0, warmup_runs - 1))
    return prefix + truths


def test_run_benchmark_dry_run_writes_expected_structure(tmp_path):
    frozen_manifest = write_manifest(
        tmp_path / "frozen",
        [
            row("a.png", "30M71854", group_id="g1"),
            row("b.png", "51B67890", group_id="g2"),
        ],
    )
    expanded_manifest = write_manifest(
        tmp_path / "expanded",
        [row("c.png", "29A12345", group_id="g3")],
    )

    truths = ["30M71854", "51B67890", "29A12345"]
    reader_predictions = _reader_predictions(truths, warmup_runs=0)

    def candidate_factory() -> _FakeReader:
        return _FakeReader(list(reader_predictions), latency_ms=4.0)

    def paddle_factory() -> _FakeReader:
        return _FakeReader(list(reader_predictions), latency_ms=12.0)

    report = run_benchmark(
        frozen_manifest=frozen_manifest,
        expanded_manifest=expanded_manifest,
        engine="both",
        warmup_runs=0,
        candidate_factory=candidate_factory,
        paddle_factory=paddle_factory,
        candidate_model_path=tmp_path / "candidate.onnx",
    )
    (tmp_path / "candidate.onnx").write_bytes(b"onnx")

    assert report["passed"] is True
    assert report["reasons"] == []
    assert report["candidate"]["frozen"]["sample_count"] == 2
    assert report["candidate"]["expanded"]["sample_count"] == 1
    assert report["paddle"]["frozen"]["sample_count"] == 2
    assert report["paddle"]["expanded"]["sample_count"] == 1
    assert len(report["candidate"]["frozen"]["samples"]) == 2
    assert len(report["candidate"]["expanded"]["samples"]) == 1
    sample = report["candidate"]["frozen"]["samples"][0]
    assert set(sample) == {
        "image_path",
        "truth",
        "prediction",
        "confidence",
        "latency_ms",
        "correct",
        "valid_format",
    }
    assert sample["correct"] is True


def test_run_benchmark_output_json_roundtrip(tmp_path):
    frozen_manifest = write_manifest(
        tmp_path / "frozen",
        [row("a.png", "30M71854", group_id="g1")],
    )
    expanded_manifest = write_manifest(
        tmp_path / "expanded",
        [row("b.png", "51B67890", group_id="g2")],
    )
    truths = ["30M71854", "51B67890"]
    predictions = _reader_predictions(truths, warmup_runs=0)

    report = run_benchmark(
        frozen_manifest=frozen_manifest,
        expanded_manifest=expanded_manifest,
        engine="candidate",
        warmup_runs=0,
        candidate_factory=lambda: _FakeReader(list(predictions)),
        candidate_model_path=tmp_path / "candidate.onnx",
    )
    (tmp_path / "candidate.onnx").write_bytes(b"onnx")
    output_path = tmp_path / "gate.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["passed"] is True
    assert isinstance(loaded["passed"], bool)
