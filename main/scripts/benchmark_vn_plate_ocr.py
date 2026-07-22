#!/usr/bin/env python3
"""Benchmark Vietnamese plate OCR candidates against PaddleOCR with a deployment gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from src.datasets.plate_ocr_dataset import PlateManifestRow, load_plate_manifest  # noqa: E402
from src.models.vn_plate_text import validate_vietnamese_plate  # noqa: E402


Reader = Callable[[np.ndarray], dict[str, float | str]]
ReaderFactory = Callable[[], Any]


class _SupportsReadPlate(Protocol):
    def read_plate(self, image: np.ndarray) -> dict[str, float | str]:
        ...


FROZEN_EXACT_THRESHOLD = 0.90
EXPANDED_EXACT_THRESHOLD = 0.90
CER_THRESHOLD = 0.031


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SampleResult:
    image_path: str
    truth: str
    prediction: str
    confidence: float
    latency_ms: float
    correct: bool
    valid_format: bool


@dataclass(frozen=True, slots=True)
class EngineReport:
    engine: str
    sample_count: int
    exact_match: float
    cer: float
    invalid_format_rate: float
    cold_start_latency_ms: float | None
    warm_p50_latency_ms: float | None
    warm_p95_latency_ms: float | None
    model_size_bytes: int | None
    samples: list[SampleResult]


def deployment_gate(
    *,
    frozen_exact: float,
    expanded_exact: float,
    cer: float,
    candidate_p95_ms: float | None = None,
    paddle_p95_ms: float | None = None,
) -> GateResult:
    """Return whether a candidate satisfies the hard deployment thresholds."""

    reasons: list[str] = []
    if frozen_exact < FROZEN_EXACT_THRESHOLD:
        reasons.append(
            f"frozen exact match {frozen_exact:.4f} is below {FROZEN_EXACT_THRESHOLD:.2f}"
        )
    if expanded_exact < EXPANDED_EXACT_THRESHOLD:
        reasons.append(
            f"expanded exact match {expanded_exact:.4f} is below {EXPANDED_EXACT_THRESHOLD:.2f}"
        )
    if cer > CER_THRESHOLD:
        reasons.append(f"combined CER {cer:.4f} exceeds {CER_THRESHOLD:.3f}")
    if candidate_p95_ms is not None and paddle_p95_ms is not None:
        if candidate_p95_ms >= paddle_p95_ms:
            reasons.append(
                "candidate warm p95 latency "
                f"{candidate_p95_ms:.2f} ms is not lower than Paddle p95 {paddle_p95_ms:.2f} ms"
            )
    return GateResult(passed=not reasons, reasons=reasons)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (left_character != right_character)
            current.append(
                min(current[-1] + 1, previous[right_index] + 1, substitution)
            )
        previous = current
    return previous[-1]


def compute_cer(predictions: Sequence[str], truths: Sequence[str]) -> float:
    if len(predictions) != len(truths):
        raise ValueError("predictions and truths must have the same length")
    total_chars = sum(len(truth) for truth in truths)
    if total_chars == 0:
        return 0.0
    total_edits = sum(edit_distance(prediction, truth) for prediction, truth in zip(predictions, truths))
    return total_edits / total_chars


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _evaluate_manifest(
    rows: list[PlateManifestRow],
    reader: _SupportsReadPlate,
    *,
    engine_name: str,
    warmup_runs: int,
    model_size_bytes: int | None = None,
) -> EngineReport:
    if not rows:
        return EngineReport(
            engine=engine_name,
            sample_count=0,
            exact_match=0.0,
            cer=0.0,
            invalid_format_rate=0.0,
            cold_start_latency_ms=None,
            warm_p50_latency_ms=None,
            warm_p95_latency_ms=None,
            model_size_bytes=model_size_bytes,
            samples=[],
        )

    warmup_image = cv2.imread(str(rows[0].image_path), cv2.IMREAD_COLOR)
    if warmup_image is None:
        raise RuntimeError(f"could not decode warmup image: {rows[0].image_path}")

    cold_started = time.perf_counter()
    reader.read_plate(warmup_image)
    cold_start_latency_ms = (time.perf_counter() - cold_started) * 1000.0

    for _ in range(max(0, warmup_runs - 1)):
        reader.read_plate(warmup_image)

    samples: list[SampleResult] = []
    latencies: list[float] = []
    for row in rows:
        image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode plate image: {row.image_path}")
        started = time.perf_counter()
        reading = reader.read_plate(image)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = str(reading.get("text", ""))
        confidence = float(reading.get("ocr_conf", 0.0))
        valid_format = validate_vietnamese_plate(prediction)
        samples.append(
            SampleResult(
                image_path=str(row.image_path),
                truth=row.label,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                correct=prediction == row.label,
                valid_format=valid_format,
            )
        )
        latencies.append(latency_ms)

    predictions = [sample.prediction for sample in samples]
    truths = [sample.truth for sample in samples]
    exact_match = sum(sample.correct for sample in samples) / len(samples)
    cer = compute_cer(predictions, truths)
    invalid_format_rate = sum(not sample.valid_format for sample in samples) / len(samples)
    return EngineReport(
        engine=engine_name,
        sample_count=len(samples),
        exact_match=exact_match,
        cer=cer,
        invalid_format_rate=invalid_format_rate,
        cold_start_latency_ms=cold_start_latency_ms,
        warm_p50_latency_ms=_percentile(latencies, 50.0),
        warm_p95_latency_ms=_percentile(latencies, 95.0),
        model_size_bytes=model_size_bytes,
        samples=samples,
    )


def _report_to_dict(report: EngineReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["samples"] = [asdict(sample) for sample in report.samples]
    return payload


def run_benchmark(
    *,
    frozen_manifest: str | Path,
    expanded_manifest: str | Path,
    engine: str,
    warmup_runs: int,
    candidate_factory: ReaderFactory | None = None,
    paddle_factory: ReaderFactory | None = None,
    candidate_model_path: str | Path | None = None,
    candidate_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    frozen_rows = load_plate_manifest(frozen_manifest)
    expanded_rows = load_plate_manifest(expanded_manifest)
    if engine not in {"paddle", "candidate", "both"}:
        raise ValueError("engine must be one of: paddle, candidate, both")

    if candidate_factory is None:
        if candidate_model_path is None or candidate_metadata_path is None:
            raise ValueError("candidate model and metadata paths are required for candidate engine")
        from src.models.vn_plate_recognizer import VnPlateRecognizer

        def candidate_factory() -> VnPlateRecognizer:
            return VnPlateRecognizer(candidate_model_path, candidate_metadata_path)

    if paddle_factory is None:
        from src.models.ppocr_reader import PaddleOCRReader

        def paddle_factory() -> PaddleOCRReader:
            return PaddleOCRReader()

    candidate_report: EngineReport | None = None
    paddle_report: EngineReport | None = None
    candidate_size: int | None = None
    if candidate_model_path is not None:
        candidate_path = Path(candidate_model_path)
        if candidate_path.is_file():
            candidate_size = candidate_path.stat().st_size

    if engine in {"candidate", "both"}:
        candidate_reader = candidate_factory()
        candidate_report = _evaluate_manifest(
            frozen_rows + expanded_rows,
            candidate_reader,
            engine_name="candidate",
            warmup_runs=warmup_runs,
            model_size_bytes=candidate_size,
        )

    if engine in {"paddle", "both"}:
        paddle_reader = paddle_factory()
        paddle_report = _evaluate_manifest(
            frozen_rows + expanded_rows,
            paddle_reader,
            engine_name="paddle",
            warmup_runs=warmup_runs,
            model_size_bytes=None,
        )

    frozen_sample_count = len(frozen_rows)
    expanded_sample_count = len(expanded_rows)

    def _split_report(report: EngineReport | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if report is None:
            return None, None
        frozen_samples = report.samples[:frozen_sample_count]
        expanded_samples = report.samples[frozen_sample_count:]
        frozen_predictions = [sample.prediction for sample in frozen_samples]
        frozen_truths = [sample.truth for sample in frozen_samples]
        expanded_predictions = [sample.prediction for sample in expanded_samples]
        expanded_truths = [sample.truth for sample in expanded_samples]
        frozen_exact = (
            sum(sample.correct for sample in frozen_samples) / len(frozen_samples)
            if frozen_samples
            else 0.0
        )
        expanded_exact = (
            sum(sample.correct for sample in expanded_samples) / len(expanded_samples)
            if expanded_samples
            else 0.0
        )
        frozen_payload = {
            "exact_match": frozen_exact,
            "cer": compute_cer(frozen_predictions, frozen_truths) if frozen_samples else 0.0,
            "invalid_format_rate": (
                sum(not sample.valid_format for sample in frozen_samples) / len(frozen_samples)
                if frozen_samples
                else 0.0
            ),
            "sample_count": len(frozen_samples),
            "samples": [asdict(sample) for sample in frozen_samples],
        }
        expanded_payload = {
            "exact_match": expanded_exact,
            "cer": compute_cer(expanded_predictions, expanded_truths) if expanded_samples else 0.0,
            "invalid_format_rate": (
                sum(not sample.valid_format for sample in expanded_samples) / len(expanded_samples)
                if expanded_samples
                else 0.0
            ),
            "sample_count": len(expanded_samples),
            "samples": [asdict(sample) for sample in expanded_samples],
        }
        return frozen_payload, expanded_payload

    candidate_frozen, candidate_expanded = _split_report(candidate_report)
    paddle_frozen, paddle_expanded = _split_report(paddle_report)

    combined_cer = 0.0
    if candidate_report is not None:
        combined_cer = candidate_report.cer
    elif paddle_report is not None:
        combined_cer = paddle_report.cer

    frozen_exact = candidate_frozen["exact_match"] if candidate_frozen is not None else 0.0
    expanded_exact = candidate_expanded["exact_match"] if candidate_expanded is not None else 0.0
    gate = deployment_gate(
        frozen_exact=frozen_exact,
        expanded_exact=expanded_exact,
        cer=combined_cer,
        candidate_p95_ms=(
            candidate_report.warm_p95_latency_ms if candidate_report is not None else None
        ),
        paddle_p95_ms=paddle_report.warm_p95_latency_ms if paddle_report is not None else None,
    )

    return {
        "engine": engine,
        "frozen_manifest": str(Path(frozen_manifest).resolve()),
        "expanded_manifest": str(Path(expanded_manifest).resolve()),
        "candidate": {
            "report": _report_to_dict(candidate_report) if candidate_report is not None else None,
            "frozen": candidate_frozen,
            "expanded": candidate_expanded,
        },
        "paddle": {
            "report": _report_to_dict(paddle_report) if paddle_report is not None else None,
            "frozen": paddle_frozen,
            "expanded": paddle_expanded,
        },
        "combined_cer": combined_cer,
        "passed": gate.passed,
        "reasons": gate.reasons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--expanded-manifest", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--engine", choices=("paddle", "candidate", "both"), default="both")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    try:
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)
    except ImportError:
        pass

    report = run_benchmark(
        frozen_manifest=args.frozen_manifest,
        expanded_manifest=args.expanded_manifest,
        engine=args.engine,
        warmup_runs=args.warmup_runs,
        candidate_model_path=args.candidate,
        candidate_metadata_path=args.metadata,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "reasons": report["reasons"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
