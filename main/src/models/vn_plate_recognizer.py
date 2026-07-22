"""ONNX Runtime reader for the Vietnamese plate CTC recognizer."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.models.vn_plate_ctc import BLANK_INDEX, VOCABULARY
from src.models.vn_plate_text import (
    greedy_ctc_decode,
    normalize_plate_crop,
    normalize_plate_text,
    validate_vietnamese_plate,
)

SCHEMA = "vn-plate-ctc-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise RuntimeError(f"metadata file not found: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"metadata is not valid JSON: {metadata_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"metadata must be a JSON object: {metadata_path}")
    return payload


def _validate_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise RuntimeError(f"unsupported metadata schema {schema!r}; expected {SCHEMA!r}")

    vocabulary = payload.get("vocabulary")
    if vocabulary != VOCABULARY:
        raise RuntimeError(
            "metadata vocabulary does not match the recognizer vocabulary "
            f"(expected {VOCABULARY!r}, found {vocabulary!r})"
        )

    blank_index = payload.get("blank_index")
    if blank_index != BLANK_INDEX:
        raise RuntimeError(
            f"metadata blank_index {blank_index!r} is inconsistent with model blank {BLANK_INDEX}"
        )
    if blank_index != len(VOCABULARY):
        raise RuntimeError(
            f"metadata blank_index {blank_index} must equal vocabulary length {len(VOCABULARY)}"
        )

    input_size = payload.get("input_size")
    if not isinstance(input_size, list) or len(input_size) != 2:
        raise RuntimeError("metadata input_size must be a two-element list [width, height]")

    input_tensor_shape = payload.get("input_tensor_shape")
    if not isinstance(input_tensor_shape, list) or len(input_tensor_shape) != 4:
        raise RuntimeError(
            "metadata input_tensor_shape must be a four-element list [N, C, H, W]"
        )

    onnx_sha256 = payload.get("onnx_sha256")
    if not isinstance(onnx_sha256, str) or len(onnx_sha256) != 64:
        raise RuntimeError("metadata onnx_sha256 must be a 64-character hex digest")

    return payload


def greedy_decode_batch_logits(
    logits: np.ndarray,
    *,
    vocab: str = VOCABULARY,
    blank_index: int = BLANK_INDEX,
) -> tuple[list[str], list[float]]:
    """Decode time-major ``[T, N, C]`` logits using greedy CTC collapse."""

    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError(f"logits must have shape [T, N, C], got {scores.shape}")
    if scores.shape[2] != len(vocab) + 1:
        raise ValueError("logits classes must equal vocabulary size plus one blank")

    texts: list[str] = []
    confidences: list[float] = []
    for batch_index in range(scores.shape[1]):
        reading = greedy_ctc_decode(
            scores[:, batch_index, :],
            vocab=vocab,
            blank_index=blank_index,
        )
        texts.append(reading.text)
        confidences.append(reading.confidence)
    return texts, confidences


class VnPlateRecognizer:
    """Serve the exported Vietnamese plate CTC recognizer through ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise RuntimeError(f"ONNX model file not found: {self.model_path}")

        payload = _validate_metadata(_load_metadata(self.metadata_path))
        self._metadata = payload
        self._vocabulary = str(payload["vocabulary"])
        self._blank_index = int(payload["blank_index"])
        self._input_size = (int(payload["input_size"][0]), int(payload["input_size"][1]))
        self._input_tensor_shape = tuple(int(value) for value in payload["input_tensor_shape"])
        self._fixed_batch_size = int(self._input_tensor_shape[0])
        expected_onnx_sha256 = str(payload["onnx_sha256"]).lower()
        actual_onnx_sha256 = _sha256(self.model_path)
        if actual_onnx_sha256 != expected_onnx_sha256:
            raise RuntimeError(
                "ONNX model checksum mismatch: "
                f"expected {expected_onnx_sha256}, found {actual_onnx_sha256}"
            )

        self._session = session
        self.last_diagnostics: dict[str, Any] = {}

    def _ensure_session(self) -> Any:
        if self._session is None:
            import onnxruntime

            self._session = onnxruntime.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def _preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        strip = normalize_plate_crop(bgr_crop, output_size=self._input_size)
        chw = np.ascontiguousarray(strip.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return chw

    def _build_batch_input(self, image_chw: np.ndarray) -> np.ndarray:
        batch = np.zeros(self._input_tensor_shape, dtype=np.float32)
        batch[0] = image_chw
        return batch

    def read_plate(self, bgr_crop: np.ndarray) -> dict[str, float | str]:
        """Recognize plate text from a localized BGR crop."""

        image = self._preprocess(bgr_crop)
        batch = self._build_batch_input(image)
        session = self._ensure_session()
        started = time.perf_counter()
        outputs = session.run(None, {"images": batch})
        latency_ms = (time.perf_counter() - started) * 1000.0

        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.ndim == 3:
            sample_logits = logits[:, 0, :]
        else:
            sample_logits = logits

        reading = greedy_ctc_decode(
            sample_logits,
            vocab=self._vocabulary,
            blank_index=self._blank_index,
        )
        raw_text = normalize_plate_text(reading.text)
        confidence = float(reading.confidence)
        valid = validate_vietnamese_plate(raw_text)
        self.last_diagnostics = {
            "raw_text": raw_text,
            "confidence": confidence,
            "valid": valid,
            "latency_ms": float(latency_ms),
        }
        if not valid:
            return {"text": "", "ocr_conf": confidence}
        return {"text": raw_text, "ocr_conf": confidence}


__all__ = [
    "SCHEMA",
    "VnPlateRecognizer",
    "greedy_decode_batch_logits",
]
