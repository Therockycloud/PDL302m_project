"""Tests for the Vietnamese plate ONNX recognizer runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.models.vn_plate_ctc import BLANK_INDEX, VOCABULARY
from src.models.vn_plate_recognizer import (
    SCHEMA,
    VnPlateRecognizer,
    greedy_decode_batch_logits,
)


def _write_metadata(
    tmp_path: Path,
    *,
    onnx_sha256: str,
    schema: str = SCHEMA,
    vocabulary: str = VOCABULARY,
    blank_index: int = BLANK_INDEX,
    batch_size: int = 1,
) -> tuple[Path, Path]:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-onnx-payload")
    metadata_path = tmp_path / "model.json"
    metadata = {
        "schema": schema,
        "vocabulary": vocabulary,
        "blank_index": blank_index,
        "input_size": [192, 64],
        "input_tensor_shape": [batch_size, 3, 64, 192],
        "onnx_sha256": onnx_sha256,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, metadata_path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _char_index(character: str) -> int:
    return VOCABULARY.index(character)


def _logits_for_text(
    text: str,
    *,
    timesteps: int = 24,
    batch_size: int = 1,
    token_logit: float = 1.5,
    blank_logit: float = -1.0,
    other_logit: float = -8.0,
) -> np.ndarray:
    logits = np.full((timesteps, batch_size, len(VOCABULARY) + 1), other_logit, dtype=np.float32)
    logits[:, :, BLANK_INDEX] = token_logit
    for index, character in enumerate(text):
        if index >= timesteps:
            break
        logits[index, :, _char_index(character)] = token_logit
        logits[index, :, BLANK_INDEX] = blank_logit
    return logits


class _FakeSession:
    def __init__(self, logits: np.ndarray) -> None:
        self._logits = logits
        self.last_input: np.ndarray | None = None

    def run(self, _output_names, feed_dict):
        self.last_input = feed_dict["images"]
        return [self._logits]


def sample_crop() -> np.ndarray:
    return np.full((20, 100, 3), 180, dtype=np.uint8)


def test_reader_rejects_wrong_model_checksum(tmp_path):
    model_path, metadata_path = _write_metadata(tmp_path, onnx_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="checksum"):
        VnPlateRecognizer(model_path, metadata_path)


def test_reader_rejects_wrong_schema(tmp_path):
    model_bytes = b"fake-onnx"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(model_bytes)
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "wrong-schema",
                "vocabulary": VOCABULARY,
                "blank_index": BLANK_INDEX,
                "input_size": [192, 64],
                "input_tensor_shape": [1, 3, 64, 192],
                "onnx_sha256": _sha256_bytes(model_bytes),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="schema"):
        VnPlateRecognizer(model_path, metadata_path)


def test_reader_rejects_wrong_vocabulary(tmp_path):
    model_bytes = b"fake-onnx"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(model_bytes)
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "vocabulary": "ABC",
                "blank_index": 3,
                "input_size": [192, 64],
                "input_tensor_shape": [1, 3, 64, 192],
                "onnx_sha256": _sha256_bytes(model_bytes),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="vocabulary"):
        VnPlateRecognizer(model_path, metadata_path)


def test_reader_returns_plate_reader_contract(tmp_path):
    model_path, metadata_path = _write_metadata(
        tmp_path, onnx_sha256=_sha256_bytes(b"fake-onnx-payload")
    )
    logits = _logits_for_text("30M71854")
    fake_session = _FakeSession(logits)
    reader = VnPlateRecognizer(model_path, metadata_path, session=fake_session)
    result = reader.read_plate(sample_crop())
    assert result["text"] == "30M71854"
    assert result["ocr_conf"] == pytest.approx(0.924, rel=1e-2)


def test_invalid_format_returns_empty_text_but_keeps_confidence(tmp_path):
    model_path, metadata_path = _write_metadata(
        tmp_path, onnx_sha256=_sha256_bytes(b"fake-onnx-payload")
    )
    logits = _logits_for_text("INVALID")
    reader = VnPlateRecognizer(model_path, metadata_path, session=_FakeSession(logits))
    result = reader.read_plate(sample_crop())
    assert result == {"text": "", "ocr_conf": pytest.approx(0.924, rel=1e-2)}
    assert reader.last_diagnostics["raw_text"] == "INVALID"
    assert reader.last_diagnostics["valid"] is False
    assert reader.last_diagnostics["confidence"] == pytest.approx(0.924, rel=1e-2)
    assert reader.last_diagnostics["latency_ms"] >= 0.0


def test_fixed_batch_padding_path_uses_only_first_row(tmp_path):
    batch_size = 4
    model_path, metadata_path = _write_metadata(
        tmp_path,
        onnx_sha256=_sha256_bytes(b"fake-onnx-payload"),
        batch_size=batch_size,
    )
    logits = _logits_for_text("30M71854", batch_size=batch_size)
    fake_session = _FakeSession(logits)
    reader = VnPlateRecognizer(model_path, metadata_path, session=fake_session)
    reader.read_plate(sample_crop())
    assert fake_session.last_input is not None
    assert fake_session.last_input.shape == (batch_size, 3, 64, 192)
    assert np.any(fake_session.last_input[0] != 0.0)
    assert np.all(fake_session.last_input[1:] == 0.0)


def test_greedy_decode_batch_logits_matches_runtime_helper():
    logits = _logits_for_text("30M71854", batch_size=2)
    texts, confidences = greedy_decode_batch_logits(logits)
    assert texts == ["30M71854", "30M71854"]
    assert confidences == pytest.approx([0.924, 0.924], rel=1e-2)


def test_reader_raises_when_model_file_missing(tmp_path):
    model_path, metadata_path = _write_metadata(
        tmp_path, onnx_sha256=_sha256_bytes(b"fake-onnx-payload")
    )
    model_path.unlink()
    with pytest.raises(RuntimeError, match="not found"):
        VnPlateRecognizer(model_path, metadata_path)
