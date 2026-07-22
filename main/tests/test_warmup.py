"""WS-1 Task 5: model warmup at load time.

Runs every model once on a throwaway frame right after construction so the
FIRST REAL vehicle frame doesn't pay model cold-start latency (lazy kernel
compilation, ONNX session warmup, first-call JIT, etc.). Each model is
warmed in its own try/except so one model's failure never blocks the
others or crashes app startup.
"""
import numpy as np
import pytest

from src.utils.warmup import warmup_models


@pytest.fixture(autouse=True)
def _enable_warmup(monkeypatch):
    """These tests exercise the warmup itself — neutralize the Docker/compose
    DPL_DISABLE_WARMUP opt-out so they behave identically in every environment."""
    monkeypatch.delenv("DPL_DISABLE_WARMUP", raising=False)


class _OKVehicleDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return []


class _OKPlateDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return []


class _OKColorClf:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return ("WHITE", 0.5)


class _OKOcr:
    def __init__(self) -> None:
        self.calls = 0

    def read_plate(self, image):
        self.calls += 1
        return ""


class _RaisingModel:
    """Any of the four model kinds — always raises on its call."""

    def detect(self, *_a, **_k):
        raise RuntimeError("boom")

    def predict(self, *_a, **_k):
        raise RuntimeError("boom")

    def read_plate(self, *_a, **_k):
        raise RuntimeError("boom")


def test_warmup_calls_each_available_model_once():
    vehicle_det = _OKVehicleDetector()
    plate_det = _OKPlateDetector()
    color_clf = _OKColorClf()
    ocr = _OKOcr()

    warmup_models(
        vehicle_detector=vehicle_det,
        plate_detector=plate_det,
        color_clf=color_clf,
        ocr=ocr,
    )

    assert vehicle_det.calls == 1
    assert plate_det.calls == 1
    assert color_clf.calls == 1
    assert ocr.calls == 1


def test_warmup_uses_a_throwaway_320x320_frame():
    seen = {}

    class _CapturingDetector:
        def detect(self, frame):
            seen["frame"] = frame
            return []

    warmup_models(vehicle_detector=_CapturingDetector())

    assert seen["frame"].shape == (320, 320, 3)
    assert seen["frame"].dtype == np.uint8


def test_warmup_one_model_raising_does_not_block_others():
    vehicle_det = _RaisingModel()
    plate_det = _OKPlateDetector()
    color_clf = _OKColorClf()
    ocr = _OKOcr()

    # Must not raise even though vehicle_det blows up.
    warmup_models(
        vehicle_detector=vehicle_det,
        plate_detector=plate_det,
        color_clf=color_clf,
        ocr=ocr,
    )

    assert plate_det.calls == 1
    assert color_clf.calls == 1
    assert ocr.calls == 1


def test_warmup_with_no_models_does_not_raise():
    warmup_models()  # all None/absent — must be a no-op, not an error


def test_warmup_with_none_models_does_not_raise():
    warmup_models(vehicle_detector=None, plate_detector=None, color_clf=None, ocr=None)


def test_warmup_skipped_when_env_set(monkeypatch):
    """DPL_DISABLE_WARMUP=1 must short-circuit warmup before touching any model.

    On a low-RAM VPS, forcing every model to run a throwaway inference at
    Docker startup (especially PaddleOCR's doc-orientation + unwarping
    sub-models) is what OOMs/hangs the container. Setting this env var must
    make warmup_models() a pure no-op.
    """
    monkeypatch.setenv("DPL_DISABLE_WARMUP", "1")

    vehicle_det = _OKVehicleDetector()
    plate_det = _OKPlateDetector()
    color_clf = _OKColorClf()
    ocr = _OKOcr()

    warmup_models(
        vehicle_detector=vehicle_det,
        plate_detector=plate_det,
        color_clf=color_clf,
        ocr=ocr,
    )

    assert vehicle_det.calls == 0
    assert plate_det.calls == 0
    assert color_clf.calls == 0
    assert ocr.calls == 0


def test_warmup_runs_when_env_unset(monkeypatch):
    """Without the env var, warmup must keep its original behaviour (native dev)."""
    monkeypatch.delenv("DPL_DISABLE_WARMUP", raising=False)

    vehicle_det = _OKVehicleDetector()
    plate_det = _OKPlateDetector()
    color_clf = _OKColorClf()
    ocr = _OKOcr()

    warmup_models(
        vehicle_detector=vehicle_det,
        plate_detector=plate_det,
        color_clf=color_clf,
        ocr=ocr,
    )

    assert vehicle_det.calls == 1
    assert plate_det.calls == 1
    assert color_clf.calls == 1
    assert ocr.calls == 1
