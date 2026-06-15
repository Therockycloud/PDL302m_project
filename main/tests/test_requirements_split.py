"""Guards the runtime/training dependency split (WS4).

TensorFlow conflicts with PaddleOCR at runtime, so it must live only in the
training requirements, never the runtime set the dashboard installs.
"""
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]  # .../main


def _read(name: str) -> str:
    return (MAIN / name).read_text(encoding="utf-8").lower()


def test_tensorflow_absent_from_runtime_requirements():
    assert "tensorflow" not in _read("requirements.txt")


def test_tensorflow_present_in_train_requirements():
    assert "tensorflow" in _read("requirements-train.txt")


def test_dashboard_has_no_toplevel_tensorflow_import():
    src = (MAIN / "src" / "ui" / "dashboard.py").read_text(encoding="utf-8")
    assert "import tensorflow" not in src
    assert "tf.keras" not in src
