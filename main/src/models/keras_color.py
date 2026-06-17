"""Drop-in TF/Keras colour classifier for the runtime, served out-of-process.

Mirrors :class:`TorchColorClassifier` (``CLASS_NAMES`` + ``predict(bgr) -> (label,
conf)``) but the actual TensorFlow inference runs in a separate process
(``keras_color_worker.py``) launched with the isolated ``dpl-train`` interpreter.
This keeps the project on TF/Keras for the colour classifier — as required by the
DPL302m syllabus / Report 1 — WITHOUT importing TensorFlow into the dashboard
process, where it would crash alongside PaddleOCR.

The worker is persistent (model loaded once); each call ships a cropped image as
a temp PNG path over stdin and reads one JSON line back. Colour inference is
gated (once per parked vehicle), so the pipe round-trip is negligible.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

_CLASSES = ["Black", "Blue", "Brown", "Grey", "Red", "Silver", "White", "Yellow"]
_WORKER = Path(__file__).resolve().parent / "keras_color_worker.py"
_DEFAULT_PY = "/opt/homebrew/Caskroom/miniforge/base/envs/dpl-train/bin/python"


def _resolve_worker_python(explicit: str | None) -> str:
    """Pick the interpreter that has TensorFlow: arg > env > default path."""
    for cand in (explicit, os.environ.get("DPL_TRAIN_PYTHON"), _DEFAULT_PY):
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "No TensorFlow interpreter found for the Keras colour worker. "
        "Set DPL_TRAIN_PYTHON to a python with tensorflow installed."
    )


class KerasColorClassifier:
    """Predict vehicle colour from a BGR crop via an out-of-process Keras model."""

    CLASS_NAMES = _CLASSES

    def __init__(self, model_path: str, worker_python: str | None = None,
                 timeout: float = 30.0) -> None:
        self.timeout = timeout
        self._lock = threading.Lock()
        py = _resolve_worker_python(worker_python)
        env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0",
                   KMP_DUPLICATE_LIB_OK="TRUE")
        self._proc = subprocess.Popen(
            [py, str(_WORKER), str(model_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        # Block until the worker has loaded the model (or died trying).
        ready = self._proc.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            raise RuntimeError("Keras colour worker failed to start.")
        self._tmp = os.path.join(tempfile.gettempdir(), f"kc_{os.getpid()}.png")
        logger.info("KerasColorClassifier worker ready (pid=%s).", self._proc.pid)

    def predict(self, image) -> tuple[str, float]:
        """Return (colour_label, confidence) for a BGR image crop."""
        if self._proc.poll() is not None:
            return "UNKNOWN", 0.0
        with self._lock:
            try:
                cv2.imwrite(self._tmp, image)
                self._proc.stdin.write(json.dumps({"path": self._tmp}) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                resp = json.loads(line)
                return resp.get("label", "UNKNOWN"), float(resp.get("conf", 0.0))
            except Exception:  # noqa: BLE001
                logger.exception("Keras colour worker call failed.")
                return "UNKNOWN", 0.0

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self):
        self.close()
