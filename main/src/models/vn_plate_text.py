"""Pure helpers shared by Vietnamese license-plate recognizers.

The functions in this module deliberately depend only on NumPy and OpenCV so
they can be used by both model training tools and the low-end CPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import cv2
import numpy as np


_PLATE_FORMAT = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$", re.ASCII)


@dataclass(frozen=True, slots=True)
class PlateReading:
    """An immutable recognition result with confidence in the unit interval."""

    text: str
    confidence: float


def normalize_plate_text(text: object) -> str:
    """Return uppercase ASCII alphanumerics without guessing OCR characters.

    Punctuation and whitespace are formatting, so they are removed.  Characters
    such as ``O`` and ``I`` remain letters rather than being changed to digits.
    """

    if text is None:
        return ""
    return "".join(
        character.upper()
        for character in str(text)
        if character.isascii() and character.isalnum()
    )


def validate_vietnamese_plate(text: object) -> bool:
    """Whether *text* has the structural ASCII Vietnamese plate format."""

    return bool(_PLATE_FORMAT.fullmatch(str(text)))


def _as_bgr_uint8(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    array = np.asarray(image)
    if array.size == 0:
        return None
    if array.ndim not in (2, 3):
        raise ValueError("plate crop must be a two- or three-dimensional array")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("plate crop must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError("plate crop must contain only finite values")

    if np.issubdtype(array.dtype, np.floating) and array.min() >= 0.0 and array.max() <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.shape[2] == 1:
        array = cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    elif array.shape[2] != 3:
        raise ValueError("plate crop must have 1, 3, or 4 channels")
    return np.ascontiguousarray(array)


def _fit_with_padding(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, min(width, round(image.shape[1] * scale)))
    resized_height = max(1, min(height, round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


def normalize_plate_crop(
    image: np.ndarray | None,
    output_size: Sequence[int] = (192, 64),
) -> np.ndarray:
    """Convert a plate crop into a padded BGR recognition strip.

    Compact, near-square crops are treated as two-row plates.  Their top and
    bottom rows are placed left-to-right, preserving Vietnamese reading order.
    ``output_size`` follows OpenCV's ``(width, height)`` convention.
    """

    if len(output_size) != 2:
        raise ValueError("output_size must contain width and height")
    width, height = (int(output_size[0]), int(output_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("output_size dimensions must be positive")

    crop = _as_bgr_uint8(image)
    if crop is None or crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    crop_height, crop_width = crop.shape[:2]
    if crop_height >= 2 and crop_width / crop_height < 2.0:
        split = crop_height // 2
        top = crop[:split]
        bottom = crop[split:]
        row_height = max(top.shape[0], bottom.shape[0])
        if top.shape[0] != row_height:
            top = cv2.copyMakeBorder(top, 0, row_height - top.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        if bottom.shape[0] != row_height:
            bottom = cv2.copyMakeBorder(
                bottom, 0, row_height - bottom.shape[0], 0, 0, cv2.BORDER_CONSTANT
            )
        crop = np.concatenate((top, bottom), axis=1)

    return _fit_with_padding(crop, width, height)


def greedy_ctc_decode(
    logits: np.ndarray,
    vocab: str | Sequence[str],
    blank_index: int | None = None,
) -> PlateReading:
    """Greedily decode ``[time, classes]`` logits using standard CTC collapse."""

    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 2:
        raise ValueError("logits must have shape [time, classes] or [1, time, classes]")

    symbols = tuple(vocab)
    blank = len(symbols) if blank_index is None else int(blank_index)
    if blank != len(symbols):
        raise ValueError("blank index must equal vocabulary size (blank-last CTC)")
    if blank < 0 or blank >= scores.shape[1]:
        raise ValueError("blank index is outside the logits classes")
    if scores.shape[1] != len(symbols) + 1:
        raise ValueError("logits classes must equal vocabulary size plus one blank")
    if scores.shape[0] == 0:
        return PlateReading("", 0.0)
    if not np.isfinite(scores).all():
        raise ValueError("logits must contain only finite values")

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    best = np.argmax(probabilities, axis=1)

    characters: list[str] = []
    selected_probabilities: list[float] = []
    previous: int | None = None
    for timestep, class_index in enumerate(best):
        index = int(class_index)
        if index != previous and index != blank:
            characters.append(symbols[index])
            selected_probabilities.append(float(probabilities[timestep, index]))
        previous = index

    confidence = float(np.mean(selected_probabilities)) if selected_probabilities else 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return PlateReading("".join(characters), confidence)


__all__ = [
    "PlateReading",
    "greedy_ctc_decode",
    "normalize_plate_crop",
    "normalize_plate_text",
    "validate_vietnamese_plate",
]
