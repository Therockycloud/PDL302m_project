#!/usr/bin/env python3
"""Train and export the Vietnamese plate CTC recognizer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from src.datasets.plate_ocr_dataset import (  # noqa: E402
    PlateManifestRow,
    compose_plate_manifests,
    load_plate_manifest,
)
from src.models.vn_plate_ctc import (  # noqa: E402
    BLANK_INDEX,
    VOCABULARY,
    VnPlateCTC,
    ctc_loss_for_batch,
)
from src.models.vn_plate_text import normalize_plate_crop  # noqa: E402

INPUT_WIDTH = 192
INPUT_HEIGHT = 64
MEAN = [0.0, 0.0, 0.0]
STD = [1.0, 1.0, 1.0]


def checkpoint_rank(metrics: dict[str, float | int]) -> tuple[float, float, int]:
    """Rank validation metrics without consulting any reserved test set."""

    return (
        float(metrics["val_exact_match"]),
        -float(metrics["val_cer"]),
        -int(metrics["epoch"]),
    )


def calibrate_sequence_confidence(
    *,
    confidences: list[float],
    correct: list[bool],
    minimum_precision: float = 0.99,
) -> dict[str, float | int | bool | None]:
    """Find the lowest observed threshold meeting precision on validation."""

    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must have the same non-zero length")
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in confidences):
        raise ValueError("confidences must be finite values in [0, 1]")

    total = len(correct)
    for threshold in sorted(set(confidences)):
        selected = [index for index, value in enumerate(confidences) if value >= threshold]
        support = len(selected)
        precision = sum(bool(correct[index]) for index in selected) / support
        if precision >= minimum_precision:
            return {
                "threshold": float(threshold),
                "precision": float(precision),
                "coverage": support / total,
                "support": support,
                "total": total,
                "available": True,
            }
    return {
        "threshold": None,
        "precision": None,
        "coverage": 0.0,
        "support": 0,
        "total": total,
        "available": False,
    }


def seed_everything(seed: int) -> None:
    """Seed supported RNGs and select deterministic Torch algorithms."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _RowsDataset(Dataset):
    def __init__(self, rows: Sequence[PlateManifestRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        row = self.rows[index]
        image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode plate image: {row.image_path}")
        strip = normalize_plate_crop(image, output_size=(INPUT_WIDTH, INPUT_HEIGHT))
        chw = np.ascontiguousarray(strip.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return chw, row.label


def _load_isolated_rows(
    train_manifests: Sequence[str | Path],
    validation_manifest: str | Path,
    reserved_manifests: Sequence[str | Path],
) -> tuple[list[PlateManifestRow], list[PlateManifestRow]]:
    """Load selection inputs and fail closed on held-out collisions."""

    if not train_manifests:
        raise ValueError("at least one training manifest is required")
    for manifest in train_manifests:
        rows = load_plate_manifest(manifest)
        if not rows or any(row.split != "train" for row in rows):
            raise ValueError(f"training manifest must contain only train rows: {manifest}")
    validation_source = load_plate_manifest(validation_manifest)
    if not validation_source or any(
        row.split != "val" or row.source_type != "real" or not row.verified
        for row in validation_source
    ):
        raise ValueError(
            "validation manifest must contain only manually verified real val rows"
        )

    all_reserved = [validation_manifest, *reserved_manifests]
    train_rows = compose_plate_manifests(
        train_manifests, split="train", reserved_manifests=all_reserved
    )
    validation_rows = compose_plate_manifests(
        [validation_manifest], split="val", reserved_manifests=reserved_manifests
    )
    if not train_rows or not validation_rows:
        raise ValueError("training and validation selections must be non-empty")
    return train_rows, validation_rows


def greedy_decode_logits(logits: torch.Tensor) -> tuple[list[str], list[float]]:
    """Decode logits using the same mean emitted-token confidence as runtime."""

    if logits.ndim != 3 or logits.shape[2] != len(VOCABULARY) + 1:
        raise ValueError("logits must have shape [T, N, 37]")
    probabilities = logits.softmax(dim=2)
    scores, indices = probabilities.max(dim=2)
    texts: list[str] = []
    confidences: list[float] = []
    for batch_index in range(logits.shape[1]):
        previous = BLANK_INDEX
        characters: list[str] = []
        emitted: list[float] = []
        for time_index in range(logits.shape[0]):
            current = int(indices[time_index, batch_index])
            if current != BLANK_INDEX and current != previous:
                characters.append(VOCABULARY[current])
                emitted.append(float(scores[time_index, batch_index]))
            previous = current
        texts.append("".join(characters))
        confidences.append(float(np.mean(emitted)) if emitted else 0.0)
    return texts, confidences


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _evaluate(
    model: VnPlateCTC, loader: DataLoader
) -> tuple[float, float, list[float], list[bool]]:
    model.eval()
    predictions: list[str] = []
    truths: list[str] = []
    confidences: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            decoded, batch_confidences = greedy_decode_logits(model(images.float()))
            predictions.extend(decoded)
            truths.extend(labels)
            confidences.extend(batch_confidences)
    correct = [prediction == truth for prediction, truth in zip(predictions, truths)]
    exact = sum(correct) / len(truths)
    cer = sum(_edit_distance(prediction, truth) for prediction, truth in zip(predictions, truths)) / sum(
        len(truth) for truth in truths
    )
    return exact, cer, confidences, correct


def _dependency_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    for package in ("torchvision", "onnx", "onnxruntime", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def run_smoke_training(*, seed: int, steps: int) -> list[float]:
    """Run a tiny deterministic head-training loop used by the smoke test."""

    if steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(seed)
    model = VnPlateCTC().eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    images = torch.rand(2, 3, 64, 192)
    labels = ["30A12345", "30A12345"]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = ctc_loss_for_batch(model, images, labels)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    return losses


def export_fixed_batch_onnx(
    model: VnPlateCTC,
    images: torch.Tensor,
    target: str | Path,
) -> torch.Tensor:
    """Export a fixed-batch ONNX graph and return its reference Torch logits."""

    import onnx

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    with torch.no_grad():
        expected = model(images).detach().cpu()
    torch.onnx.export(
        model,
        (images,),
        str(target),
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(target)))
    import onnxruntime as ort

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": images.detach().cpu().numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-4)
    return expected


def train(
    *,
    train_manifests: Sequence[str | Path],
    validation_manifest: str | Path,
    reserved_manifests: Sequence[str | Path],
    output_dir: str | Path,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    patience: int = 5,
    num_workers: int = 0,
    threads: int = 4,
) -> dict[str, object]:
    """Train using train rows and select/calibrate only on verified real validation."""

    if epochs < 1 or batch_size < 1 or patience < 1 or threads < 1:
        raise ValueError("epochs, batch_size, patience and threads must be positive")
    seed_everything(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    train_rows, validation_rows = _load_isolated_rows(
        train_manifests, validation_manifest, reserved_manifests
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _RowsDataset(train_rows), batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=num_workers, worker_init_fn=_seed_worker,
    )
    validation_loader = DataLoader(
        _RowsDataset(validation_rows), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=_seed_worker,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    model = VnPlateCTC()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float | int] | None = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = ctc_loss_for_batch(model, images.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        exact, cer, _confidences, _correct = _evaluate(model, validation_loader)
        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_exact_match": exact,
            "val_cer": cer,
        }
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if best_metrics is None or checkpoint_rank(metrics) > checkpoint_rank(best_metrics):
            best_metrics = metrics
            stale_epochs = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics},
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    assert best_metrics is not None
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    exact, cer, confidences, correct = _evaluate(model, validation_loader)
    calibration = calibrate_sequence_confidence(
        confidences=confidences, correct=correct, minimum_precision=0.99
    )
    onnx_path = output / "vn_plate_recognizer.onnx"
    export_fixed_batch_onnx(
        model, torch.zeros(batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH), onnx_path
    )

    config = {
        "seed": seed, "epochs_requested": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "patience": patience,
        "num_workers": num_workers, "threads": threads,
    }
    history_payload = {
        "selection_source": "real_validation_only",
        "history": history,
        "best_epoch": int(best_metrics["epoch"]),
        "best_val_exact_match": exact,
        "best_val_cer": cer,
    }
    (output / "training_history.json").write_text(
        json.dumps(history_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot([item["epoch"] for item in history], [item["train_loss"] for item in history])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="CTC loss")
    axes[1].plot([item["epoch"] for item in history], [item["val_exact_match"] for item in history], label="Exact")
    axes[1].plot([item["epoch"] for item in history], [item["val_cer"] for item in history], label="CER")
    axes[1].set(title="Real validation", xlabel="Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=140)
    plt.close(figure)

    metadata = {
        "schema": "vn-plate-ctc-v1",
        "vocabulary": VOCABULARY,
        "blank_index": BLANK_INDEX,
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "input_tensor_shape": [batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH],
        "mean": MEAN, "std": STD,
        "fixed_batch_size": batch_size,
        "recommended_lock_conf": calibration["threshold"],
        "confidence_calibration": calibration,
        "deployment_ready": bool(calibration["available"]),
        "model_sha256": _sha256(checkpoint_path),
        "onnx_sha256": _sha256(onnx_path),
        "train_manifests": [
            {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for path in train_manifests
        ],
        "validation_manifest": {
            "path": str(Path(validation_manifest).resolve()),
            "sha256": _sha256(validation_manifest),
        },
        "reserved_manifests": [
            {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for path in reserved_manifests
        ],
        "best_metrics": best_metrics,
        "config": config,
        "dependencies": _dependency_versions(),
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    (output / "vn_plate_recognizer.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", action="append", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--reserved-manifest", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = train(
        train_manifests=args.train_manifest,
        validation_manifest=args.validation_manifest,
        reserved_manifests=args.reserved_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=args.num_workers,
        threads=args.threads,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
