"""Trainable MobileNetV3-Small CTC model for normalized plate crops."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torchvision import models

from src.models.vn_plate_text import normalize_plate_text

VOCABULARY = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VOCAB = VOCABULARY
BLANK_INDEX = len(VOCABULARY)
NUM_CLASSES = BLANK_INDEX + 1

_CHAR_TO_INDEX = {character: index for index, character in enumerate(VOCABULARY)}
_EXPECTED_DOWNSAMPLING_CONVS = (
    "0.0",
    "1.block.0.0",
    "2.block.1.0",
    "4.block.1.0",
    "9.block.1.0",
)
_HORIZONTAL_STRIDES_TO_REMOVE = ("4.block.1.0", "9.block.1.0")


def _is_graph_capture() -> bool:
    """Whether data-dependent eager validation must be omitted from capture."""

    if torch.compiler.is_compiling() or torch.jit.is_tracing():
        return True
    is_in_onnx_export = getattr(torch.onnx, "is_in_onnx_export", None)
    return bool(is_in_onnx_export is not None and is_in_onnx_export())


def _adapt_mobilenet_v3_small_strides(features: nn.Sequential) -> tuple[str, ...]:
    """Keep the final feature width at 1/8 input width without dropping blocks.

    Torchvision's MobileNetV3-Small normally downsamples both axes by 32. Plate
    text needs a longer CTC sequence, so the last two depthwise convolutions
    continue downsampling height while preserving width. Exact structure checks
    make a torchvision architecture change an explicit error instead of quietly
    producing an unusable six-timestep sequence.
    """

    if len(features) != 13:
        raise RuntimeError(
            "unexpected MobileNetV3-Small structure: expected 13 feature blocks, "
            f"found {len(features)}"
        )

    modules = dict(features.named_modules())
    actual_downsampling = tuple(
        name
        for name, module in features.named_modules()
        if isinstance(module, nn.Conv2d) and module.stride == (2, 2)
    )
    if actual_downsampling != _EXPECTED_DOWNSAMPLING_CONVS:
        raise RuntimeError(
            "unexpected MobileNetV3-Small downsampling structure: "
            f"expected {_EXPECTED_DOWNSAMPLING_CONVS}, found {actual_downsampling}"
        )

    for name in _HORIZONTAL_STRIDES_TO_REMOVE:
        convolution = modules.get(name)
        if not isinstance(convolution, nn.Conv2d) or convolution.groups != convolution.in_channels:
            raise RuntimeError(
                f"unexpected MobileNetV3-Small layer at {name}: expected a depthwise Conv2d"
            )
        convolution.stride = (2, 1)

    return _HORIZONTAL_STRIDES_TO_REMOVE


def encode_labels(labels: Sequence[str]) -> tuple[Tensor, Tensor]:
    """Encode labels as concatenated non-blank targets and per-label lengths."""

    encoded: list[int] = []
    lengths: list[int] = []
    for sample_index, label in enumerate(labels):
        if not isinstance(label, str):
            raise TypeError(f"label {sample_index} must be a string")
        label = normalize_plate_text(label)
        if not label:
            raise ValueError(
                f"label {sample_index} normalizes to empty; empty CTC targets are not supported"
            )

        for character in label:
            try:
                encoded.append(_CHAR_TO_INDEX[character])
            except KeyError as error:
                raise ValueError(
                    f"label {sample_index} contains unsupported character {character!r}"
                ) from error
        lengths.append(len(label))

    if not lengths:
        raise ValueError("labels batch is empty")

    return torch.tensor(encoded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def required_ctc_timesteps(label: str) -> int:
    """Return target length plus separators needed by adjacent repeats."""

    return len(label) + sum(left == right for left, right in zip(label, label[1:]))


def ctc_input_lengths(logits: Tensor) -> Tensor:
    """Create CPU-long lengths from ``[T, N, C]`` logits for CTC compatibility."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [T, N, C], got {tuple(logits.shape)}")
    return torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long)


class VnPlateCTC(nn.Module):
    """MobileNetV3-Small encoder producing time-major plate character logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, projection_size: int = 128) -> None:
        super().__init__()
        if num_classes != NUM_CLASSES:
            raise ValueError(
                f"num_classes must be {NUM_CLASSES} for the fixed vocabulary and blank-last class"
            )
        if projection_size <= 0:
            raise ValueError("projection_size must be positive")

        backbone = models.mobilenet_v3_small(weights=None)
        self.encoder = backbone.features
        self.adapted_stride_names = _adapt_mobilenet_v3_small_strides(self.encoder)
        self.sequence_projection = nn.Sequential(
            nn.Linear(576, projection_size),
            nn.Hardswish(),
        )
        self.classifier = nn.Linear(projection_size, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return logits shaped ``[T, N, C]`` for ``[N, 3, 64, 192]`` crops."""

        self._validate_input(images)
        features = self.encoder(images)
        if features.shape[1] != 576 or features.shape[2] != 2 or features.shape[3] != 24:
            raise RuntimeError(
                "adapted MobileNetV3-Small produced unexpected feature shape "
                f"{tuple(features.shape)}; expected [N, 576, 2, 24]"
            )
        sequence = features.mean(dim=2).transpose(1, 2)
        logits = self.classifier(self.sequence_projection(sequence))
        return logits.permute(1, 0, 2)

    @staticmethod
    def _validate_input(images: Tensor) -> None:
        if not isinstance(images, Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4:
            raise ValueError(f"images must have shape [N, 3, 64, 192], got shape {tuple(images.shape)}")
        if images.shape[0] == 0:
            raise ValueError("images batch is empty")
        if images.shape[1] != 3:
            raise ValueError(f"images must have 3 channels, got {images.shape[1]}")
        if tuple(images.shape[2:]) != (64, 192):
            raise ValueError(f"images must have spatial size 64x192, got {tuple(images.shape[2:])}")
        if images.dtype != torch.float32:
            raise TypeError(f"images must have dtype float32, got {images.dtype}")
        if _is_graph_capture():
            return
        if not torch.isfinite(images).all().item():
            raise ValueError("images must contain only finite values")
        minimum = images.detach().amin().item()
        maximum = images.detach().amax().item()
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"images must be normalized to [0, 1], got range [{minimum}, {maximum}]"
            )


def ctc_loss_for_batch(model: VnPlateCTC, images: Tensor, labels: Sequence[str]) -> Tensor:
    """Compute CTC loss and reject paths impossible for the produced sequence."""

    if len(labels) != images.shape[0]:
        raise ValueError(
            f"labels batch size {len(labels)} does not match image batch size {images.shape[0]}"
        )
    targets, target_lengths = encode_labels(labels)
    normalized_labels = [normalize_plate_text(label) for label in labels]
    logits = model(images)
    # CTCLoss length tensors intentionally stay CPU long for backend compatibility.
    input_lengths = ctc_input_lengths(logits)
    available = logits.shape[0]
    for sample_index, label in enumerate(normalized_labels):
        required = required_ctc_timesteps(label)
        if required > available:
            raise ValueError(
                f"label {sample_index} requires {required} CTC timesteps, but only "
                f"{available} available"
            )

    loss_function = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    return loss_function(
        logits.log_softmax(dim=2),
        targets.to(device=logits.device),
        input_lengths,
        target_lengths,
    )
