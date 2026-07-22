"""Behavior tests for the trainable Vietnamese plate CTC model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.vn_plate_ctc import (  # noqa: E402
    BLANK_INDEX,
    VOCAB,
    VOCABULARY,
    VnPlateCTC,
    ctc_input_lengths,
    ctc_loss_for_batch,
    encode_labels,
    required_ctc_timesteps,
)
from scripts.train_vn_plate_ocr import (  # noqa: E402
    calibrate_sequence_confidence,
    checkpoint_rank,
    export_fixed_batch_onnx,
    run_smoke_training,
)


def _assert_all_trainable_gradients_are_finite(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_model_outputs_time_major_character_logits():
    model = VnPlateCTC(num_classes=37).eval()

    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 64, 192))

    assert logits.ndim == 3
    assert logits.shape[1:] == (2, 37)
    assert logits.shape[0] >= 20
    assert logits.shape[0] == 24


def test_full_encoder_adapts_only_late_horizontal_downsampling():
    model = VnPlateCTC()

    assert len(model.encoder) == 13
    adapted = model.adapted_stride_names
    assert len(adapted) == 2
    assert all(dict(model.encoder.named_modules())[name].stride == (2, 1) for name in adapted)


def test_seeded_eval_outputs_are_deterministic():
    torch.manual_seed(3)
    first = VnPlateCTC().eval()
    torch.manual_seed(3)
    second = VnPlateCTC().eval()
    images = torch.rand(1, 3, 64, 192)

    with torch.no_grad():
        first_output = first(images)
        repeated_output = first(images)
        second_output = second(images)

    assert first_output.shape == (24, 1, 37)
    torch.testing.assert_close(repeated_output, first_output)
    torch.testing.assert_close(second_output, first_output)


def test_model_can_be_captured_with_torch_export():
    model = VnPlateCTC().eval()
    images = torch.rand(1, 3, 64, 192)

    exported = torch.export.export(model, (images,))
    logits = exported.module()(images)

    assert logits.shape == (24, 1, 37)


@pytest.mark.parametrize(
    "images, error",
    [
        (torch.zeros(3, 64, 192), "shape"),
        (torch.zeros(1, 1, 64, 192), "channels"),
        (torch.zeros(1, 3, 32, 192), "64x192"),
        (torch.zeros(1, 3, 64, 192, dtype=torch.float64), "float32"),
        (torch.full((1, 3, 64, 192), -0.1), r"\[0, 1\]"),
        (torch.full((1, 3, 64, 192), 1.1), r"\[0, 1\]"),
    ],
)
def test_model_rejects_invalid_inputs(images, error):
    with pytest.raises((TypeError, ValueError), match=error):
        VnPlateCTC()(images)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_model_rejects_non_finite_inputs(value):
    images = torch.zeros(1, 3, 64, 192)
    images[0, 0, 0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        VnPlateCTC()(images)


def test_model_rejects_empty_image_batch():
    with pytest.raises(ValueError, match="batch.*empty"):
        VnPlateCTC()(torch.zeros(0, 3, 64, 192))


def test_training_rejects_empty_batch():
    with pytest.raises(ValueError, match="batch.*empty"):
        ctc_loss_for_batch(VnPlateCTC(), torch.zeros(0, 3, 64, 192), [])


def test_encoding_uses_blank_last_vocabulary():
    targets, lengths = encode_labels(["0Z", "A1"])

    assert VOCABULARY == "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert VOCAB == VOCABULARY
    assert BLANK_INDEX == 36
    assert targets.dtype == torch.long
    assert targets.tolist() == [0, 35, 10, 1]
    assert lengths.tolist() == [2, 2]
    assert BLANK_INDEX not in targets.tolist()


def test_encoding_normalizes_plate_formatting_and_whitespace():
    targets, lengths = encode_labels(["  30a-123.45\t"])

    assert targets.tolist() == [3, 0, 10, 1, 2, 3, 4, 5]
    assert lengths.tolist() == [8]


def test_encoding_rejects_label_that_normalizes_to_empty():
    with pytest.raises(ValueError, match="normalizes to empty"):
        encode_labels([" - . \t"])


def test_encoding_rejects_empty_and_non_string_labels():
    with pytest.raises(ValueError, match="empty"):
        encode_labels([""])
    with pytest.raises(TypeError, match="string"):
        encode_labels([None])


def test_adjacent_repeats_consume_extra_ctc_timesteps():
    assert required_ctc_timesteps("AABCC") == 7
    assert required_ctc_timesteps("ABACA") == 5


def test_ctc_input_lengths_are_derived_from_logit_time_dimension():
    logits = torch.zeros(24, 3, 37)

    lengths = ctc_input_lengths(logits)

    assert lengths.dtype == torch.long
    assert lengths.tolist() == [24, 24, 24]


def test_loss_rejects_target_whose_repeat_path_exceeds_available_time():
    model = VnPlateCTC().eval()
    images = torch.zeros(1, 3, 64, 192)

    with pytest.raises(ValueError, match=r"requires 25 CTC timesteps.*24 available"):
        ctc_loss_for_batch(model, images, ["A" * 13])


def test_tiny_batch_backpropagates():
    torch.manual_seed(7)
    model = VnPlateCTC(num_classes=37)
    images = torch.rand(2, 3, 64, 192)
    labels = ["30A12345", "51B67890"]

    loss = ctc_loss_for_batch(model, images, labels)
    loss.backward()

    assert torch.isfinite(loss)
    _assert_all_trainable_gradients_are_finite(model)


def test_feasible_repeated_target_backpropagates():
    torch.manual_seed(17)
    model = VnPlateCTC()
    images = torch.rand(1, 3, 64, 192)

    loss = ctc_loss_for_batch(model, images, ["AABB1122"])
    loss.backward()

    assert torch.isfinite(loss)
    _assert_all_trainable_gradients_are_finite(model)


def test_state_dict_roundtrip_preserves_eval_output():
    torch.manual_seed(11)
    source = VnPlateCTC().eval()
    restored = VnPlateCTC().eval()
    restored.load_state_dict(source.state_dict())
    images = torch.rand(1, 3, 64, 192)

    with torch.no_grad():
        expected = source(images)
        actual = restored(images)

    torch.testing.assert_close(actual, expected)


def test_deterministic_smoke_training_reduces_ctc_loss():
    first = run_smoke_training(seed=23, steps=6)
    second = run_smoke_training(seed=23, steps=6)

    assert first[-1] < first[0]
    assert second == pytest.approx(first, rel=0.0, abs=0.0)


def test_fixed_batch_onnx_logits_match_torch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(29)
    model = VnPlateCTC().eval()
    images = torch.rand(1, 3, 64, 192)
    target = tmp_path / "plate.onnx"

    expected = export_fixed_batch_onnx(model, images, target)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": images.numpy()})[0]

    assert target.is_file()
    torch.testing.assert_close(
        torch.from_numpy(actual), expected, rtol=1e-3, atol=1e-4
    )


def test_checkpoint_rank_uses_exact_then_cer_then_earlier_epoch():
    epochs = [
        {"epoch": 3, "val_exact_match": 0.8, "val_cer": 0.1},
        {"epoch": 2, "val_exact_match": 0.8, "val_cer": 0.1},
        {"epoch": 1, "val_exact_match": 0.8, "val_cer": 0.2},
        {"epoch": 4, "val_exact_match": 0.7, "val_cer": 0.01},
    ]

    assert max(epochs, key=checkpoint_rank)["epoch"] == 2


def test_confidence_calibration_selects_lowest_threshold_at_99_percent_precision():
    result = calibrate_sequence_confidence(
        confidences=[0.2, 0.4, 0.6, 0.8],
        correct=[False, True, True, True],
        minimum_precision=0.99,
    )

    assert result == {
        "threshold": 0.4,
        "precision": 1.0,
        "coverage": 0.75,
        "support": 3,
        "total": 4,
        "available": True,
    }


def test_confidence_calibration_is_explicitly_unavailable_without_correct_predictions():
    result = calibrate_sequence_confidence(
        confidences=[0.2, 0.8], correct=[False, False], minimum_precision=0.99
    )

    assert result["available"] is False
    assert result["threshold"] is None
    assert result["coverage"] == 0.0
    assert result["support"] == 0
