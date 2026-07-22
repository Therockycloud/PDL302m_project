import cv2
import numpy as np
import pytest

from src.models.vn_plate_text import (
    PlateReading,
    greedy_ctc_decode,
    normalize_plate_crop,
    normalize_plate_text,
    validate_vietnamese_plate,
)


VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def fake_logits_for_indices(indices, class_count=37):
    logits = np.full((len(indices), class_count), -8.0, dtype=np.float32)
    logits[np.arange(len(indices)), indices] = 8.0
    return logits


def fixture_two_line_plate():
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (112, 31), (0, 0, 255), thickness=-1)
    cv2.rectangle(image, (8, 47), (112, 71), (255, 0, 0), thickness=-1)
    return image


def test_normalization_keeps_only_uppercase_ascii_alphanumerics():
    assert normalize_plate_text(" 30m-718.54 ") == "30M71854"
    assert normalize_plate_text("3O-0I") == "3O0I"
    assert normalize_plate_text("30M_71/854!") == "30M71854"


def test_normalization_does_not_transliterate_non_ascii_characters():
    assert normalize_plate_text("３０M-71854") == "M71854"
    assert normalize_plate_text("30M-71é854") == "30M71854"
    assert normalize_plate_text("30ß-71854") == "3071854"


@pytest.mark.parametrize(
    "text",
    ["30M71854", "51F12345", "29A112345", "59S212345", "80A12345", "30I12345"],
)
def test_valid_vietnamese_plate_formats(text):
    assert validate_vietnamese_plate(text) is True


def test_two_letter_plate_series_is_valid():
    assert validate_vietnamese_plate("30AB12345") is True


@pytest.mark.parametrize(
    "text",
    [
        "VF3",
        "3O71854",
        "30M71O54",
        "30M123",
        "30M1234567",
        "30M-718.54",
        "30Á12345",
        "３０M12345",
    ],
)
def test_invalid_plate_is_not_lockable(text):
    assert validate_vietnamese_plate(text) is False


def test_validator_rejects_separators_and_does_not_substitute():
    assert validate_vietnamese_plate("30M-718.54") is False
    assert validate_vietnamese_plate("3OM-718.54") is False


def test_plate_reading_is_immutable():
    reading = PlateReading(text="30M71854", confidence=0.9)
    with pytest.raises((AttributeError, TypeError)):
        reading.text = "51F12345"


def test_single_line_crop_is_safely_padded_to_target_shape():
    image = np.full((20, 100), 127, dtype=np.uint8)
    strip = normalize_plate_crop(image, output_size=(192, 64))
    assert strip.shape == (64, 192, 3)
    assert strip.dtype == np.uint8


def test_invalid_crop_returns_a_safe_blank_strip():
    strip = normalize_plate_crop(np.empty((0, 0, 3), dtype=np.uint8), output_size=(192, 64))
    assert strip.shape == (64, 192, 3)
    assert np.count_nonzero(strip) == 0


def test_float_grayscale_values_are_clipped_before_uint8_conversion():
    image = np.array([[-1.0, 300.0, 128.0, 5.0, 9.0], [0.0, 255.0, 64.0, 3.0, 7.0]])
    strip = normalize_plate_crop(image, output_size=(5, 2))
    assert strip[:, :, 0].tolist() == [[0, 255, 128, 5, 9], [0, 255, 64, 3, 7]]
    assert np.array_equal(strip[:, :, 0], strip[:, :, 1])
    assert np.array_equal(strip[:, :, 1], strip[:, :, 2])


def test_unit_float_grayscale_is_scaled_to_full_uint8_range():
    image = np.array([[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
    strip = normalize_plate_crop(image, output_size=(5, 2))
    assert set(np.unique(strip)) == {0, 255}


def test_float_bgra_uses_same_scaling_before_channel_conversion():
    image = np.empty((2, 5, 4), dtype=np.float32)
    image[:] = [0.0, 0.5, 1.0, 0.25]
    strip = normalize_plate_crop(image, output_size=(5, 2))
    assert np.all(strip[:, :, 0] == 0)
    assert np.all((strip[:, :, 1] == 127) | (strip[:, :, 1] == 128))
    assert np.all(strip[:, :, 2] == 255)


def test_one_channel_numeric_input_is_clipped_and_converted_to_bgr():
    image = np.array([[[-10], [300], [12], [13], [14]], [[1], [2], [3], [4], [5]]])
    strip = normalize_plate_crop(image, output_size=(5, 2))
    assert strip[0, 0].tolist() == [0, 0, 0]
    assert strip[0, 1].tolist() == [255, 255, 255]


def test_uint8_bgr_input_is_preserved():
    image = np.full((2, 5, 3), [10, 20, 30], dtype=np.uint8)
    assert np.array_equal(normalize_plate_crop(image, output_size=(5, 2)), image)


@pytest.mark.parametrize(
    "image, message",
    [
        (np.zeros((2, 5, 2), dtype=np.uint8), "channels"),
        (np.full((2, 5), np.nan), "finite"),
        (np.full((2, 5), "plate"), "numeric"),
    ],
)
def test_invalid_crop_arrays_are_rejected_clearly(image, message):
    with pytest.raises(ValueError, match=message):
        normalize_plate_crop(image, output_size=(5, 2))


def test_two_line_plate_is_stacked_in_reading_order():
    image = fixture_two_line_plate()
    strip = normalize_plate_crop(image, output_size=(192, 64))
    assert strip.shape == (64, 192, 3)
    red = np.count_nonzero((strip[:, :, 2] > 180) & (strip[:, :, 0] < 80))
    blue = np.count_nonzero((strip[:, :, 0] > 180) & (strip[:, :, 2] < 80))
    assert red > 0 and blue > 0
    assert np.mean(np.argwhere((strip[:, :, 2] > 180) & (strip[:, :, 0] < 80))[:, 1]) < np.mean(
        np.argwhere((strip[:, :, 0] > 180) & (strip[:, :, 2] < 80))[:, 1]
    )


def test_ctc_decode_collapses_repeats_and_blanks():
    logits = fake_logits_for_indices([3, 3, 36, 0, 0, 22])
    out = greedy_ctc_decode(logits, VOCAB)
    assert out.text == "30M"
    assert 0.0 <= out.confidence <= 1.0


def test_ctc_repeat_separated_by_blank_is_preserved():
    logits = fake_logits_for_indices([3, 36, 3])
    assert greedy_ctc_decode(logits, VOCAB).text == "33"


def test_ctc_decode_accepts_a_single_item_batch():
    logits = fake_logits_for_indices([5, 1, 16])[None, ...]
    assert greedy_ctc_decode(logits, VOCAB).text == "51G"


def test_ctc_decode_rejects_class_count_mismatch():
    logits = np.zeros((2, len(VOCAB)), dtype=np.float32)
    with pytest.raises(ValueError, match="classes"):
        greedy_ctc_decode(logits, VOCAB)


def test_ctc_decode_rejects_non_last_blank_index():
    logits = fake_logits_for_indices([0])
    with pytest.raises(ValueError, match="blank index must equal vocabulary size"):
        greedy_ctc_decode(logits, VOCAB, blank_index=0)
