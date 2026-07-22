"""Unit tests for near-reserved label exclusion."""

from __future__ import annotations

import pytest

from scripts.near_reserved_filter import (
    classify_reserved_label_drop,
    is_near_reserved_label,
    levenshtein_distance,
)


RESERVED = {
    "30M71854",
    "51F07973",
    "29A12345",
}


def test_levenshtein_distance_basic():
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("abc", "ab") == 1
    assert levenshtein_distance("kitten", "sitting") == 3


@pytest.mark.parametrize(
    ("draft", "reason"),
    [
        ("30M71854", "reserved-label"),
        ("51F0793", "near-reserved-label"),
        ("51F079731", "near-reserved-label"),
        ("51F07972", "near-reserved-label"),
        ("51F07974", "near-reserved-label"),
        ("51F06973", "near-reserved-label"),
        ("30M7185", "near-reserved-label"),
        ("29A1234", "near-reserved-label"),
    ],
)
def test_near_reserved_drops_truncation_prefix_and_edit_one(draft: str, reason: str):
    assert classify_reserved_label_drop(draft, RESERVED) == reason
    assert is_near_reserved_label(draft, RESERVED) is True


@pytest.mark.parametrize(
    "draft",
    [
        "99Z99999",
        "12B34567",
        "52F97973",
        "ABCD1234",
    ],
)
def test_near_reserved_keeps_distinct_labels(draft: str):
    assert classify_reserved_label_drop(draft, RESERVED) is None
    assert is_near_reserved_label(draft, RESERVED) is False


def test_single_substitution_is_near_reserved():
    assert classify_reserved_label_drop("51G07973", RESERVED) == "near-reserved-label"
    assert classify_reserved_label_drop("51F17973", RESERVED) == "near-reserved-label"


def test_prefix_guard_requires_min_length_five_on_draft_side():
    short_reserved = {"51F07"}
    assert classify_reserved_label_drop("51F", short_reserved) is None
    assert classify_reserved_label_drop("51F07", short_reserved) == "reserved-label"


def test_prefix_drop_when_reserved_is_long_prefix_of_draft():
    assert classify_reserved_label_drop("51F0797", RESERVED) == "near-reserved-label"
    assert classify_reserved_label_drop("51F079", RESERVED) == "near-reserved-label"
    assert classify_reserved_label_drop("51F07", RESERVED) == "near-reserved-label"
