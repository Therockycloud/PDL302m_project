"""Reserved-label exclusion with prefix and near-duplicate guards."""

from __future__ import annotations


def levenshtein_distance(left: str, right: str) -> int:
    """Return the edit distance between two strings."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_char in enumerate(right, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            replace_cost = previous[col_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def is_near_reserved_label(draft: str, reserved_labels: set[str]) -> bool:
    """Return True when *draft* should be dropped as reserved or near-reserved."""

    return classify_reserved_label_drop(draft, reserved_labels) is not None


def classify_reserved_label_drop(draft: str, reserved_labels: set[str]) -> str | None:
    """Return a drop reason or None when the draft is safe to keep."""

    if draft in reserved_labels:
        return "reserved-label"

    for reserved in reserved_labels:
        if len(draft) >= 5 and reserved.startswith(draft):
            return "near-reserved-label"
        if len(reserved) >= 5 and draft.startswith(reserved):
            return "near-reserved-label"
        if abs(len(draft) - len(reserved)) <= 1 and levenshtein_distance(draft, reserved) <= 1:
            return "near-reserved-label"
    return None
