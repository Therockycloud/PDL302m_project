"""Controlled evaluation of the anti-plate-swap (colour cross-verification) capability.

This is a SECURITY measurement, not a colour-accuracy measurement (that's
``eval_color_deployed.py``). It answers: "if someone clones a registered
plate and puts it on a DIFFERENT-coloured car, how often does the deployed
system's colour cross-check actually catch it?"

System concept under test (plate-primary, colour as soft cross-check):
    DatabaseMatcher.verify_vehicle(plate, predicted_colour) returns
    AUTHORIZED/ALLOW when plate is registered and colour matches; AUTHORIZED/
    ALLOW_WARN (color_warning=True) when the plate is registered but the
    predicted colour differs from the registered colour (a clone wearing a
    different-coloured body would land here); UNREGISTERED/DENY_ALERT when
    the plate isn't in the DB at all.

This script wires together the REAL deployed pieces — no mocks, no
fabricated numbers:
    - Images: VCoR held-out TEST split, reusing
      ``colab_train_color.py.load_samples`` + ``stratified_split`` (seed=42,
      70/15/15) — the EXACT split used to fine-tune and to evaluate the
      deployed weights in ``eval_color_deployed.py``, so these images are
      genuinely unseen by the model.
    - Colour predictions: ``main.src.models.torch_color.TorchColorClassifier``
      loaded from the DEPLOYED weights
      (``main/data/models/color_MobileNetV3Small.pt``), called via its real
      ``.predict(bgr_image)`` interface (body-crop included internally).
    - Decision logic: ``main.src.utils.matching.DatabaseMatcher.verify_vehicle``
      (unmodified), against a TEMPORARY registration CSV built just for this
      run (same columns as ``main/data/database.csv``:
      license_plate,car_brand,car_color). The temp CSV is written under the
      system tmp dir and is not part of the repo's real database.

Three scenario types, ~200 balanced trials each, fixed RNG seed for
reproducibility (no datetime/unseeded random anywhere):
    1. legitimate    — image of true colour C, plate registered to C.
                        Correct = AUTHORIZED + no warning. A warning here
                        is a FALSE ALARM.
    2. plate_swap    — image of true colour C2, presented with a plate
                        registered to a DIFFERENT colour C1 (C1 != C2),
                        simulating a cloned plate moved onto a different-
                        coloured car. Correct = color_warning True (swap
                        CAUGHT). No warning = MISSED.
    3. unregistered  — a plate that is not in the DB at all. Correct =
                        UNREGISTERED / DENY_ALERT.

All three scenarios use the model's REAL predicted colour (not the ground-
truth label) when calling verify_vehicle, so the measured rates reflect
actual deployed behaviour, including the colour classifier's own mistakes.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python \\
        main/scripts/eval_security.py \\
        --data-dir /Users/konalyn/Downloads/archive

Writes:
    docs/benchmarks/security_eval.json
    docs/benchmarks/security_eval.md

Does NOT modify any runtime code, model weights, or the repo's real
main/data/database.csv, and does not commit anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import cv2

# ---------------------------------------------------------------------------
# Make main/scripts importable (colab_train_color.py has no package
# __init__), and make the repo's main/src importable as `src.*` (matching
# the convention used by main/tests/test_matching.py and main/src itself).
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
_MAIN_DIR = _SCRIPTS_DIR.parent  # .../main
_REPO_ROOT = _MAIN_DIR.parent  # repo root

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MAIN_DIR))

from colab_train_color import (  # noqa: E402
    CLASSES,
    load_samples,
    pick_device,
    set_seed,
    stratified_split,
)
from src.models.torch_color import TorchColorClassifier  # noqa: E402
from src.utils.matching import DatabaseMatcher  # noqa: E402

DEFAULT_WEIGHTS = _MAIN_DIR / "data" / "models" / "color_MobileNetV3Small.pt"
DEFAULT_JSON_OUT = _REPO_ROOT / "docs" / "benchmarks" / "security_eval.json"
DEFAULT_MD_OUT = _REPO_ROOT / "docs" / "benchmarks" / "security_eval.md"

# Fixed seed for reproducibility — every random choice in this script (which
# test images get sampled into each scenario, which colour C1 a plate is
# registered to, plate-number generation) is derived from this constant.
# Matches the SEED used by colab_train_color.py / eval_color_deployed.py for
# the data split itself, so re-running this script reproduces identical
# trial composition AND identical scenario sampling.
SEED = 42

TRIALS_PER_SCENARIO = 200


# ---------------------------------------------------------------------------
# Synthetic plate / registration-DB helpers
# ---------------------------------------------------------------------------

def make_plate(rng: random.Random, index: int) -> str:
    """Generate a synthetic Vietnamese-style plate, e.g. '51A-12345'.

    Deterministic given the rng — used only to build temp DB rows / probe
    plates, never written back into the repo's real database.csv.
    """
    province = rng.randint(11, 99)
    letter = rng.choice("ABCDEFGHKLMNPSTUVXYZ")
    serial = index % 100000
    return f"{province}{letter}-{serial:05d}"


def write_registration_csv(rows: List[Tuple[str, str, str]], path: pathlib.Path) -> None:
    """Write a temp registration CSV with the SAME columns as
    main/data/database.csv: license_plate,car_brand,car_color."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("license_plate,car_brand,car_color\n")
        for plate, brand, colour in rows:
            f.write(f"{plate},{brand},{colour}\n")


# ---------------------------------------------------------------------------
# Trial construction
# ---------------------------------------------------------------------------

def group_by_class(samples: List[Tuple[str, int]]) -> Dict[str, List[str]]:
    by_class: Dict[str, List[str]] = {c: [] for c in CLASSES}
    for path, label in samples:
        by_class[CLASSES[label]].append(path)
    return by_class


def build_legitimate_trials(
    by_class: Dict[str, List[str]], rng: random.Random, n: int
) -> List[Dict]:
    """Sample n images (round-robin across colours with stock), each
    registered under its OWN true colour. No swap involved."""
    trials = []
    classes_with_stock = [c for c in CLASSES if by_class[c]]
    idx = 0
    pool = {c: list(by_class[c]) for c in classes_with_stock}
    for c in pool:
        rng.shuffle(pool[c])
    cursors = {c: 0 for c in classes_with_stock}
    while len(trials) < n:
        c = classes_with_stock[idx % len(classes_with_stock)]
        idx += 1
        if cursors[c] >= len(pool[c]):
            # Wrap around (sampling with replacement once a class is
            # exhausted) — VCoR test split is large enough per class that
            # this only kicks in for very small classes.
            cursors[c] = 0
            rng.shuffle(pool[c])
        img_path = pool[c][cursors[c]]
        cursors[c] += 1
        plate = make_plate(rng, len(trials))
        trials.append({
            "scenario": "legitimate",
            "image_path": img_path,
            "true_colour": c,
            "registered_colour": c,
            "plate": plate,
        })
    return trials


def build_plate_swap_trials(
    by_class: Dict[str, List[str]], rng: random.Random, n: int
) -> List[Dict]:
    """Sample n images of true colour C2, each registered to a DIFFERENT
    colour C1 (the 'cloned plate' belongs to a C1-coloured vehicle, but is
    shown here on a C2-coloured vehicle). C1 is chosen uniformly at random
    from the other 7 classes so all colour-pair combinations get coverage,
    balanced across true colours via round-robin."""
    trials = []
    classes_with_stock = [c for c in CLASSES if by_class[c]]
    pool = {c: list(by_class[c]) for c in classes_with_stock}
    for c in pool:
        rng.shuffle(pool[c])
    cursors = {c: 0 for c in classes_with_stock}
    idx = 0
    while len(trials) < n:
        c2 = classes_with_stock[idx % len(classes_with_stock)]
        idx += 1
        if cursors[c2] >= len(pool[c2]):
            cursors[c2] = 0
            rng.shuffle(pool[c2])
        img_path = pool[c2][cursors[c2]]
        cursors[c2] += 1
        other_colours = [c for c in CLASSES if c != c2]
        c1 = rng.choice(other_colours)
        plate = make_plate(rng, 10_000 + len(trials))
        trials.append({
            "scenario": "plate_swap",
            "image_path": img_path,
            "true_colour": c2,
            "registered_colour": c1,  # plate is registered to C1, shown on C2 car
            "plate": plate,
        })
    return trials


def build_unregistered_trials(
    by_class: Dict[str, List[str]], rng: random.Random, n: int
) -> List[Dict]:
    """Sample n images of any colour, paired with a plate that will NOT be
    inserted into the registration DB at all."""
    trials = []
    classes_with_stock = [c for c in CLASSES if by_class[c]]
    pool = {c: list(by_class[c]) for c in classes_with_stock}
    for c in pool:
        rng.shuffle(pool[c])
    cursors = {c: 0 for c in classes_with_stock}
    idx = 0
    while len(trials) < n:
        c = classes_with_stock[idx % len(classes_with_stock)]
        idx += 1
        if cursors[c] >= len(pool[c]):
            cursors[c] = 0
            rng.shuffle(pool[c])
        img_path = pool[c][cursors[c]]
        cursors[c] += 1
        plate = make_plate(rng, 20_000 + len(trials))
        trials.append({
            "scenario": "unregistered",
            "image_path": img_path,
            "true_colour": c,
            "registered_colour": None,  # not registered
            "plate": plate,
        })
    return trials


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trials(
    trials: List[Dict],
    classifier: TorchColorClassifier,
    matcher: DatabaseMatcher,
) -> List[Dict]:
    """Run colour prediction + verify_vehicle for each trial, in place
    (adds 'predicted_colour', 'predicted_conf', 'result' keys)."""
    for t in trials:
        img = cv2.imread(t["image_path"])
        if img is None:
            t["predicted_colour"] = None
            t["predicted_conf"] = 0.0
            t["result"] = {
                "status": "ERROR",
                "action": "DENY",
                "message": f"Could not read image: {t['image_path']}",
                "color_warning": False,
            }
            continue
        pred_colour, pred_conf = classifier.predict(img)
        t["predicted_colour"] = pred_colour
        t["predicted_conf"] = pred_conf
        # WS-2: pass the real colour-classification confidence through so
        # this eval measures the SAME gated logic running in production
        # (verify_vehicle no longer warns on a cross-cluster colour mismatch
        # when the model wasn't confident in its colour read).
        t["result"] = matcher.verify_vehicle(t["plate"], pred_colour, pred_conf)
    return trials


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_legitimate(trials: List[Dict]) -> Dict:
    n = len(trials)
    correct = sum(
        1 for t in trials
        if t["result"]["status"] == "AUTHORIZED" and not t["result"]["color_warning"]
    )
    false_alarms = sum(1 for t in trials if t["result"]["color_warning"])
    wrong_status = sum(1 for t in trials if t["result"]["status"] != "AUTHORIZED")
    return {
        "n_trials": n,
        "correct_no_warning": correct,
        "false_alarm_rate": round(false_alarms / n, 4) if n else 0.0,
        "false_alarm_count": false_alarms,
        "unexpected_status_count": wrong_status,
    }


def score_plate_swap(trials: List[Dict]) -> Dict:
    n = len(trials)
    caught = sum(1 for t in trials if t["result"]["color_warning"])
    missed = n - caught

    # Per-colour-pair breakdown: (registered_colour C1 -> true_colour C2)
    pair_counts: Dict[str, Dict[str, int]] = {}
    pair_misses: Dict[str, Dict[str, int]] = {}
    for t in trials:
        c1, c2 = t["registered_colour"], t["true_colour"]
        pair_counts.setdefault(c1, {}).setdefault(c2, 0)
        pair_misses.setdefault(c1, {}).setdefault(c2, 0)
        pair_counts[c1][c2] += 1
        if not t["result"]["color_warning"]:
            pair_misses[c1][c2] += 1

    pair_rows = []
    for c1 in pair_counts:
        for c2 in pair_counts[c1]:
            total = pair_counts[c1][c2]
            miss = pair_misses[c1][c2]
            pair_rows.append({
                "registered_colour": c1,
                "true_colour": c2,
                "n_trials": total,
                "n_missed": miss,
                "miss_rate": round(miss / total, 4) if total else 0.0,
            })
    pair_rows.sort(key=lambda r: (-r["n_missed"], -r["miss_rate"]))

    # Neutral-cluster aggregate (Black/Grey/Silver/White <-> each other),
    # called out separately because Report 3 Section 5.1 identifies this as
    # the colour classifier's known weak/ambiguous cluster.
    neutral = {"Black", "Grey", "Silver", "White"}
    neutral_trials = [t for t in trials if t["registered_colour"] in neutral and t["true_colour"] in neutral]
    neutral_missed = sum(1 for t in neutral_trials if not t["result"]["color_warning"])
    neutral_n = len(neutral_trials)

    return {
        "n_trials": n,
        "caught": caught,
        "missed": missed,
        "detection_rate": round(caught / n, 4) if n else 0.0,
        "miss_rate": round(missed / n, 4) if n else 0.0,
        "colour_pair_breakdown": pair_rows,
        "neutral_cluster_pairs": {
            "definition": "registered AND true colour both in {Black, Grey, Silver, White}",
            "n_trials": neutral_n,
            "n_missed": neutral_missed,
            "miss_rate": round(neutral_missed / neutral_n, 4) if neutral_n else None,
        },
    }


def score_unregistered(trials: List[Dict]) -> Dict:
    n = len(trials)
    correct = sum(
        1 for t in trials
        if t["result"]["status"] == "UNREGISTERED" and t["result"]["action"] == "DENY_ALERT"
    )
    return {
        "n_trials": n,
        "correct": correct,
        "detection_rate": round(correct / n, 4) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_json_report(report: Dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] JSON written: {path}")


def write_md_report(report: Dict, path: pathlib.Path) -> None:
    lines = []
    lines.append("# Security Evaluation — Anti-Plate-Swap (Colour Cross-Verification)")
    lines.append("")
    lines.append(
        "> Đo lường có kiểm soát (controlled evaluation) năng lực **chống tráo biển số** "
        "của hệ thống: nếu một biển số bị nhân bản (clone) từ xe A (màu C1) và gắn lên xe "
        "B khác màu (C2≠C1), việc đối chiếu màu (`DatabaseMatcher.verify_vehicle`) có phát "
        "hiện sai khác và trả `color_warning=True` (ALLOW_WARN) không? Đây là LẦN ĐẦU có "
        "số đo định lượng cho năng lực này — đề xuất ban đầu (Report 1) nêu mục tiêu ≥95% "
        "phát hiện gian lận nhưng chưa từng được đo; Report 4 trước đó chỉ test 5 ảnh, "
        "không đo an ninh."
    )
    lines.append("")
    lines.append(
        f"**Model:** weights ĐANG CHẠY ở runtime (`{report['weights_path']}`), gọi qua "
        f"`TorchColorClassifier.predict()` thật (không mock). **Decision logic:** "
        f"`DatabaseMatcher.verify_vehicle()` thật (không mock), DB đăng ký tạm thời "
        f"(`{report['temp_db_path']}`, không phải `main/data/database.csv`)."
    )
    lines.append("")
    lines.append(
        f"**Ảnh test:** VCoR held-out TEST split (cùng split với "
        f"`eval_color_deployed.py`: seed={report['seed']}, stratified 70/15/15, "
        f"từ `colab_train_color.py.load_samples`/`stratified_split`) — "
        f"{report['n_test_pool']} ảnh giữ-riêng, model chưa từng thấy khi huấn luyện. "
        f"Data layout: `{report['data_layout']}`."
    )
    lines.append("")
    lines.append(f"**Seed (reproducibility):** `{report['seed']}` — cố định cho toàn bộ việc chọn mẫu/màu đăng ký.")
    lines.append("")
    lines.append("## Headline Results")
    lines.append("")
    ps = report["plate_swap"]
    leg = report["legitimate"]
    un = report["unregistered"]
    lines.append("| Scenario | Trials | Metric | Value |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Plate-swap detection** (headline) | {ps['n_trials']} | detection rate (color_warning=True) | **{ps['detection_rate']*100:.1f}%** ({ps['caught']}/{ps['n_trials']}) |")
    lines.append(f"| Plate-swap MISSED | {ps['n_trials']} | miss rate | {ps['miss_rate']*100:.1f}% ({ps['missed']}/{ps['n_trials']}) |")
    lines.append(f"| Legitimate (no swap) | {leg['n_trials']} | false-alarm rate | {leg['false_alarm_rate']*100:.1f}% ({leg['false_alarm_count']}/{leg['n_trials']}) |")
    lines.append(f"| Unregistered plate | {un['n_trials']} | detection rate (DENY_ALERT) | {un['detection_rate']*100:.1f}% ({un['correct']}/{un['n_trials']}) |")
    lines.append("")
    lines.append(
        f"- **Plate-swap detection rate = {ps['detection_rate']*100:.1f}%** — khi biển số bị tráo lên xe "
        f"KHÁC MÀU, hệ thống bắt được {ps['caught']}/{ps['n_trials']} lần qua cảnh báo màu lệch.\n"
        f"- **False-alarm rate = {leg['false_alarm_rate']*100:.1f}%** — xe hợp lệ (không tráo) bị cảnh báo "
        f"nhầm {leg['false_alarm_count']}/{leg['n_trials']} lần (do model màu dự đoán sai ngay cả khi biển đúng).\n"
        f"- **Unregistered detection rate = {un['detection_rate']*100:.1f}%** — biển không có trong CSDL bị "
        f"chặn đúng {un['correct']}/{un['n_trials']} lần (như kỳ vọng, không phụ thuộc màu)."
    )
    lines.append("")
    lines.append("## Colour-Pair Breakdown — Plate-Swap Misses")
    lines.append("")
    lines.append(
        "Cặp (màu đăng ký C1 → màu thật C2) bị MISS nhiều nhất (color_warning vẫn False dù màu thật khác màu đăng ký):"
    )
    lines.append("")
    lines.append("| Registered (C1) | True (C2) | Trials | Missed | Miss rate |")
    lines.append("|---|---|---|---|---|")
    top_misses = [r for r in ps["colour_pair_breakdown"] if r["n_missed"] > 0][:15]
    if not top_misses:
        lines.append("| — | — | — | 0 | 0% — không có cặp màu nào bị miss |")
    for r in top_misses:
        lines.append(
            f"| {r['registered_colour']} | {r['true_colour']} | {r['n_trials']} | "
            f"{r['n_missed']} | {r['miss_rate']*100:.1f}% |"
        )
    lines.append("")
    nc = ps["neutral_cluster_pairs"]
    if nc["n_trials"]:
        lines.append(
            f"**Cụm màu trung tính (Black/Grey/Silver/White ↔ nhau):** {nc['n_trials']} trial, "
            f"miss {nc['n_missed']} ({nc['miss_rate']*100:.1f}%) — "
            "khớp với cụm nhập nhằng đã ghi nhận ở Report 3 §5.1 (confusion matrix màu), "
            "đây là nơi cross-check màu YẾU NHẤT vì model màu chính nó cũng nhầm trong cụm này."
        )
    else:
        lines.append("**Cụm màu trung tính:** không có trial nào rơi vào cụm này trong lần chạy này.")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "1. Lấy ảnh từ VCoR TEST split giữ-riêng (seed=42, 70/15/15 stratified — giống "
        "`eval_color_deployed.py`), nên ảnh hoàn toàn chưa từng thấy khi huấn luyện model màu.\n"
        "2. Với mỗi trial, chạy `TorchColorClassifier.predict(bgr_image)` THẬT trên ảnh để lấy "
        "màu dự đoán (không dùng nhãn ground-truth) — phản ánh đúng hành vi triển khai thực tế, "
        "kể cả khi model màu đoán sai.\n"
        "3. Xây CSDL đăng ký TẠM (cùng schema `main/data/database.csv`: "
        "`license_plate,car_brand,car_color`), không đụng tới CSDL thật của repo.\n"
        "4. Gọi `DatabaseMatcher.verify_vehicle(plate, predicted_colour)` THẬT (logic quyết "
        "định không sửa đổi) cho 3 loại scenario, mỗi loại ~200 trial cân bằng theo màu, "
        "RNG seed cố định để tái lập được.\n"
        "5. **legitimate**: ảnh màu thật C, biển đăng ký màu C → đúng = AUTHORIZED + không cảnh báo.\n"
        "6. **plate_swap**: ảnh màu thật C2, biển đăng ký màu C1≠C2 (giả lập biển bị nhân bản từ "
        "xe màu C1, gắn lên xe màu C2) → đúng = `color_warning=True` (bắt được tráo).\n"
        "7. **unregistered**: biển hoàn toàn không có trong CSDL → đúng = UNREGISTERED/DENY_ALERT."
    )
    lines.append("")
    lines.append("## Limitations (đọc trước khi trích số liệu)")
    lines.append("")
    lines.append(
        "- **Chỉ bắt được khi xe gắn biển tráo có MÀU KHÁC màu đăng ký.** Nếu kẻ tráo biển "
        "dùng đúng xe cùng màu (hoặc sơn/dán decal giả màu), cross-check màu KHÔNG có cơ chế "
        "phát hiện — đây là lỗ hổng cố hữu của cơ chế \"màu là cảnh báo mềm\", không phải lỗi "
        "đo lường.\n"
        "- **Phụ thuộc hoàn toàn vào việc OCR đọc đúng biển số trước đó** (Benchmark C: ~81% "
        "exact-match). Eval này giả định biển được đọc đúng (test cách ly bước cross-check màu); "
        "nếu OCR đọc sai/đọc thiếu, biển sẽ rơi vào UNREGISTERED hoặc match nhầm bản ghi khác — "
        "số đo plate-swap ở đây KHÔNG bao gồm lỗi OCR thực tế.\n"
        "- **Đo trên VCoR (ảnh web/marketplace sạch)** — CCTV bãi xe thật (ánh sáng yếu, góc "
        "nghiêng, nén ảnh, độ phân giải thấp) nhiều khả năng cho tỉ lệ phát hiện THẤP HƠN do "
        "domain gap (xem Report 3 §5.1, Report 4 §5.1 caveat tương tự cho colour accuracy).\n"
        "- Biển số và CSDL đăng ký trong eval này là DỮ LIỆU TỔNG HỢP (synthetic), sinh từ RNG "
        "seed cố định — không phải biển số thật, chỉ dùng để dựng tình huống kiểm thử có kiểm soát."
    )
    lines.append("")
    lines.append(
        f"_Sinh bởi `main/scripts/eval_security.py`, chạy lúc thực thi script này trên model "
        f"đang triển khai (`{report['weights_path']}`), {report['n_total_trials']} trial tổng, "
        f"elapsed {report['elapsed_seconds']:.1f}s._"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[report] Markdown written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default="/Users/konalyn/Downloads/archive",
        help="Path to PURE VCoR dataset root (layout: {train,val,test}/<lowercolor>/*.jpg). "
             "Falls back to main/data/raw/car_colors_vcor (flat layout) if missing.",
    )
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="DEPLOYED runtime weights (.pt). Read-only, never overwritten.")
    p.add_argument("--device", default="auto", help="cuda | mps | cpu | auto (default: auto).")
    p.add_argument("--trials-per-scenario", type=int, default=TRIALS_PER_SCENARIO)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    p.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = p.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    fallback_dir = _MAIN_DIR / "data" / "raw" / "car_colors_vcor"
    used_fallback = False
    if not data_dir.is_dir():
        print(f"[WARN] VCoR archive dir not found: {data_dir}")
        if fallback_dir.is_dir():
            print(f"[WARN] Falling back to flat layout: {fallback_dir}")
            data_dir = fallback_dir
            used_fallback = True
        else:
            print(f"[FATAL] Fallback dir also not found: {fallback_dir}")
            sys.exit(1)

    weights_path = pathlib.Path(args.weights)
    if not weights_path.is_file():
        print(f"[FATAL] Deployed weights not found: {weights_path}")
        sys.exit(1)

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[device] Using: {device}")

    print(f"[data] Loading VCoR-mapped samples from: {data_dir}")
    samples, layout = load_samples(str(data_dir))
    print(f"[data] Layout detected: {layout} (fallback used: {used_fallback})")
    print(f"[data] Total images (pooled, mapped to {len(CLASSES)} classes): {len(samples)}")

    # SAME 70/15/15 stratified split, seed=args.seed (default 42), as
    # colab_train_color.py / eval_color_deployed.py -> genuinely held-out.
    train_s, val_s, test_s = stratified_split(samples, seed=args.seed)
    print(f"[data] Split — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)} (HELD-OUT, used here)")

    by_class = group_by_class(test_s)
    counts = {c: len(by_class[c]) for c in CLASSES}
    print("[data] Test per-class counts: " + ", ".join(f"{c}={counts[c]}" for c in CLASSES))
    empty_classes = [c for c in CLASSES if counts[c] == 0]
    if empty_classes:
        print(f"[FATAL] Test split has zero images for class(es): {empty_classes}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build the 3 scenario trial sets from a single seeded RNG (sequential
    # draws -> deterministic given --seed, no datetime/unseeded random).
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    n = args.trials_per_scenario
    legit_trials = build_legitimate_trials(by_class, rng, n)
    swap_trials = build_plate_swap_trials(by_class, rng, n)
    unreg_trials = build_unregistered_trials(by_class, rng, n)
    print(f"[trials] legitimate={len(legit_trials)} plate_swap={len(swap_trials)} unregistered={len(unreg_trials)}")

    # ------------------------------------------------------------------
    # Build temp registration DB: every legitimate + plate_swap plate gets
    # a row (registered to its OWN scenario's registered_colour). Unregistered
    # plates are deliberately NOT inserted.
    # ------------------------------------------------------------------
    db_rows: List[Tuple[str, str, str]] = []
    for t in legit_trials + swap_trials:
        db_rows.append((t["plate"], "TestBrand", t["registered_colour"]))

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="dpl302m_security_eval_"))
    temp_db_path = tmp_dir / "temp_registration.csv"
    write_registration_csv(db_rows, temp_db_path)
    print(f"[db] Temp registration CSV written ({len(db_rows)} rows): {temp_db_path}")

    # Sanity: no unregistered-scenario plate should collide with a DB row
    # (extremely unlikely given the index offsets in make_plate, but assert
    # to fail loudly rather than silently corrupt the "unregistered" scenario).
    db_plates = {row[0] for row in db_rows}
    collisions = [t["plate"] for t in unreg_trials if t["plate"] in db_plates]
    if collisions:
        print(f"[FATAL] {len(collisions)} 'unregistered' plates collided with registered plates: {collisions[:5]}...")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load REAL deployed model + REAL decision logic.
    # ------------------------------------------------------------------
    print(f"[model] Loading DEPLOYED weights: {weights_path}")
    classifier = TorchColorClassifier(str(weights_path), device=str(device))
    matcher = DatabaseMatcher(str(temp_db_path))

    # ------------------------------------------------------------------
    # Run trials.
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[run] Scenario 1/3: legitimate...")
    run_trials(legit_trials, classifier, matcher)
    print("[run] Scenario 2/3: plate_swap...")
    run_trials(swap_trials, classifier, matcher)
    print("[run] Scenario 3/3: unregistered...")
    run_trials(unreg_trials, classifier, matcher)
    elapsed = time.time() - t0
    print(f"[run] Done. Elapsed: {elapsed:.1f}s for {len(legit_trials) + len(swap_trials) + len(unreg_trials)} trials")

    # ------------------------------------------------------------------
    # Score.
    # ------------------------------------------------------------------
    legit_metrics = score_legitimate(legit_trials)
    swap_metrics = score_plate_swap(swap_trials)
    unreg_metrics = score_unregistered(unreg_trials)

    print(f"\n[result] Plate-swap detection rate : {swap_metrics['detection_rate']*100:.1f}% "
          f"({swap_metrics['caught']}/{swap_metrics['n_trials']})")
    print(f"[result] False-alarm rate (legit)   : {legit_metrics['false_alarm_rate']*100:.1f}% "
          f"({legit_metrics['false_alarm_count']}/{legit_metrics['n_trials']})")
    print(f"[result] Unregistered detection rate: {unreg_metrics['detection_rate']*100:.1f}% "
          f"({unreg_metrics['correct']}/{unreg_metrics['n_trials']})")
    print("\n[result] Top colour-pair misses (registered C1 -> true C2):")
    for r in [r for r in swap_metrics["colour_pair_breakdown"] if r["n_missed"] > 0][:10]:
        print(f"  {r['registered_colour']:8s} -> {r['true_colour']:8s} : "
              f"{r['n_missed']}/{r['n_trials']} missed ({r['miss_rate']*100:.1f}%)")

    report = {
        "weights_path": str(weights_path),
        "temp_db_path": str(temp_db_path),
        "data_dir": str(data_dir),
        "data_layout": layout,
        "used_fallback_data_dir": used_fallback,
        "seed": args.seed,
        "n_test_pool": len(test_s),
        "n_total_trials": len(legit_trials) + len(swap_trials) + len(unreg_trials),
        "elapsed_seconds": round(elapsed, 2),
        "legitimate": legit_metrics,
        "plate_swap": swap_metrics,
        "unregistered": unreg_metrics,
        "eval_script": "main/scripts/eval_security.py",
        "note": (
            "Real deployed colour model (TorchColorClassifier, color_MobileNetV3Small.pt) + "
            "real DatabaseMatcher.verify_vehicle decision logic, on VCoR held-out TEST split "
            "(seed=42, 70/15/15 stratified, same split as eval_color_deployed.py). Synthetic "
            "plates/registration CSV built only for this run; main/data/database.csv untouched."
        ),
    }

    write_json_report(report, pathlib.Path(args.json_out))
    write_md_report(report, pathlib.Path(args.md_out))

    print("\n[done] Security evaluation complete. No runtime code/weights/database.csv modified.")


if __name__ == "__main__":
    main()
