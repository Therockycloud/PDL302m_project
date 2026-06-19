"""Reproducible held-out evaluation of the DEPLOYED colour classifier.

Re-creates the exact test split used during the Colab fine-tune
(`colab_train_color.py`: pure VCoR layout, seed=42, 70/15/15 stratified
split) and evaluates the weights currently shipped at
``main/data/models/color_MobileNetV3Small.pt`` on the held-out TEST split
only. This is a verification script, not a training script — it does not
write or modify the .pt weights, and it imports its data/model helpers
directly from ``colab_train_color.py`` instead of re-implementing them, so
the split/model definitions are guaranteed to match the ones used to
produce the deployed weights.

Usage (run from anywhere; paths below are resolved relative to this file):

    KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python \\
        main/scripts/eval_color_deployed.py \\
        --data-dir /Users/konalyn/Downloads/archive

Writes:
    docs/benchmarks/color_finetune_report.json
    docs/benchmarks/color_finetune_report.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

# Make main/scripts importable as a plain directory (colab_train_color.py
# has no package __init__, so add its parent dir to sys.path).
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from colab_train_color import (  # noqa: E402
    CLASSES,
    compute_class_weights,
    load_samples,
    pick_device,
    predict_all_tta,
    set_seed,
    stratified_split,
    build_model,
    build_report,
    write_json_report,
)

# Repo root = two levels up from this file (main/scripts/eval_color_deployed.py).
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
DEFAULT_WEIGHTS = _REPO_ROOT / "main" / "data" / "models" / "color_MobileNetV3Small.pt"
DEFAULT_JSON_OUT = _REPO_ROOT / "docs" / "benchmarks" / "color_finetune_report.json"
DEFAULT_MD_OUT = _REPO_ROOT / "docs" / "benchmarks" / "color_finetune_report.md"


def write_md_report(report: dict, path: pathlib.Path) -> None:
    lines = []
    lines.append("# Color Classifier Fine-Tune Report — Deployed Model (VCoR held-out test)")
    lines.append("")
    lines.append(
        "> Đánh giá tái lập (reproducible) cho model **đang chạy ở runtime** "
        f"(`main/data/models/color_MobileNetV3Small.pt`), đo trên tập TEST giữ-riêng "
        "của VCoR (Kaggle), split 70/15/15 stratified seed=42 — cùng split dùng khi "
        "fine-tune trên Colab. Sinh bởi `main/scripts/eval_color_deployed.py`, tái sử dụng "
        "hàm load/split/eval từ `main/scripts/colab_train_color.py`."
    )
    lines.append("")
    lines.append(f"> baseline frozen-backbone (data cũ, trước VCoR) ≈ 0.5508 (55.1%) — xem `baseline_note`.")
    lines.append("")
    lines.append(f"**Data layout:** `{report['data_layout']}`  |  **Tổng ảnh (pool, 8 lớp):** {report['n_samples_total']}  |  **Test split:** {report['n_test']} ảnh")
    lines.append("")
    lines.append(f"**Test Accuracy (plain, no TTA):** {report['test_accuracy_plain']:.4f}  ")
    lines.append(f"**Test Accuracy (TTA, hflip-averaged):** {report['test_accuracy_tta']:.4f}  ")
    lines.append(f"**Test Macro-F1 (plain):** {report['test_macro_f1_plain']:.4f}  ")
    lines.append(f"**Test Macro-F1 (TTA):** {report['test_macro_f1_tta']:.4f}")
    lines.append("")
    lines.append("## Per-Class Metrics (TTA predictions)")
    lines.append("")
    lines.append("| Class | Precision | Recall |")
    lines.append("|-------|-----------|--------|")
    for cls in CLASSES:
        pc = report["per_class"][cls]
        lines.append(f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} |")
    lines.append("")
    lines.append("## Confusion Matrix (TTA)")
    lines.append("")
    lines.append("Rows = true class, Columns = predicted class.")
    lines.append("")
    header = "| | " + " | ".join(CLASSES) + " |"
    sep = "|---|" + "---|" * len(CLASSES)
    lines.append(header)
    lines.append(sep)
    cm = report["confusion_matrix"]
    for i, cls in enumerate(CLASSES):
        row = " | ".join(str(v) for v in cm[i])
        lines.append(f"| {cls} | {row} |")
    lines.append("")
    lines.append("## Levers (fine-tune recipe, Colab GPU)")
    lines.append("")
    for lever in report["levers"]:
        lines.append(f"- {lever}")
    lines.append("")
    lines.append(
        "**Lưu ý trung thực:** số liệu trên đo trên VCoR (ảnh web sạch, không phải CCTV "
        "bãi xe thật) — hiệu năng triển khai thực tế sẽ thấp hơn do domain gap "
        "(ánh sáng/độ phân giải CCTV); xem caveat đầy đủ trong các báo cáo chính."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[report] Markdown written: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default="/Users/konalyn/Downloads/archive",
        help="Path to PURE VCoR dataset root (layout: {train,val,test}/<lowercolor>/*.jpg).",
    )
    p.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to the DEPLOYED runtime weights (.pt) to evaluate. Read-only — never overwritten.",
    )
    p.add_argument("--device", default="auto", help="cuda | mps | cpu | auto (default: auto).")
    p.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    p.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = p.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[FATAL] VCoR data dir not found: {data_dir}")
        sys.exit(1)

    weights_path = pathlib.Path(args.weights)
    if not weights_path.is_file():
        print(f"[FATAL] Deployed weights not found: {weights_path}")
        sys.exit(1)

    set_seed(42)
    device = pick_device(args.device)
    print(f"[device] Using: {device}")
    print(f"[data] Loading PURE VCoR from: {data_dir}")

    samples, layout = load_samples(str(data_dir))
    if layout != "vcor":
        print(f"[FATAL] Expected vcor layout, detected: {layout}")
        sys.exit(1)
    print(f"[data] Total images (pooled, mapped to our {len(CLASSES)} classes): {len(samples)}")

    # SAME 70/15/15 stratified split, seed=42, as colab_train_color.py.
    train_s, val_s, test_s = stratified_split(samples)
    print(f"[data] Split — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    counts = {c: 0 for c in CLASSES}
    for _, lbl in test_s:
        counts[CLASSES[lbl]] += 1
    print("[data] Test per-class counts: " + ", ".join(f"{c}={counts[c]}" for c in CLASSES))

    # Class weights are reported for context only (loss weighting happened
    # at TRAIN time on Colab); recomputing here from the same train split
    # for the report's "levers" section, NOT used to alter eval.
    class_weights = compute_class_weights(train_s)

    print(f"[model] Building MobileNetV3-Small (8-class head)...")
    model = build_model()
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[model] Loaded DEPLOYED weights from: {weights_path}")

    print("\n[test] Evaluating on held-out TEST split (plain + TTA)...")
    t0 = time.time()
    y_true, y_pred_plain, y_pred_tta = predict_all_tta(model, test_s, device)
    elapsed = time.time() - t0

    report = build_report(y_true, y_pred_plain, y_pred_tta, class_weights, layout, len(samples))
    report["n_test"] = len(test_s)
    report["weights_path"] = str(weights_path)
    report["eval_script"] = "main/scripts/eval_color_deployed.py"
    report["note"] = (
        "Reproducible held-out eval of the DEPLOYED runtime weights "
        "(color_MobileNetV3Small.pt), NOT a retrain. Same data load / "
        "70-15-15 stratified split (seed=42) as colab_train_color.py, "
        "reused via import."
    )

    print(f"\n[test] Accuracy (plain) : {report['test_accuracy_plain']:.4f}")
    print(f"[test] Accuracy (TTA)   : {report['test_accuracy_tta']:.4f}")
    print(f"[test] Macro-F1 (plain) : {report['test_macro_f1_plain']:.4f}")
    print(f"[test] Macro-F1 (TTA)   : {report['test_macro_f1_tta']:.4f}")
    print(f"[test] Eval time: {elapsed:.1f}s for {len(test_s)} images")

    print("\n[test] Per-class (TTA):")
    for cls in CLASSES:
        pc = report["per_class"][cls]
        print(f"  {cls:8s}: prec={pc['precision']:.3f} recall={pc['recall']:.3f}")

    print("\n[test] Confusion matrix (rows=true, cols=predicted, TTA):")
    header = "          " + " ".join(f"{c[:4]:>5s}" for c in CLASSES)
    print(header)
    for i, row in enumerate(report["confusion_matrix"]):
        print(f"  {CLASSES[i]:8s}" + " ".join(f"{v:5d}" for v in row))

    write_json_report(report, args.json_out)
    write_md_report(report, pathlib.Path(args.md_out))

    print("\n[done] Deployed-model held-out evaluation complete (weights file untouched).")


if __name__ == "__main__":
    main()
