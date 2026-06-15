"""Render design-A benchmark & dataset charts (PNG) from the real CSV data.

Outputs to ``presentations/`` for embedding in the redesigned slides (WS2).
Style matches the "Clean Light Systems" palette: off-white bg, forest-green
accent, ink text, mono numerals. Run:

    KMP_DUPLICATE_LIB_OK=TRUE python main/scripts/make_benchmark_charts.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs" / "benchmarks"
OUT = ROOT / "presentations"

# --- design-A tokens -------------------------------------------------------
BG = "#fafaf9"
INK = "#18181b"
MUTED = "#71717a"
ACCENT = "#15803d"      # forest green (winner / selected)
ALERT = "#b91c1c"       # red (loser)
HAIRLINE = "#e4e4e7"
NEUTRAL = "#d4d4d8"     # grey bars

# Prefer a squared/mono face for numerals if present, else default sans.
for fam in ("JetBrains Mono", "DejaVu Sans Mono"):
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        MONO = fam
        break
    except Exception:
        MONO = "monospace"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": HAIRLINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.titlesize": 12,
})


def _read(name: str) -> list[dict[str, str]]:
    with open(BENCH / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _style_axes(ax) -> None:
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HAIRLINE)


# --- 1. OCR exact-match ----------------------------------------------------
def chart_ocr() -> None:
    rows = _read("ocr_benchmark.csv")
    labels = [r["method"] for r in rows]
    vals = [float(r["exact_match"]) * 100 for r in rows]
    colors = [ACCENT if r["method"].startswith("ppocr") and r["method"] == "ppocr"
              else (ACCENT if "ppocr" in r["method"] else ALERT if v == 0 else NEUTRAL)
              for r, v in zip(rows, vals)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(labels, vals, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exact-match (%)  ·  16 biển CCTV thật", color=MUTED)
    ax.set_title("Benchmark C — OCR đọc đúng biển số",
                 color=INK, fontweight="bold", loc="left", pad=12)
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", ha="left", color=INK, fontfamily=MONO, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "chart_ocr_benchmark.png", dpi=160)
    plt.close(fig)


# --- 2. Colour CNN ---------------------------------------------------------
def chart_color() -> None:
    rows = _read("color_benchmark.csv")
    names = [r["name"] for r in rows]
    acc = [float(r["accuracy"]) * 100 for r in rows]
    size = [float(r["size_mb"]) for r in rows]
    # MobileNetV3 is the shipped model -> accent; others neutral
    colors = [ACCENT if n == "MobileNetV3Small" else NEUTRAL for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(names, acc, color=colors, width=0.6)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Accuracy (%)  ·  226 ảnh val", color=MUTED)
    ax.set_title("Benchmark A — CNN màu xe  (xanh = đã giao)",
                 color=INK, fontweight="bold", loc="left", pad=12)
    for b, a, s in zip(bars, acc, size):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.2, f"{a:.1f}%",
                ha="center", color=INK, fontfamily=MONO, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 3, f"{s:.1f} MB",
                ha="center", color=BG, fontfamily=MONO, fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "chart_color_benchmark.png", dpi=160)
    plt.close(fig)


# --- 3. Plate detector mAP -------------------------------------------------
def chart_plate() -> None:
    rows = _read("plate_benchmark.csv")
    names = ["Fine-tune" if r["name"] == "plate_finetune" else "Scratch" for r in rows]
    m = [float(r["mAP50"]) * 100 for r in rows]
    colors = [ACCENT if n == "Fine-tune" else NEUTRAL for n in names]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bars = ax.bar(names, m, color=colors, width=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("mAP@0.5 (%)", color=MUTED)
    ax.set_title("Benchmark B — phát hiện biển (YOLOv8n)",
                 color=INK, fontweight="bold", loc="left", pad=12)
    for b, v in zip(bars, m):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%",
                ha="center", color=INK, fontfamily=MONO, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "chart_plate_benchmark.png", dpi=160)
    plt.close(fig)


# --- 4. Dataset distribution ----------------------------------------------
def chart_dataset() -> None:
    brands = {"Toyota": 168, "Hyundai": 200, "Kia": 120, "Mazda": 120,
              "Honda": 161, "VinFast": 120, "Ford": 200, "Mitsubishi": 120}
    colours = {"White": 185, "Black": 200, "Grey": 200, "Silver": 175,
               "Red": 110, "Blue": 200, "Brown": 35, "Yellow": 25}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (title, data, total) in zip(
        axes,
        [("Hãng xe — 1,209 ảnh / 8 lớp", brands, 1209),
         ("Màu xe — 1,130 ảnh / 8 lớp", colours, 1130)],
    ):
        keys = list(data.keys())
        vals = list(data.values())
        ax.bar(keys, vals, color=ACCENT, width=0.66)
        ax.set_title(title, color=INK, fontweight="bold", loc="left", pad=10, fontsize=12)
        ax.tick_params(axis="x", labelrotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.set_ylim(0, 220)
        for i, v in enumerate(vals):
            ax.text(i, v + 4, str(v), ha="center", color=MUTED,
                    fontfamily=MONO, fontsize=9)
        _style_axes(ax)
    fig.suptitle("Phân bố tập dữ liệu thực (đã làm sạch)", color=INK,
                 fontweight="bold", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "chart_dataset.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chart_ocr()
    chart_color()
    chart_plate()
    chart_dataset()
    print("Wrote:", *[p.name for p in sorted(OUT.glob("chart_*.png"))])
