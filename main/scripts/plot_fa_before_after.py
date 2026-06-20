"""Before/after bar chart for the WS-2 colour-warning gate change.

Plots two grouped metrics side by side:
    1. False Alarm rate   (legitimate, no-swap vehicles wrongly warned)
    2. Plate-swap Detection rate (cloned plate on a different-coloured car
       correctly flagged via color_warning=True)

"Before" = the pre-WS-2 decision logic (no neutral-colour clustering, no
confidence gating; effectively color_warn_conf=0.0, every cross-colour
mismatch warns regardless of model confidence).
"After"  = the deployed WS-2 logic with the gate set to color_warn_conf=0.40
(see main/configs/config.yaml `decision.color_warn_conf` and
docs/benchmarks/security_eval.md for the full gate sweep).

Numbers are hard-coded from the measured, held-out VCoR evaluation
(main/scripts/eval_security.py, seed=42, neutral-merge enabled where
applicable) rather than re-run here, because "before" requires reverting the
neutral-cluster + gating logic entirely (a different code path, not just a
config value) -- re-deriving it on every plot run would require a second,
separate evaluation harness. The values below are the ones already verified
against docs/benchmarks/security_eval.json and reports/documents/
Report_4_Final_Report.md §4.3:

    Before (legacy, pre-WS-2):      FA = 14.5% (29/200), detection = 98.5% (197/200)
    After  (WS-2, gate = 0.40):     FA =  2.5%  (5/200), detection = 69.0% (138/200)

Usage:
    cd main && KMP_DUPLICATE_LIB_OK=TRUE <python> scripts/plot_fa_before_after.py

Output:
    docs/benchmarks/security_fa_before_after.png (dpi=200)
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display backend required
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Measured values (percent). Keep in sync with docs/benchmarks/security_eval.md
# and reports/documents/Report_4_Final_Report.md §4.3.
# ---------------------------------------------------------------------------
METRICS = ["False Alarm", "Plate-swap Detection"]
BEFORE = [14.5, 98.5]  # legacy logic, no neutral-cluster / no confidence gate
AFTER = [2.5, 69.0]  # WS-2 logic, color_warn_conf = 0.40 (deployed gate)

BEFORE_COLOR = "#d62728"  # red -- legacy / higher false-alarm exposure
AFTER_COLOR = "#1f77b4"  # blue -- deployed WS-2 operating point

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "benchmarks", "security_fa_before_after.png")


def make_chart(output_path: str = OUTPUT_PATH) -> str:
    import numpy as np

    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars_before = ax.bar(
        x - width / 2, BEFORE, width, label="Before (legacy, no gate)", color=BEFORE_COLOR
    )
    bars_after = ax.bar(
        x + width / 2, AFTER, width, label="After (WS-2, gate=0.40)", color=AFTER_COLOR
    )

    for bars in (bars_before, bars_after):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title(
        "Anti-plate-swap security: false-alarm reduction vs. detection trade-off",
        fontsize=13,
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=12)
    ax.set_ylim(0, 112)  # headroom above the tallest bar + its label (98.5%)
    # Legend in the top-LEFT corner: with this data (False Alarm bars both
    # short, Plate-swap bars both tall) the only consistently empty region
    # at the top of the axes is above the "False Alarm" group, so anchor
    # the legend there instead of centring it above the tallest bar.
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.015,
        "Before = legacy logic (no neutral-cluster merge / no confidence gate)"
        "   |   After = WS-2 gate, color_warn_conf = 0.40",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    fig.tight_layout(rect=(0.01, 0.05, 0.99, 1))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = make_chart()
    size_bytes = os.path.getsize(path)
    print(f"[plot] Saved: {path} ({size_bytes / 1024:.1f} KB)")
