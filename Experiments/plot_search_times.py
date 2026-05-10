"""
Search Time Comparison Plot
============================
Bar chart of mean per-query search time (ms) on a log scale.

Usage
-----
# From CSV output of run_multi_tokeniser.py:
python plot_search_times.py --csv results/best_search_times.csv --out results/search_times.pdf

# Using hardcoded values (no CSV needed):
python plot_search_times.py --out results/search_times.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


FALLBACK = {
    "TF-IDF":    0.38,
    "BEST-STD":  18.61,
    "H-QuEST":   54.56,
    "PMI-QuEST": 56.47,
    "Token-DTW": 75.03,
}
SYSTEM_ORDER = ["TF-IDF", "BEST-STD",  "H-QuEST", "PMI-QuEST", "Token-DTW"]

COL_MAP = {
    "TF-IDF":    "tfidf_ms_mean",
    "BEST-STD":  "beststd_ms_mean",
    "H-QuEST":   "hquest_ms_mean",
    "PMI-QuEST": "pmi_ms_mean",
    "Token-DTW": "dtw_ms_mean",
}


def load_from_csv(csv_path: str) -> dict:
    """Average mean latency across tokeniser configs if multiple rows."""
    sums  = {s: 0.000 for s in SYSTEM_ORDER}
    count = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            count += 1
            for sys_name, col in COL_MAP.items():
                sums[sys_name] += float(row[col])
    if count == 0:
        raise ValueError(f"No rows in {csv_path}")
    return {s: sums[s] / count for s in SYSTEM_ORDER}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(data: dict, out_path: str):
    systems = SYSTEM_ORDER
    means   = [data[s] for s in systems]

    # colour gradient: light blue (TF-IDF) -> dark blue (PMI-QuEST)
    colors = ["#c6ddf0", "#7eaee0", "#4a8fc4", "#1f6aa8", "#0c3d6e"]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    x    = np.arange(len(systems))
    bars = ax.bar(x, means, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55, zorder=3)

    # value labels above each bar
    for bar, mean in zip(bars, means):
        if mean < 1:
            label = f"{mean:.2f} ms"
        elif mean < 1000:
            label = f"{mean:.2f} ms"
        else:
            label = f"{mean/1000:.3f} s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.7,
            label,
            ha="center", va="bottom", fontsize=8.5, color="#1a1a1a",
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_xlabel("Systems", fontsize=10)
    ax.set_ylabel("Mean search time per query (ms) (log scale)", fontsize=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: (
                f"{v:.4g}"       if v < 1    else
                f"{v:.0f}"       if v < 1000 else
                f"{v/1000:.0f}k"
            )
        )
    )

    ax.set_ylim(bottom=0.05, top=max(means) * 8)
    ax.grid(axis="y", color="#d3d1c7", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b4b2a9")
    ax.tick_params(axis="both", colors="#5f5e5a", labelsize=9)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="CSV from run_multi_tokeniser.py. "
                             "Omit to use hardcoded fallback values.")
    parser.add_argument("--out", default="results/search_times.pdf",
                        help="Output path (.pdf / .png / .svg).")
    args = parser.parse_args()

    if args.csv:
        data = load_from_csv(args.csv)
        print(f"Loaded from {args.csv}")
    else:
        data = FALLBACK
        print("Using hardcoded fallback values.")

    for s in SYSTEM_ORDER:
        print(f"  {s:<14} {data[s]:>10.2f} ms")

    plot(data, args.out)
