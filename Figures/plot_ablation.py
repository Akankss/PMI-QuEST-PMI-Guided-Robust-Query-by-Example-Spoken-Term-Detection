"""
plot_ablation.py
----------------
Plots PMI-QuEST ablation results (Groups A–E) as a 2×3 subplot figure
suitable for a TASLP journal paper.

Usage:
    python plot_ablation.py
    python plot_ablation.py --out figures/ablation.pdf
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

# Group A: Bigram Presence
A_labels = ["H-QuEST\n(Unigrams only)", "PMI-QuEST\n(PMI-TF-IDF)"]
A = {
    "MAP": [0.6760, 0.7867],
    "MRR": [0.8621, 0.9809],
    "P@1": [0.8539, 0.9775],
    "P@5": [0.2315, 0.2629],
    "P@10":[0.1236, 0.1427],
}

# Group B: PMI Threshold τ
B_tau     = [0.0, 0.5, 1.0, 1.5, 2.0]
B_bigrams = [79827, 64272, 50113, 37715]
B = {
    "MAP": [0.7867, 0.7867, 0.7867, 0.7849, 0.7845],
    "MRR": [0.9809, 0.9809, 0.9809, 0.9697, 0.9697],
    "P@1": [0.9775, 0.9775, 0.9775, 0.9663, 0.9663],
    "P@5": [0.2629, 0.2629, 0.2629, 0.2607, 0.2629],
    "P@10":[0.1427, 0.1427, 0.1427, 0.1416, 0.1416],
}
B_proposed_idx = 1   # τ=0.5

# Group C: Bigram Weight α
C_alpha = [0.25, 0.50, 0.75, 1.00, 2.00, 5.00]
C = {
    "MAP": [0.7337, 0.7666, 0.7864, 0.7867, 0.7877, 0.7872],
    "MRR": [0.9180, 0.9622, 0.9809, 0.9809, 0.9809, 0.9809],
    "P@1": [0.9101, 0.9551, 0.9775, 0.9775, 0.9775, 0.9775],
    "P@5": [0.2427, 0.2584, 0.2629, 0.2629, 0.2607, 0.2607],
    "P@10":[0.1315, 0.1393, 0.1427, 0.1427, 0.1427, 0.1427],
}
C_proposed_idx = 3   # α=1.0

# Group D: HNSW K
D_K = [10, 20, 50, 100, 200]
D = {
    "MAP": [0.5041, 0.5746, 0.7059, 0.7470, 0.7867],
    "MRR": [0.6896, 0.7664, 0.8956, 0.9474, 0.9809],
    "P@1": [0.6742, 0.7528, 0.8876, 0.9438, 0.9775],
    "P@5": [0.1663, 0.1820, 0.2315, 0.2517, 0.2629],
    "P@10":[0.0865, 0.1045, 0.1247, 0.1371, 0.1427],
}
D_proposed_idx = 4   # K=200

# Group E: SW Reranking
E_labels = ["HNSW\nonly", "HNSW\n+ SW"]
E = {
    "MAP": [0.3221, 0.7867],
    "MRR": [0.4641, 0.9809],
    "P@1": [0.3483, 0.9775],
    "P@5": [0.1303, 0.2629],
    "P@10":[0.0865, 0.1427],
}

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "MAP": "#2563EB",   # blue
    "MRR": "#DC2626",   # red
    "P@1": "#16A34A",   # green
    "P@5": "#9333EA",   # purple
    "P@10":"#EA580C",   # orange
}
PROPOSED_COLOR = "#F59E0B"   # amber dashed line
MARKERS = {"MAP": "o", "MRR": "s", "P@1": "^", "P@5": "D", "P@10": "v"}
LINEWIDTH = 2.4
MARKERSIZE = 9
FONTSIZE_TITLE  = 13
FONTSIZE_AXIS   = 12
FONTSIZE_TICK   = 11
FONTSIZE_LEGEND = 11

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.weight":       "bold",
    "axes.labelweight":  "bold",
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        350,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
})


def add_proposed_vline(ax, x, label=True):
    ax.axvline(x, color=PROPOSED_COLOR, linestyle="--", linewidth=1.4,
               zorder=0, label="Proposed" if label else None)


def plot_line(ax, xs, data, xlabel, title, proposed_x=None,
              x_is_categorical=False, xtick_labels=None):
    """Generic line-plot helper."""
    x_pos = list(range(len(xs))) if x_is_categorical else xs

    for metric, color in COLORS.items():
        ax.plot(x_pos, data[metric], color=color,
                marker=MARKERS[metric], linewidth=LINEWIDTH,
                markersize=MARKERSIZE, label=metric, zorder=3)

    if proposed_x is not None:
        px = proposed_x if not x_is_categorical else proposed_x
        add_proposed_vline(ax, px)

    if x_is_categorical:
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xtick_labels or xs, fontsize=FONTSIZE_TICK)
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=FONTSIZE_TICK)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight="bold", pad=6)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)





# ── Figure ────────────────────────────────────────────────────────────────────

def main(out_path="ablation.pdf"):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle(
        "PMI-QuEST — Ablation Study (LibriSpeech test-clean)",
        fontsize=11, fontweight="bold", y=1.03
    )

    # ── A: Bigram Presence (line) ──────────────────────────────────────────
    plot_line(axes[0], [0, 1], A,
              xlabel="System",
              title="A  |  Bigram Presence",
              proposed_x=1,
              x_is_categorical=True,
              xtick_labels=A_labels)
    axes[0].set_ylabel("Score", fontsize=FONTSIZE_AXIS)

    # ── B: PMI Threshold τ (line) ──────────────────────────────────────────
    plot_line(axes[1], B_tau, B,
              xlabel=r"PMI threshold $\tau$",
              title=r"B  |  PMI Threshold $\tau$",
              proposed_x=B_tau[B_proposed_idx])
    axes[1].set_ylabel("Score", fontsize=FONTSIZE_AXIS)

    # annotate bigram counts below x-axis ticks
    '''ax_b = axes[1]
    for xi, (tau, nb) in enumerate(zip(B_tau, B_bigrams)):
        ax_b.text(tau, -0.13, f"{nb//1000}k",
                  ha="center", va="top", fontsize=6.5,
                  color="0.45", transform=ax_b.get_xaxis_transform())
    ax_b.text(0.5, -0.18, "#bigrams", ha="center", va="top",
              fontsize=6.5, color="0.45",
              transform=ax_b.get_xaxis_transform())'''

    # ── C: Bigram Weight α (line) ──────────────────────────────────────────
    plot_line(axes[2], C_alpha, C,
              xlabel=r"Bigram weight $\alpha$",
              title=r"C  |  Bigram Weight $\alpha$",
              proposed_x=C_alpha[C_proposed_idx])
    
    axes[2].set_ylabel("Score", fontsize=FONTSIZE_AXIS)
  

    # ── D: HNSW K (line, log-x) ────────────────────────────────────────────
    ax_d = axes[3]
    for metric, color in COLORS.items():
        ax_d.semilogx(D_K, D[metric], color=color,
                      marker=MARKERS[metric], linewidth=LINEWIDTH,
                      markersize=MARKERSIZE, label=metric, zorder=3)
    ax_d.axvline(D_K[D_proposed_idx], color=PROPOSED_COLOR,
                 linestyle="--", linewidth=1.4, zorder=0, label="Proposed")
    ax_d.set_xticks(D_K)
    ax_d.set_xticklabels([str(k) for k in D_K], fontsize=FONTSIZE_TICK)
    ax_d.set_xlabel("Top-C HNSW candidates", fontsize=FONTSIZE_AXIS)
    ax_d.set_title("D  |  HNSW Candidate Count C", fontsize=FONTSIZE_TITLE,
                   fontweight="bold", pad=6)
    ax_d.set_ylim(0.0, 1.05)
    ax_d.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax_d.grid(True, which="both", alpha=0.3, linestyle="--")
    axes[3].set_ylabel("Score", fontsize=FONTSIZE_AXIS)

    # ── E: SW Reranking (line) ─────────────────────────────────────────────
    plot_line(axes[4], [0, 1], E,
              xlabel="SW presence",
              title="E  |  Smith-Waterman Reranking",
              proposed_x=1,
              x_is_categorical=True,
              xtick_labels=E_labels)
    
    axes[4].set_ylabel("Score", fontsize=FONTSIZE_AXIS)

    # ── shared legend below all plots ──────────────────────────────────────
    metric_handles = [
        mpatches.Patch(color=COLORS[m], label=m) for m in COLORS
    ]
    proposed_handle = mpatches.Patch(
        color=PROPOSED_COLOR, alpha=0.5, label="Proposed config"
    )
    fig.legend(
        handles=metric_handles + [proposed_handle],
        loc="lower center", ncol=6, fontsize=FONTSIZE_LEGEND,
        frameon=True, framealpha=0.9, edgecolor="0.8",
        bbox_to_anchor=(0.5, -0.18),
        prop={"weight": "bold", "size": FONTSIZE_LEGEND}
    )

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ablationplots.pdf",
                        help="Output path (pdf/png/svg)")
    args = parser.parse_args()
    main(args.out)
