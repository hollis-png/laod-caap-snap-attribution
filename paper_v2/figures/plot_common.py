"""Shared matplotlib publication styling for LAOD paper figures (NeurIPS workshop).

Numbers are hand-transcribed from experiment_plan.md (full 5000-image corpus
results). This module has no data-fetching logic — it only holds style
constants shared by gen_fig_*.py scripts.
"""
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
})

# Okabe-Ito colorblind-safe palette (used for data charts per academic-plotting skill)
ORANGE = "#E69F00"
SKY_BLUE = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PINK = "#CC79A7"
GRAY = "#8C8C8C"

FIG_SINGLE = (5.5, 3.2)   # NeurIPS single/full column width per style-guide
FIG_SMALL = (4.5, 3.0)


def annotate_bars(ax, bars, values, n_values=None, fmt="{:.1f}%", fontsize=8):
    """Put value labels above each bar; optionally append n= below the % label."""
    for i, (bar, v) in enumerate(zip(bars, values)):
        label = fmt.format(v)
        if n_values is not None:
            label += f"\n(n={n_values[i]:,})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            label,
            ha="center", va="bottom", fontsize=fontsize, color="#333",
        )
