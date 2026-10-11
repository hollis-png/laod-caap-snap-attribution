#!/usr/bin/env python3
"""Fig 1 — RQ2: CAAP-TP rate by vocabulary commonness (native / plural / out-of-vocab).

Data source: experiment_plan.md, section "5000 張全量主評測執行結果", RQ2 row.
native: 80.9% (n=11557), out-of-vocab: 31.6% (n=13290), plural_of_native: 65.6% (n=2426)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import plt, ORANGE, SKY_BLUE, VERMILLION, FIG_SMALL, annotate_bars

CATEGORIES = ["Native COCO-80", "Plural of native", "Out-of-vocabulary"]
TP_RATES = [80.9, 66.8, 31.0]
N_VALUES = [11557, 2560, 13156]
COLORS = [SKY_BLUE, ORANGE, VERMILLION]

fig, ax = plt.subplots(figsize=FIG_SMALL)
x = range(len(CATEGORIES))
bars = ax.bar(x, TP_RATES, width=0.55, color=COLORS, edgecolor="white", linewidth=0.5)
annotate_bars(ax, bars, TP_RATES, N_VALUES)

ax.set_xticks(list(x))
ax.set_xticklabels(CATEGORIES)
ax.set_ylabel("IoU-0.5 hit rate (%)")
ax.set_ylim(0, 100)
ax.set_title("Localization accuracy by vocabulary commonness")

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "fig1_rq2_vocab_commonness.pdf"))
fig.savefig(os.path.join(out_dir, "fig1_rq2_vocab_commonness.png"))
print("Saved fig1_rq2_vocab_commonness.{pdf,png}")
