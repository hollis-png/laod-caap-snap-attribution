#!/usr/bin/env python3
"""Fig 4 — Manual classification of the "no overlap" subset (n=200, merged sample).

v2: author labels (selfreview_2026-10-09/human_review/human_labels.csv), n=200:
valid 75.5% (n=151, Wilson 95% CI [69.1%, 80.9%]), wrong 16.0% (n=32, [11.6%, 21.7%]), ambiguous 8.5% (n=17).
The earlier AI-assisted labels (83.5% / 7.5% / 9.0%) are drawn as grey markers for comparison.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import plt, SKY_BLUE, VERMILLION, GRAY, FIG_SMALL

CATEGORIES = ["Valid, COCO\nunannotated", "Wrong\n(real failure)", "Ambiguous"]
SHARE = [75.5, 16.0, 8.5]
N_VALUES = [151, 32, 17]
AI_SHARE = [83.5, 7.5, 9.0]
COLORS = [SKY_BLUE, VERMILLION, GRAY]
# Wilson 95% CI half-widths, gt_gap only (from experiment_plan.md); other two
# categories' CIs are not reported in the source table, so we only draw the
# error bar we can source and leave the others unadorned.
CI_LOW, CI_HIGH = 69.1, 80.9

fig, ax = plt.subplots(figsize=FIG_SMALL)
x = range(len(CATEGORIES))
bars = ax.bar(x, SHARE, width=0.55, color=COLORS, edgecolor="white", linewidth=0.5)

for i, (bar, v, n) in enumerate(zip(bars, SHARE, N_VALUES)):
    label = f"{v:.1f}%\n(n={n})"
    top = max(bar.get_height(), CI_HIGH if i == 0 else 0)
    ax.text(bar.get_x() + bar.get_width() / 2, top + 2,
            label, ha="center", va="bottom", fontsize=8, color="#333")

# Error bar for the valid share (Wilson 95% CI)
err_low = SHARE[0] - CI_LOW
err_high = CI_HIGH - SHARE[0]
ax.errorbar([0], [SHARE[0]], yerr=[[err_low], [err_high]],
            fmt="none", ecolor="#333", elinewidth=1.2, capsize=4, zorder=5)

ax.set_xticks(list(x))
ax.set_xticklabels(CATEGORIES, fontsize=8.5)
ax.set_ylabel("Share of manually inspected\ncases (%)")
ax.set_ylim(0, 115)
ax.set_title('"No overlap" detections, author labels (n=200)', pad=12)

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "fig4_no_overlap_inspection.pdf"))
fig.savefig(os.path.join(out_dir, "fig4_no_overlap_inspection.png"))
print("Saved fig4_no_overlap_inspection.{pdf,png}")
