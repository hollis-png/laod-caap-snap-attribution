#!/usr/bin/env python3
"""Fig 2 (v2) -- decomposition of the 12,139 CAAP false positives.

Data: selfreview_2026-10-09/recompute_report.json (group sizes), author-labelled n=200 no-overlap audit
(valid 151 / ambiguous 17 / wrong 32), n=96 overlap-other-class audit
(valid 88 / ambiguous 2 / wrong 6). Audit shares are extrapolated to the group sizes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import plt, SKY_BLUE, ORANGE, VERMILLION, GREEN, GRAY, FIG_SINGLE

groups = [  # (label, size, [(share, color, name)])
    ("No overlap\n(3,940)", 3940, [(151/200, GREEN, "valid, unannotated"), (17/200, GRAY, "ambiguous"), (32/200, VERMILLION, "wrong")]),
    ("Overlaps only a\ndifferent-class box\n(6,494)", 6494, [(88/96, GREEN, "valid, unannotated"), (2/96, GRAY, "ambiguous"), (6/96, VERMILLION, "wrong")]),
    ("Loose box on the\nnamed object\n(1,705)", 1705, [(1.0, SKY_BLUE, "genuine loose box")]),
]
fig, ax = plt.subplots(figsize=FIG_SINGLE)
seen = set()
for i, (lab, size, parts) in enumerate(groups):
    bottom = 0
    for share, color, name in parts:
        h = share * size
        ax.bar(i, h, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, width=0.6,
               label=None if name in seen else name)
        seen.add(name)
        if h > 350:
            ax.text(i, bottom + h / 2, f"~{h:,.0f}" if len(parts) > 1 else f"{h:,.0f}", ha="center", va="center", fontsize=8, color="white" if color != GRAY else "black")
        bottom += h
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
ax.set_ylabel("IoU-0.5 failures (estimated)")
ax.set_title("What CAAP counts as failure (n = 12,139)", pad=10)
ax.legend(loc="upper right")
out = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out, "fig2_v2_fp_decomposition.pdf")); fig.savefig(os.path.join(out, "fig2_v2_fp_decomposition.png"))
print("saved")
