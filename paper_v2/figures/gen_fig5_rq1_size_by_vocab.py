#!/usr/bin/env python3
"""Fig 5 -- RQ1 re-analysis: CAAP-TP rate by detection-box size, split by vocabulary category.

Data: selfreview_2026-10-09 (detections.csv on the remote host; see rq1_size_recheck.py).
Bins by detection-box area / image area: tiny <2%, small 2-8%, medium 8-25%, large >25%.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import plt, BLUE, ORANGE, SKY_BLUE, FIG_SINGLE
bins = ["tiny\n(<2%)", "small\n(2-8%)", "medium\n(8-25%)", "large\n(>25%)"]
series = {"native vocabulary": ([71.7, 83.4, 89.9, 94.7], BLUE), "plural of native": ([54.8, 76.7, 86.1, 84.5], SKY_BLUE), "out-of-vocabulary": ([17.4, 31.8, 45.9, 47.8], ORANGE)}
fig, ax = plt.subplots(figsize=FIG_SINGLE)
for name, (vals, col) in series.items():
    ax.plot(range(4), vals, marker="o", color=col, label=name)
    for x, v in enumerate(vals): ax.text(x, v + 2.5, f"{v:.0f}", ha="center", fontsize=8, color=col)
ax.set_xticks(range(4)); ax.set_xticklabels(bins, fontsize=8.5)
ax.set_ylim(0, 105); ax.set_ylabel("IoU-0.5 hit rate (%)"); ax.set_xlabel("Detection-box size (share of image area)")
ax.set_title("Vocabulary effect holds at every size", pad=10); ax.legend(loc="lower right")
out = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out, "fig5_rq1_size_by_vocab.pdf")); fig.savefig(os.path.join(out, "fig5_rq1_size_by_vocab.png"))
print("saved")
