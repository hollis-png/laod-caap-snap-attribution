#!/usr/bin/env python3
"""Fig 3 (v2) -- same-object vs different-object CAAP-TP rate under three documented judges.
Data: selfreview_2026-10-09/judge/results_binary.json (documented runs) and the v1 table (4,299 / 77.4%, 7,179 / 12.0%)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import plt, SKY_BLUE, VERMILLION, GRAY, FIG_SINGLE
r = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selfreview_2026-10-09", "judge", "results_binary.json")))
names = ["DeepSeek v1", "DeepSeek v2", "Mistral v1"]
same = [100 * r[f"{k}|table"]["same_rate"] for k in ("deepseek_v1", "deepseek_v2", "mistral_v1")]
diff = [100 * r[f"{k}|table"]["diff_rate"] for k in ("deepseek_v1", "deepseek_v2", "mistral_v1")]
fig, ax = plt.subplots(figsize=FIG_SINGLE); w = 0.36; x = range(len(names))
b1 = ax.bar([i - w/2 for i in x], same, w, color=SKY_BLUE, label="names the annotated object (same object)")
b2 = ax.bar([i + w/2 for i in x], diff, w, color=VERMILLION, label="names a different object")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}", ha="center", fontsize=8)
ax.axhline(80.9, color=GRAY, linestyle="--", linewidth=1.1, label="native COCO-80 vocabulary (80.9%)")
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0, 100); ax.set_ylabel("IoU-0.5 hit rate (%)")
ax.set_title("Label relation vs. localization, three judges", pad=10); ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.16), fontsize=7.5, frameon=False)
out = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out, "fig3_v2_judge_contrast.pdf")); fig.savefig(os.path.join(out, "fig3_v2_judge_contrast.png")); print("saved")
