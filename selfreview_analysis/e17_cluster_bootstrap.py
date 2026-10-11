"""E17: image-clustered bootstrap intervals and same-denominator comparison for the headline rates (review round 3, points 5 and 6).

Detections of one image are not independent, so Wilson/Beta intervals that treat the 27,273 detections as independent are too narrow. Here the
unit of resampling is the image (5,000 images, 2,000 resamples, seed 20261010); every rate is recomputed on each resample.
Rates (CAAP-TP = caap_tp column of detections.csv, IoU >= 0.5 with a GT box of the same image):
  vocab     native / plural_of_native / out_of_vocabulary rate with all detections as denominator, and with the denominator restricted to
            detections that overlap some GT box (best_iou > 0), and the native - OOV gaps under both denominators
  overall   hit rate over all detections
  samediff  the section-3.3 contrast with the text-only DeepSeek judge: labels judged to name a COCO category ('same') vs 'none' ('different'),
            denominator = detections that overlap some GT box, same eligibility as judge/run_textonly.py (labels with n_with_gt_match > 0 and a
            matched COCO category in raw_full.json). The 'same' rate is then comparable with 'native' under the matched denominator.
Usage: python e17_cluster_bootstrap.py <detections.csv> <judge_dir> <out.json>
"""
import csv, json, sys
import numpy as np

det_p, judge_dir, out_p = sys.argv[1:4]
rows = list(csv.DictReader(open(det_p, encoding="utf-8")))
img_ids = sorted({int(r["image_id"]) for r in rows}); idx = {k: i for i, k in enumerate(img_ids)}; n_img = len(img_ids)
raw = {x["label"]: x for x in json.load(open(f"{judge_dir}/raw_full.json", encoding="utf-8")) if x.get("most_common_matched_gt") and x.get("n_with_gt_match", 0) > 0}
judge = {}
for l in open(f"{judge_dir}/out_textonly_deepseek.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        if r.get("parsed") is not None: judge[r["label"]] = "different" if r["parsed"] == "none" else "same"
GROUPS = ["native", "plural_of_native", "out_of_vocabulary", "same", "different"]
# per image and group: [n_all, tp_all, n_matched, tp_matched]   (matched = best_iou > 0)
cnt = np.zeros((n_img, len(GROUPS), 4)); overall = np.zeros((n_img, 2))
for r in rows:
    i = idx[int(r["image_id"])]; tp = int(r["caap_tp"] == "1"); m = float(r["best_iou"]) > 0
    overall[i] += (1, tp)
    gs = [GROUPS.index(r["vocab"])]
    lab = r["label"]
    if lab in raw and lab in judge and m: gs.append(GROUPS.index(judge[lab]))
    for g in gs:
        cnt[i, g, 0] += 1; cnt[i, g, 1] += tp
        if m: cnt[i, g, 2] += 1; cnt[i, g, 3] += tp

def stats(c, ov):
    t = c.sum(0); o = ov.sum(0); out = {"overall": o[1] / o[0]}
    for k, g in enumerate(GROUPS[:3]):
        out[f"{g}_all"] = t[k, 1] / t[k, 0]; out[f"{g}_matched"] = t[k, 3] / t[k, 2]
    out["gap_all_pp"] = 100 * (out["native_all"] - out["out_of_vocabulary_all"]); out["gap_matched_pp"] = 100 * (out["native_matched"] - out["out_of_vocabulary_matched"])
    for g in ("same", "different"):
        k = GROUPS.index(g); out[f"{g}_matched"] = t[k, 3] / t[k, 2] if t[k, 2] else float("nan")
    out["samediff_gap_pp"] = 100 * (out["same_matched"] - out["different_matched"])
    out["same_minus_native_matched_pp"] = 100 * (out["same_matched"] - out["native_matched"])
    return out

point = stats(cnt, overall)
rng = np.random.default_rng(20261010); B = 2000; draws = []
for _ in range(B):
    s = rng.integers(0, n_img, n_img); draws.append(stats(cnt[s], overall[s]))
res = {"n_images": n_img, "bootstrap_resamples": B,
       "counts": {g: dict(n_all=int(cnt[:, k, 0].sum()), n_matched=int(cnt[:, k, 2].sum())) for k, g in enumerate(GROUPS)}, "estimates": {}}
for k, v in point.items():
    d = np.array([x[k] for x in draws]); lo, hi = np.percentile(d, [2.5, 97.5]); pp = k.endswith("pp"); f = 1 if pp else 100
    res["estimates"][k] = dict(point=float(v), lo=float(lo), hi=float(hi))
    print(f"{k:32s} {f*v:7.2f}  [{f*lo:.2f}, {f*hi:.2f}]" + ("" if pp else " %"))
print(res["counts"])
json.dump(res, open(out_p, "w"), indent=1)
