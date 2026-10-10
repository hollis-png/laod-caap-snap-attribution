"""E6 (exploratory): class-agnostic AP@0.5 with soft validity labels.
Detections that the AP matcher counts as false positives are split into groups (same rule as the main analysis); each group gets a
probability of being a valid object that COCO does not annotate, taken from the author audits (no-overlap 151/200, other-class overlap
86/96; duplicates of annotated objects and rule-based loose boxes get 0 in the base case, 0.24 in the COCO-ReM sensitivity case).
A valid detection is assumed to be a distinct new object with probability `delta` (the rest are duplicates of another valid detection).
Expected newly recognised objects are added to the ground-truth count. Objects that COCO does not annotate AND the pipeline misses are
unknowable, so this is an optimistic estimate of AP against 'annotated plus detected-valid' objects, not a measured AP."""
import json, csv, sys
import numpy as np
base = sys.argv[1]; coco = json.load(open(f"{base}/instances_val2017.json"))
rows = list(csv.DictReader(open(f"{base}/detections.csv", encoding="utf-8")))
G = {}
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]; G.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
G = {k: np.array(v, float) for k, v in G.items()}; E = np.zeros((0, 4)); NGT = sum(len(v) for v in G.values())
def iou(b, g):
    if len(g) == 0: return np.zeros(0)
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I
    return np.where(U > 0, I / U, 0)
person = {"man","men","woman","women","boy","girl","child","children","kid","kids","people","lady","guy","player","skier","surfer","skateboarder","snowboarder","baby","person","persons","crowd","pedestrian","rider","batter","pitcher","catcher","umpire","skiers","players","girls","boys"}
syn = {("sofa","couch"),("couch","couch"),("table","dining table"),("tables","dining table"),("luggage","suitcase"),("skis","skis"),("television","tv"),("phone","cell phone")}
def group(r):
    if r["caap_tp"] == "1": return "dup_of_annotated"            # IoU>=0.5 with an annotated box but not matched one-to-one
    if float(r["best_iou"]) < 0.02: return "no_overlap"
    l = r["label"].lower(); n = r["near_cls"]
    if r["exact"] == "1" or (n == "person" and l in person) or (l, n) in syn: return "loose_rule"
    return "overlap_other_class"
order = sorted(range(len(rows)), key=lambda k: -float(rows[k]["conf"]))
matched = {}; tp = np.zeros(len(order)); grp = []
for pos, k in enumerate(order):
    r = rows[k]; iid = int(r["image_id"]); g = G.get(iid, E); grp.append(group(r))
    if len(g) == 0: continue
    m = matched.setdefault(iid, np.zeros(len(g), bool)); v = iou(np.array(json.loads(r["box"])), g); v[m] = 0; j = int(np.argmax(v))
    if v[j] >= 0.5 and v[j] > 0: tp[pos] = 1; m[j] = True
grp = np.array(grp)
def ap(soft_tp, ngt):
    c = np.cumsum(soft_tp); f = np.cumsum(1 - soft_tp); p = c / (c + f); r = c / ngt; ip = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum(ip * np.diff(np.concatenate(([0.0], r)))))
base_ap = ap(tp, NGT); fp_mask = tp == 0
print("baseline AP@0.5", round(base_ap, 4), "TP", int(tp.sum()), "GT", NGT, "AP-FP", int(fp_mask.sum()), {g: int(((grp == g) & fp_mask).sum()) for g in set(grp)}, flush=True)
rng = np.random.default_rng(0); res = {}
def draws(case, delta, B=1000):
    out = []
    for _ in range(B):
        pv = {"no_overlap": rng.beta(151 + 1, 49 + 1), "overlap_other_class": rng.beta(86 + 1, 10 + 1),
              "loose_rule": 0.0 if case == "base" else rng.beta(409 + 1, 1296 + 1), "dup_of_annotated": 0.0}
        p = np.array([pv[g] for g in grp]) * fp_mask * delta
        out.append(ap(tp + (1 - tp) * p, NGT + p.sum()))
    a = np.array(out); return dict(mean=float(a.mean()), lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)))
for case in ("base", "with_loose_at_corem_rate"):
    for delta in (1.0, 0.75, 0.5):
        res[f"{case}|delta={delta}"] = draws(case, delta); print(case, delta, res[f"{case}|delta={delta}"], flush=True)
res["baseline_ap50"] = base_ap
json.dump(res, open(sys.argv[2], "w"), indent=1)
