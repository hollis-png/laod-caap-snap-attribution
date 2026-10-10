"""E1: re-score the 27,273 detections against LVIS v1 val annotations (class-agnostic, per image).
LVIS is federated (each image is exhaustively labelled only for some categories), so a rise in the TP rate is a LOWER BOUND on
how many COCO 'failures' are real objects. Same IoU>=0.5 per-image rule as the main analysis, plus one-to-one class-agnostic AP@0.5."""
import json, csv, sys, collections
import numpy as np
base = sys.argv[1]            # dir holding instances_val2017.json, coco_val_subset_predictions.jsonl, detections.csv
lvis = json.load(open(sys.argv[2]))
coco = json.load(open(f"{base}/instances_val2017.json"))
rows = list(csv.DictReader(open(f"{base}/detections.csv", encoding="utf-8")))
ids_ours = {i["id"] for i in coco["images"]}
lv_img = {i["id"]: i for i in lvis["images"]}
cov = sorted(ids_ours & set(lv_img))
print("our images", len(ids_ours), "LVIS val images", len(lv_img), "overlap", len(cov), flush=True)
def boxes(anns):
    d = collections.defaultdict(list)
    for a in anns:
        x, y, w, h = a["bbox"]; d[a["image_id"]].append((x, y, x + w, y + h))
    return {k: np.array(v, float) for k, v in d.items()}
cg = boxes(coco["annotations"]); lg = boxes(lvis["annotations"])
E = np.zeros((0, 4))
def iou(b, g):
    if len(g) == 0: return np.zeros(0)
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I
    return np.where(U > 0, I / U, 0)
person = {"man","men","woman","women","boy","girl","child","children","kid","kids","people","lady","guy","player","skier","surfer","skateboarder","snowboarder","baby","person","persons","crowd","pedestrian","rider","batter","pitcher","catcher","umpire","skiers","players","girls","boys"}
syn = {("sofa","couch"),("couch","couch"),("table","dining table"),("tables","dining table"),("luggage","suitcase"),("skis","skis"),("television","tv"),("phone","cell phone")}
def group(r):
    if float(r["caap_tp"]) == 1: return "coco_tp"
    if float(r["best_iou"]) < 0.02: return "fail_no_overlap"
    l = r["label"].lower(); n = r["near_cls"]
    if r["exact"] == "1" or (n == "person" and l in person) or (l, n) in syn: return "fail_loose_box_on_named_object"
    return "fail_overlap_other_class"
covset = set(cov); out = {"images_overlap": len(cov)}
agg = collections.defaultdict(lambda: [0, 0, 0, 0]); vocab = collections.defaultdict(lambda: [0, 0, 0, 0])
dets = []
for r in rows:
    iid = int(r["image_id"])
    if iid not in covset: continue
    b = json.loads(r["box"]); c_tp = r["caap_tp"] == "1"
    lv = iou(b, lg.get(iid, E)); l_best = float(lv.max()) if len(lv) else 0.0; l_tp = l_best >= 0.5
    g = group(r); a = agg[g]; a[0] += 1; a[1] += l_tp; a[2] += (c_tp or l_tp)
    v = {"native": "native", "plural_of_native": "plural"}.get(r["vocab"], "oov"); x = vocab[v]; x[0] += 1; x[1] += c_tp; x[2] += l_tp; x[3] += (c_tp or l_tp)
    dets.append((float(r["conf"]), iid, b))
n = len(dets); coco_tp = sum(v[0] for k, v in agg.items() if k == "coco_tp")
out["detections_in_overlap"] = n
tot = collections.Counter({k: v[0] for k, v in agg.items()})
out["groups"] = {k: dict(n=v[0], lvis_match=v[1], lvis_match_rate=v[1] / v[0] if v[0] else None) for k, v in agg.items()}
fail_n = sum(v[0] for k, v in agg.items() if k != "coco_tp"); fail_res = sum(v[1] for k, v in agg.items() if k != "coco_tp")
out["coco_fail_n"] = fail_n; out["coco_fail_rescued_by_lvis"] = fail_res; out["coco_fail_rescue_rate"] = fail_res / fail_n
out["tp_rate_coco"] = coco_tp / n; out["tp_rate_lvis"] = (agg["coco_tp"][1] + fail_res) / n; out["tp_rate_union"] = (coco_tp + fail_res) / n
out["by_vocab"] = {k: dict(n=v[0], coco=v[1] / v[0], lvis=v[2] / v[0], union=v[3] / v[0]) for k, v in vocab.items()}
# one-to-one class-agnostic AP@0.5, pooled, per-image matching
dets.sort(key=lambda x: -x[0])
def ap(gt_by_img):
    matched = {i: np.zeros(len(gt_by_img.get(i, E)), bool) for i in cov}; tp = np.zeros(len(dets))
    for k, (cf, iid, b) in enumerate(dets):
        g = gt_by_img.get(iid, E)
        if len(g) == 0: continue
        v = iou(b, g); v[matched[iid]] = 0; j = int(np.argmax(v))
        if v[j] >= 0.5 and v[j] > 0: tp[k] = 1; matched[iid][j] = True
    ngt = sum(len(gt_by_img.get(i, E)) for i in cov)
    c = np.cumsum(tp); f = np.cumsum(1 - tp); p = c / (c + f); r = c / ngt; ip = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum(ip * np.diff(np.concatenate(([0.0], r))))), int(tp.sum()), ngt
union = {}
for i in cov:
    cb = cg.get(i, E); lb = lg.get(i, E); keep = [b for b in lb if len(cb) == 0 or iou(b, cb).max() < 0.7]
    union[i] = np.vstack([cb, np.array(keep).reshape(-1, 4)]) if len(keep) else cb
for name, g in (("coco", cg), ("lvis", lg), ("union_dedup", union)):
    a_, t_, n_ = ap(g); out[f"ap50_{name}"] = dict(ap=a_, tp=t_, n_gt=n_)
json.dump(out, open(sys.argv[3], "w"), indent=1); print(json.dumps(out, indent=1))
