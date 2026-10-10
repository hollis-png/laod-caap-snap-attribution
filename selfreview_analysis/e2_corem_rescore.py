"""E2: re-score the 27,273 detections against COCO-ReM val (refined COCO masks, with additional instances of the same 80 classes).
Per image, class-agnostic, IoU>=0.5. For COCO failures that COCO-ReM matches, record whether the matching box is an original COCO
instance (refined box) or an instance added by COCO-ReM ('source' field)."""
import json, csv, sys, collections
import numpy as np
base = sys.argv[1]; rem = json.load(open(sys.argv[2])); rows = list(csv.DictReader(open(f"{base}/detections.csv", encoding="utf-8")))
src = collections.Counter(a.get("source") for a in rem["annotations"]); print("REM sources", dict(src), flush=True)
g = collections.defaultdict(list)
for a in rem["annotations"]:
    x, y, w, h = a["bbox"]; g[a["image_id"]].append((x, y, x + w, y + h, 0 if a.get("source") == "coco" else 1))
G = {k: np.array(v, float) for k, v in g.items()}; E = np.zeros((0, 5))
def iou(b, gg):
    if len(gg) == 0: return np.zeros(0)
    ix = np.clip(np.minimum(b[2], gg[:, 2]) - np.maximum(b[0], gg[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], gg[:, 3]) - np.maximum(b[1], gg[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (gg[:, 2]-gg[:, 0])*(gg[:, 3]-gg[:, 1]) - I
    return np.where(U > 0, I / U, 0)
person = {"man","men","woman","women","boy","girl","child","children","kid","kids","people","lady","guy","player","skier","surfer","skateboarder","snowboarder","baby","person","persons","crowd","pedestrian","rider","batter","pitcher","catcher","umpire","skiers","players","girls","boys"}
syn = {("sofa","couch"),("couch","couch"),("table","dining table"),("tables","dining table"),("luggage","suitcase"),("skis","skis"),("television","tv"),("phone","cell phone")}
def group(r):
    if r["caap_tp"] == "1": return "coco_tp"
    if float(r["best_iou"]) < 0.02: return "fail_no_overlap"
    l = r["label"].lower(); n = r["near_cls"]
    if r["exact"] == "1" or (n == "person" and l in person) or (l, n) in syn: return "fail_loose_box_on_named_object"
    return "fail_overlap_other_class"
agg = collections.defaultdict(lambda: collections.Counter()); vocab = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    iid = int(r["image_id"]); b = json.loads(r["box"]); gg = G.get(iid, E); v = iou(b, gg)
    j = int(np.argmax(v)) if len(v) else -1; hit = j >= 0 and v[j] >= 0.5; new = bool(hit and gg[j, 4] == 1)
    k = group(r); a = agg[k]; a["n"] += 1; a["rem_match"] += int(hit); a["rem_match_added_instance"] += int(new)
    t = {"native": "native", "plural_of_native": "plural"}.get(r["vocab"], "oov"); x = vocab[t]; x[0] += 1; x[1] += int(r["caap_tp"] == "1"); x[2] += int(r["caap_tp"] == "1" or bool(hit))
out = {"groups": {k: dict(v) for k, v in agg.items()}}
fail = [k for k in agg if k != "coco_tp"]; fn = sum(agg[k]["n"] for k in fail); fr = sum(agg[k]["rem_match"] for k in fail); fa = sum(agg[k]["rem_match_added_instance"] for k in fail)
out.update(coco_fail_n=fn, rem_rescued=fr, rem_rescue_rate=fr / fn, rem_rescued_by_added_instance=fa, rem_rescued_added_rate=fa / fn)
out["by_vocab"] = {k: dict(n=v[0], coco=v[1] / v[0], coco_or_rem=v[2] / v[0]) for k, v in vocab.items()}
json.dump(out, open(sys.argv[3], "w"), indent=1); print(json.dumps(out, indent=1))
