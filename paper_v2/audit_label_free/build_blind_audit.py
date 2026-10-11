"""Build the blind audit form (see PREREGISTRATION.md).

Inputs: detections.csv (image_id,label,conf,box,best_iou,caap_tp,...), human_review/cases.json and the answer key of the earlier audit
(to re-use the same test cases), COCO instances_val2017.json (ground truth, only to place the negative controls), the COCO val2017 image folder.
Output folder: index.html, img/Bxxx_full.jpg + img/Bxxx_crop.jpg (box only: no label, no ground truth, no model name),
items_public.json (ids only), answer_key_PRIVATE.json (kind, pool, label of each item; do not open before the annotators have finished).
Usage: python build_blind_audit.py <detections.csv> <cases.json> <answer_key.json> <instances_val2017.json> <val2017_dir> <out_dir>
COCO images must not be redistributed (Flickr terms): the output folder is for the annotators only and is not part of the released artifacts.
"""
import csv, json, os, random, sys
from PIL import Image, ImageDraw

det_csv, cases_json, key_json, coco_json, img_dir, out = sys.argv[1:7]
SEED, N_NO, N_OTHER, N_POS, N_SHIFT, N_SHRINK = 20261011, 65, 35, 50, 35, 15
rng = random.Random(SEED)
os.makedirs(f"{out}/img", exist_ok=True)

rows = []
for r in csv.DictReader(open(det_csv, encoding="utf-8")):
    r["box"] = json.loads(r["box"]); r["conf"] = float(r["conf"]); r["best_iou"] = float(r["best_iou"])
    r["caap_tp"] = int(r["caap_tp"]); r["image_id"] = int(r["image_id"]); rows.append(r)
by_img_label = {}
for r in rows: by_img_label.setdefault((r["image_id"], r["label"]), []).append(r)

coco = json.load(open(coco_json))
size = {i["id"]: (i["width"], i["height"], i["file_name"]) for i in coco["images"]}
gt = {}
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]; gt.setdefault(a["image_id"], []).append([x, y, x + w, y + h])

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0

cases = {c["id"]: c for c in json.load(open(cases_json))}
key = {k["id"]: k for k in json.load(open(key_json))}
pools = {"no_overlap": [], "other_class": []}; unmatched = 0
for cid, k in key.items():
    c = cases[cid]
    m = [r for r in by_img_label.get((k["image_id"], c["label"]), []) if abs(r["conf"] - c["conf"]) < 0.006]
    if len(m) == 1: pools["no_overlap" if k["set"].startswith("no_overlap") else "other_class"].append((cid, m[0]))
    else: unmatched += 1
test = rng.sample(pools["no_overlap"], N_NO) + rng.sample(pools["other_class"], N_OTHER)
used_images = {r["image_id"] for _, r in test}

cand = [r for r in rows if r["caap_tp"] == 1 and r["best_iou"] >= 0.6 and r["image_id"] not in used_images and r["image_id"] in size]
rng.shuffle(cand)
seen = set(); distinct = []
for r in cand:
    if r["image_id"] not in seen: seen.add(r["image_id"]); distinct.append(r)
pos = distinct[:N_POS]; rest = distinct[N_POS:]

def shifted(box, W, H, G):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    for _ in range(300):
        if rng.random() < 0.5: dx, dy = rng.choice([-1, 1]) * rng.uniform(1.1, 2.0) * w, 0
        else: dx, dy = 0, rng.choice([-1, 1]) * rng.uniform(1.1, 2.0) * h
        nb = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        if nb[0] < 0 or nb[1] < 0 or nb[2] > W or nb[3] > H: continue
        if all(iou(nb, g) < 0.02 for g in G): return nb
    return None

shift_items, k = [], 0
while len(shift_items) < N_SHIFT and k < len(rest):
    r = rest[k]; k += 1
    W, H, _ = size[r["image_id"]]; nb = shifted(r["box"], W, H, gt.get(r["image_id"], []))
    if nb: shift_items.append((r, nb))
assert len(shift_items) == N_SHIFT, "not enough shiftable boxes"
shrink_items = []
for r in rest[k:k + N_SHRINK]:
    x1, y1, x2, y2 = r["box"]; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.6, (y2 - y1) * 0.6
    shrink_items.append((r, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
assert len(shrink_items) == N_SHRINK

items = ([("test_" + ("no_overlap" if cid in {c for c, _ in pools["no_overlap"]} else "other_class"), r, r["box"], cid) for cid, r in test]
         + [("pos", r, r["box"], None) for r in pos] + [("neg_shift", r, nb, None) for r, nb in shift_items]
         + [("neg_shrink", r, nb, None) for r, nb in shrink_items])
rng.shuffle(items)

def render(path, box, full_out, crop_out):
    im = Image.open(path).convert("RGB"); W, H = im.size
    s = 900 / max(W, H); f = im.resize((round(W * s), round(H * s)), Image.LANCZOS); d = ImageDraw.Draw(f)
    d.rectangle([v * s for v in box], outline=(255, 0, 0), width=3); f.save(full_out, quality=88)
    x1, y1, x2, y2 = box; mx, my = max(0.3 * (x2 - x1), 40), max(0.3 * (y2 - y1), 40)
    cb = [max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)]
    c = im.crop([round(v) for v in cb]); t = 600 / max(c.size); t = min(t, 4)
    c = c.resize((max(1, round(c.size[0] * t)), max(1, round(c.size[1] * t))), Image.LANCZOS); d = ImageDraw.Draw(c)
    d.rectangle([(x1 - cb[0]) * t, (y1 - cb[1]) * t, (x2 - cb[0]) * t, (y2 - cb[1]) * t], outline=(255, 0, 0), width=2); c.save(crop_out, quality=88)

private, public = {}, []
for n, (kind, r, box, cid) in enumerate(items, 1):
    iid = f"B{n:03d}"; fn = size[r["image_id"]][2]
    render(f"{img_dir}/{fn}", box, f"{out}/img/{iid}_full.jpg", f"{out}/img/{iid}_crop.jpg")
    public.append({"id": iid})
    private[iid] = dict(kind=kind, old_case_id=cid, image_id=r["image_id"], box=box, model_label=r["label"], conf=r["conf"],
                        source_box=r["box"], best_iou=r["best_iou"])
json.dump(public, open(f"{out}/items_public.json", "w"))
json.dump(private, open(f"{out}/answer_key_PRIVATE.json", "w"), indent=1)
counts = {}
for v in private.values(): counts[v["kind"]] = counts.get(v["kind"], 0) + 1
print("items:", len(items), counts, "| old cases not re-matched uniquely:", unmatched)

tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "form_template.html"), encoding="utf-8").read()
open(f"{out}/index.html", "w", encoding="utf-8").write(tpl.replace("__ITEMS__", json.dumps(public)))
