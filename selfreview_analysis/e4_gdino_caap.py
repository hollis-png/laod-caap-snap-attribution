import json, sys, time
sys.path.insert(0, ".")
from eval.caap import compute_ca_ap
coco = json.load(open("instances_val2017.json")); cat = {c["id"]: c["name"] for c in coco["categories"]}
def run(path):
    preds = {}
    for l in open(path):
        if l.strip(): r = json.loads(l); preds[r["image_id"]] = r["detections"]
    ids = sorted(i["id"] for i in coco["images"] if i["id"] in preds)
    gts = {i: [] for i in ids}
    for a in coco["annotations"]:
        if a["image_id"] in gts: x, y, w, h = a["bbox"]; gts[a["image_id"]].append([x, y, x + w, y + h, cat[a["category_id"]]])
    P = [[d["box"] + [d["confidence"]] for d in preds[i]] for i in ids]; G = [gts[i] for i in ids]
    res = {"images": len(ids), "detections": sum(len(p) for p in P)}
    for name, th in (("LO", [.5, .55, .6]), ("MI", [.65, .7, .75, .8]), ("HI", [.85, .9, .95]), ("mean", [0.5 + 0.05 * k for k in range(10)])):
        res[name] = round(float(compute_ca_ap(P, G, th)), 4)
    return res
out = {n: run(p) for n, p in (("gemma3_yoloworld", "coco_val_subset_predictions.jsonl"), ("gemma3_groundingdino", "gdino_predictions.jsonl"))}
print(json.dumps(out, indent=1)); json.dump(out, open("e4_result.json", "w"), indent=1)
