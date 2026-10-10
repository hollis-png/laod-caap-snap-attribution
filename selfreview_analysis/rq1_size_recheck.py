"""RQ1 re-analysis. Usage: python rq1_size_recheck.py detections.csv instances_val2017.json
Compares (A) bucketing by the size of the best-overlapping GT box (original, selection-biased)
with (B) bucketing by the detection box itself, overall and split by vocabulary category."""
import csv, json, sys, collections
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
coco = json.load(open(sys.argv[2]))
img = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
anns = collections.defaultdict(list)
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]; anns[a["image_id"]].append((x, y, x + w, y + h))
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I; return I / U if U else 0
def bucket(r): return "tiny<2%" if r < .02 else "small2-8%" if r < .08 else "medium8-25%" if r < .25 else "large>25%"
order = ["tiny<2%", "small2-8%", "medium8-25%", "large>25%"]
A = collections.defaultdict(lambda: [0, 0]); B = collections.defaultdict(lambda: [0, 0]); C = collections.defaultdict(lambda: [0, 0])
for r in rows:
    iid = int(r["image_id"]); W, H = img[iid]; box = json.loads(r["box"]); tp = int(r["caap_tp"])
    best, bi = 0, -1
    for i, g in enumerate(anns[iid]):
        v = iou(box, g)
        if v > best: best, bi = v, i
    if bi >= 0:
        g = anns[iid][bi]; k = bucket((g[2]-g[0])*(g[3]-g[1])/(W*H)); A[k][0] += tp; A[k][1] += 1
    d = bucket((box[2]-box[0])*(box[3]-box[1])/(W*H)); B[d][0] += tp; B[d][1] += 1
    v = {"native": "native", "plural_of_native": "plural"}.get(r["vocab"], "oov"); C[(v, d)][0] += tp; C[(v, d)][1] += 1
print("A matched-GT size:", {k: (A[k][1], round(A[k][0]/A[k][1], 3)) for k in order})
print("B detection size :", {k: (B[k][1], round(B[k][0]/B[k][1], 3)) for k in order})
for v in ("native", "plural", "oov"): print(v, {k: (C[(v, k)][1], round(C[(v, k)][0]/max(1, C[(v, k)][1]), 3)) for k in order})
