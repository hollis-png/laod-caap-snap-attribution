"""E18: sampling variance of the main configuration (Gemma-3 4B + YOLO-World X, no generation seed).

The configuration was run twice more on the same 5,000 images with eval/run_batch_eval.py (see repeat.sh in the same folder of the results):
  python eval/run_batch_eval.py --n 5000 --annotations instances_val2017.json --images val2017 --predictions-out pred_rK.jsonl --report-out report_rK.json
Inputs: the two prediction files and the two reports. Output: corpus CAAP and SNAP of each repeat (compare with the original run: CAAP 0.2068, SNAP 0.4559),
the spread, and the number of images whose set of generated labels is identical in the two repeats.
Usage: python e18_repeat_summary.py <pred_r1.jsonl> <pred_r2.jsonl> <report_r1.json> <report_r2.json> <out.json>
"""
import json, sys
p1, p2, r1, r2, out = sys.argv[1:6]

def labels(path):
    d = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); d[r["image_id"]] = sorted({x["label"] for x in r["detections"]})
    return d

a, b = labels(p1), labels(p2)
assert set(a) == set(b), "the two repeats cover different images"
same = sum(a[i] == b[i] for i in a)
rep = [json.load(open(r)) for r in (r1, r2)]
res = {"n_images": len(a), "images_with_identical_label_set": same,
       "corpus_caap": [r["corpus_caap"] for r in rep], "corpus_snap": [r["corpus_snap"] for r in rep],
       "original_run": {"corpus_caap": 0.2068, "corpus_snap": 0.4559}}
allc, alls = res["corpus_caap"] + [0.2068], res["corpus_snap"] + [0.4559]
res["spread_caap"], res["spread_snap"] = max(allc) - min(allc), max(alls) - min(alls)
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
