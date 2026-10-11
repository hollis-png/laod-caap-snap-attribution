"""E15: what the second annotator's labels do to the 'valid but unannotated' estimate (section 3.4 / 3.4.1).

Inputs: human_review/human_labels_final.csv (annotator 1: first_pass and final), human_review_e5/human_labels_annotator2.csv (annotator 2,
columns id,label,human), human_review/cases.json (label text). 296 cases: 200 from the no-overlap stratum (3,940 detections), 96 from the
other-class-overlap stratum (6,493). Same estimator as e8_stuff_exclusion.py / section 3.4.1: valid share per stratum, weighted by stratum
size, Beta(k+1, n-k+1) posterior draws, percentile 95% interval.
Labelings compared: a1_first (annotator 1 before the re-review), a1_final, a2, both_valid (valid by BOTH annotators: a conservative reading
that treats any disagreement as not valid), either_valid (valid by at least one). Each also with the core stuff list set aside.
Usage: python e15_second_annotator.py <selfreview_dir> <out.json>
"""
import csv, json, sys, collections
import numpy as np

base, out_p = sys.argv[1], sys.argv[2]
cases = {c["id"]: c for c in json.load(open(f"{base}/human_review/cases.json"))}
a1 = {r["id"]: r for r in csv.DictReader(open(f"{base}/human_review/human_labels_final.csv", encoding="utf-8"))}
a2 = {r["id"]: r["human"] for r in csv.DictReader(open(f"{base}/human_review_e5/human_labels_annotator2.csv", encoding="utf-8-sig"))}
assert set(a1) == set(a2) == set(cases), "id sets differ"
STRATUM = {"no_overlap_n200": 3940, "other_class_n96": 6468}; N_FAIL = 12139; N_DET = 27273; N_HIT = 15134
CORE = {"sky", "wall", "field", "beach", "sea", "baseball field"}
lab = {"a1_first": {i: a1[i]["first_pass"] for i in a1}, "a1_final": {i: a1[i]["final"] for i in a1}, "a2": dict(a2)}
lab["both_valid"] = {i: "valid" if lab["a1_first"][i] == "valid" and a2[i] == "valid" else "not_valid" for i in a1}
lab["either_valid"] = {i: "valid" if lab["a1_first"][i] == "valid" or a2[i] == "valid" else "not_valid" for i in a1}
rng = np.random.default_rng(20261010); D = 200000
def estimate(labels, drop_stuff):
    k = {s: 0 for s in STRATUM}; n = {s: 0 for s in STRATUM}
    for i, r in a1.items():
        s = r["set"]; n[s] += 1
        valid = labels[i] == "valid" and not (drop_stuff and cases[i]["label"].lower().strip() in CORE)
        k[s] += valid
    pt = sum(STRATUM[s] * k[s] / n[s] for s in STRATUM) / N_FAIL
    dr = sum(STRATUM[s] * rng.beta(k[s] + 1, n[s] - k[s] + 1, D) for s in STRATUM) / N_FAIL
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return dict(valid_counts=k, n=n, share=pt, lo=float(lo), hi=float(hi), precision=(N_HIT + pt * N_FAIL) / N_DET)
res = {name: {"all": estimate(l, False), "stuff_set_aside": estimate(l, True)} for name, l in lab.items()}
conf = collections.Counter((lab["a1_first"][i], a2[i]) for i in a1)
res["confusion_a1first_rows_a2_cols"] = {f"{x}|{y}": c for (x, y), c in sorted(conf.items())}
res["disagreement_labels_valid_vs_not"] = collections.Counter(cases[i]["label"].lower() for i in a1 if (lab["a1_first"][i] == "valid") != (a2[i] == "valid")).most_common(15)
json.dump(res, open(out_p, "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
