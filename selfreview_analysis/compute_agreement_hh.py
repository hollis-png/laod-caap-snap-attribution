"""Human-human agreement. Usage: python compute_agreement_hh.py first.csv second.csv
Both CSVs are exports of human_review/index.html (columns id,label,human,note). Only ids labelled in BOTH are compared.
Reports agreement and Cohen's kappa (three categories, and valid vs not valid), per set (n=96 / n=200) and pooled, with Wilson/bootstrap intervals.
Use the FIRST-PASS labels of the first annotator (human_labels.csv) to keep the comparison independent of the A2 re-review."""
import csv, json, sys, random, collections, os
a = {r["id"]: r["human"] for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["human"]}
b = {r["id"]: r["human"] for r in csv.DictReader(open(sys.argv[2], encoding="utf-8")) if r["human"]}
key = {k["id"]: k["set"] for k in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer_key_DO_NOT_OPEN_BEFORE_REVIEW.json"), encoding="utf-8"))}
def kappa(pairs):
    n = len(pairs); po = sum(x == y for x, y in pairs) / n
    ca = collections.Counter(x for x, _ in pairs); cb = collections.Counter(y for _, y in pairs)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb)); return (po - pe) / (1 - pe) if pe < 1 else float("nan")
def boot(pairs, B=3000):
    rng = random.Random(1); ks = []
    for _ in range(B):
        k = kappa([rng.choice(pairs) for _ in pairs])
        if k == k: ks.append(k)
    ks.sort(); return ks[int(.025 * len(ks))], ks[int(.975 * len(ks))]
common = sorted(set(a) & set(b)); print("ids labelled by both:", len(common))
for name, ids in (("n=200 no-overlap", [i for i in common if key[i] == "no_overlap_n200"]), ("n=96 other-class", [i for i in common if key[i] == "other_class_n96"]), ("pooled", common)):
    if len(ids) < 5: continue
    p3 = [(a[i], b[i]) for i in ids]; p2 = [("valid" if x == "valid" else "not", "valid" if y == "valid" else "not") for x, y in p3]
    lo3, hi3 = boot(p3); lo2, hi2 = boot(p2)
    print(f"{name}: n={len(ids)} agreement={sum(x==y for x,y in p3)/len(ids):.3f} kappa3={kappa(p3):.3f} [{lo3:.2f},{hi3:.2f}] | valid-vs-not agreement={sum(x==y for x,y in p2)/len(ids):.3f} kappa2={kappa(p2):.3f} [{lo2:.2f},{hi2:.2f}]")
