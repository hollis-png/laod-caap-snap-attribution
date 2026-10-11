"""Two-way collapse: same_object = synonym + finer_grained; different_object = out_of_ontology (unsure dropped).
Run after analyze_judge.py. Not part of the pre-registered decision rule; reported as a robustness view."""
import json, csv, random, collections
from prompts import V1_EXAMPLE_WORDS
RUNS = {"deepseek_v1": "out_deepseek_v1.jsonl", "deepseek_v2": "out_deepseek_v2.jsonl", "mistral_v1": "out_mistral_v1.jsonl"}
def load(p):
    d = {}
    for l in open(p, encoding="utf-8"):
        if l.strip(): r = json.loads(l); d[r["label"]] = r
    return d
runs = {k: load(p) for k, p in RUNS.items()}
raw = {x["label"]: x for x in json.load(open("raw_full.json", encoding="utf-8")) if x.get("most_common_matched_gt") and x.get("n_with_gt_match", 0) > 0}
sample = {c["id"]: c for c in json.load(open("../human_review_b1/sample50_private.json", encoding="utf-8"))}
human = {r["id"]: r["human"] for r in csv.DictReader(open("../human_review_b1/b1_labels.csv", encoding="utf-8"))}
two = lambda r: None if r in (None, "unsure") else ("different" if r == "out_of_ontology" else "same")
def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb)); return (po - pe) / (1 - pe)
def boot(pairs, B=3000):
    rng = random.Random(2); ks = []
    for _ in range(B):
        smp = [rng.choice(pairs) for _ in pairs]
        try: ks.append(kappa(smp))
        except ZeroDivisionError: pass          # degenerate resample (single class): skip
    ks.sort(); return ks[int(.025 * len(ks))], ks[int(.975 * len(ks))]
res = {}; lines = ["| judge | set | n | agreement | kappa | 95% bootstrap |", "|---|---|---|---|---|---|"]
non_ex = [i for i, c in sample.items() if c["label"] not in V1_EXAMPLE_WORDS]
for k, d in runs.items():
    for name, ids in (("all 50", list(sample)), ("non-example 37", non_ex)):
        pr = [(two(d[sample[i]["label"]]["parsed"]), two(human[i])) for i in ids]
        pr = [p for p in pr if p[0] and p[1]]
        try: kp = kappa(pr); lo, hi = boot(pr)
        except ZeroDivisionError: kp = lo = hi = float("nan")
        lines.append(f"| {k} | {name} | {len(pr)} | {sum(a==b for a,b in pr)/len(pr):.3f} | {kp:.3f} | {lo:.2f} to {hi:.2f} |"); res[f"{k}|{name}"] = dict(n=len(pr), acc=sum(a==b for a,b in pr)/len(pr), kappa=kp)
lines += ["", "| judge | same-object labels | same-object n | same-object CAAP-TP | different-object labels | different-object n | different-object CAAP-TP | gap (pp) |", "|---|---|---|---|---|---|---|---|"]
for k, d in runs.items():
    a = {"same": [0, 0, 0], "different": [0, 0, 0]}
    for l, x in raw.items():
        t = two(d[l]["parsed"])
        if not t: continue
        n = x["n_with_gt_match"]; a[t][0] += 1; a[t][1] += n; a[t][2] += x["caap_tp_rate"] * n
    s, df = a["same"], a["different"]
    lines.append(f"| {k} | {s[0]} | {s[1]:,} | {100*s[2]/s[1]:.1f}% | {df[0]} | {df[1]:,} | {100*df[2]/df[1]:.1f}% | {100*(s[2]/s[1]-df[2]/df[1]):.1f} |")
    res[f"{k}|table"] = dict(same_n=s[1], same_rate=s[2]/s[1], diff_n=df[1], diff_rate=df[2]/df[1])
lines.append("\nOriginal undocumented judge, same collapse: same-object 4,299 / 77.4% (synonym 2,143 at 89.3% + finer-grained 2,156 at 65.6%), different-object 7,179 / 12.0%.")
open("REPORT_binary.md", "w", encoding="utf-8").write("\n".join(lines)); json.dump(res, open("results_binary.json", "w"), indent=1); print("\n".join(lines))
