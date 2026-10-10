"""Judge-human agreement and re-computation of the section-3.3 table. Decision rule fixed before seeing outputs:
adopt the new classification if kappa(main judge, human) >= 0.6 on the labels whose words are NOT in the V1 prompt
AND kappa(deepseek v1, deepseek v2) >= 0.6 on all labels; otherwise downgrade section 3.3 to a descriptive statement."""
import json, csv, random, collections, math, sys
from prompts import V1_EXAMPLE_WORDS
CATS = ["synonym", "finer_grained", "out_of_ontology", "unsure"]
RUNS = {"deepseek_v1": "out_deepseek_v1.jsonl", "deepseek_v2": "out_deepseek_v2.jsonl", "mistral_v1": "out_mistral_v1.jsonl"}
def load(p):
    d = {}
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); d[r["label"]] = r
    return d
runs = {k: load(p) for k, p in RUNS.items()}
raw = {x["label"]: x for x in json.load(open("raw_full.json", encoding="utf-8")) if x.get("most_common_matched_gt") and x.get("n_with_gt_match", 0) > 0}
sample = {c["id"]: c for c in json.load(open("../human_review_b1/sample50_private.json", encoding="utf-8"))}
human = {r["id"]: r["human"] for r in csv.DictReader(open("../human_review_b1/b1_labels.csv", encoding="utf-8"))}
def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
def boot_kappa(pairs, B=3000, seed=1):
    rng = random.Random(seed); ks = []
    for _ in range(B):
        s = [rng.choice(pairs) for _ in pairs]; k = kappa(s)
        if k == k: ks.append(k)
    ks.sort(); return ks[int(.025 * len(ks))], ks[int(.975 * len(ks))]
out = {}; md = []
md.append("# Judge report\n")
md.append("## Run health\n\n| run | model returned | labels | parse failures | unsure | max attempts |\n|---|---|---|---|---|---|")
for k, d in runs.items():
    rs = list(d.values()); mr = collections.Counter(r.get("model_returned") for r in rs)
    md.append(f"| {k} | {dict(mr)} | {len(rs)} | {sum(r['parsed'] is None for r in rs)} | {sum(r['parsed']=='unsure' for r in rs)} | {max(r.get('attempts',0) for r in rs)} |")
non_ex = [i for i, c in sample.items() if c["label"] not in V1_EXAMPLE_WORDS]
md.append(f"\n## Judge vs human (B1, 50 labels; {len(non_ex)} not in prompt)\n")
md.append("| run | set | n | accuracy | Cohen kappa | kappa 95% bootstrap |\n|---|---|---|---|---|---|")
for k, d in runs.items():
    for name, ids in (("all 50", list(sample)), (f"non-example {len(non_ex)}", non_ex)):
        pairs = [(d[sample[i]["label"]]["parsed"], human[i]) for i in ids if sample[i]["label"] in d and d[sample[i]["label"]]["parsed"]]
        if not pairs: continue
        acc = sum(a == b for a, b in pairs) / len(pairs); kp = kappa(pairs); lo, hi = boot_kappa(pairs)
        md.append(f"| {k} | {name} | {len(pairs)} | {acc:.3f} | {kp:.3f} | {lo:.2f} to {hi:.2f} |"); out[f"{k}|{name}"] = dict(n=len(pairs), acc=acc, kappa=kp, lo=lo, hi=hi)
md.append("\nConfusion, deepseek_v1 (rows judge, cols human), non-example set:\n")
pairs = [(runs["deepseek_v1"][sample[i]["label"]]["parsed"], human[i]) for i in non_ex if sample[i]["label"] in runs["deepseek_v1"]]
cm = collections.Counter(pairs); md.append("| judge \\ human | synonym | finer_grained | out_of_ontology |\n|---|---|---|---|")
for a in CATS[:3]: md.append(f"| {a} | " + " | ".join(str(cm[(a, b)]) for b in CATS[:3]) + " |")
md.append("\nDisagreements (deepseek_v1, all 50): " + "; ".join(f"{sample[i]['label']}->{sample[i]['matched']} judge={runs['deepseek_v1'][sample[i]['label']]['parsed']} human={human[i]}" for i in sample if runs['deepseek_v1'].get(sample[i]['label'],{}).get('parsed') != human[i]))
md.append("\n## Judge vs judge (all labels both classified)\n\n| pair | n | agreement | kappa |\n|---|---|---|---|")
names = list(runs)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        common = [l for l in raw if runs[names[a]].get(l, {}).get("parsed") and runs[names[b]].get(l, {}).get("parsed")]
        pr = [(runs[names[a]][l]["parsed"], runs[names[b]][l]["parsed"]) for l in common]
        md.append(f"| {names[a]} vs {names[b]} | {len(pr)} | {sum(x == y for x, y in pr)/len(pr):.3f} | {kappa(pr):.3f} |"); out[f"{names[a]}~{names[b]}"] = dict(n=len(pr), kappa=kappa(pr))
md.append("\n## Section 3.3 table recomputed (weights n_with_gt_match; rate = TP / n)\n\n| judge | category | labels | n matched detections | CAAP-TP rate |\n|---|---|---|---|---|")
tab = {}
for k, d in runs.items():
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for l, x in raw.items():
        r = d.get(l, {}).get("parsed")
        if not r: continue
        n = x["n_with_gt_match"]; agg[r][0] += 1; agg[r][1] += n; agg[r][2] += x["caap_tp_rate"] * n
    for c in CATS:
        if agg[c][1]:
            md.append(f"| {k} | {c} | {agg[c][0]} | {agg[c][1]:,} | {100*agg[c][2]/agg[c][1]:.1f}% |"); tab[f"{k}|{c}"] = dict(labels=agg[c][0], n=agg[c][1], rate=agg[c][2] / agg[c][1])
md.append("\nOriginal (undocumented) judge: synonym 2,143 / 89.3%, finer_grained 2,156 / 65.6%, out_of_ontology 7,179 / 12.0%.")
md.append("\n## Re-check of the three caveats (deepseek_v1)\n")
d = runs["deepseek_v1"]
def show(ls):
    return "; ".join(f"{l}: {d[l]['parsed']} (TP {100*raw[l]['caap_tp_rate']:.0f}%, n={raw[l]['n_with_gt_match']})" for l in ls if l in d and l in raw)
md.append("- clothing: " + show(["suit", "dress", "jacket", "shirt", "t-shirt", "coat", "pants", "jeans"]))
md.append("- person descriptors: " + show(["man", "woman", "boy", "girl", "men", "women", "child", "surfer", "people"]))
md.append("- part-whole: " + show(["faucet", "shelves", "cushions", "cushion", "pillow", "pillows"]))
fg = [(l, raw[l]) for l in raw if d.get(l, {}).get("parsed") == "finer_grained"]
tot = sum(x["n_with_gt_match"] for _, x in fg)
hi = sum(x["n_with_gt_match"] for _, x in fg if x["caap_tp_rate"] >= 0.9); lo = sum(x["n_with_gt_match"] for _, x in fg if x["caap_tp_rate"] < 0.25)
md.append(f"- finer_grained bimodality: of {tot:,} matched detections, {100*hi/tot:.0f}% are in labels with TP >= 0.9 and {100*lo/tot:.0f}% in labels with TP < 0.25")
main = out.get(f"deepseek_v1|non-example {len(non_ex)}", {}); stab = out.get("deepseek_v1~deepseek_v2", {})
ok = main.get("kappa", 0) >= 0.6 and stab.get("kappa", 0) >= 0.6
md.append(f"\n## Decision rule\n\nkappa(deepseek_v1, human; non-example) = {main.get('kappa', float('nan')):.3f}; kappa(deepseek_v1, deepseek_v2) = {stab.get('kappa', float('nan')):.3f}. **{'ADOPT the new classification' if ok else 'DOWNGRADE section 3.3 to a descriptive statement'}.**")
open("REPORT.md", "w", encoding="utf-8").write("\n".join(md)); json.dump(dict(pairs=out, table=tab, adopt=ok), open("results.json", "w"), indent=1)
print("\n".join(md))
