"""Usage: python compute_b1.py b1_labels.csv
Human relation labels for the PPS-sampled (probability proportional to detection count) 50 labels.
Because the sample is PPS, the *unweighted* mean over sampled labels estimates the detection-weighted quantity.
The original per-label LLM-judge output could not be recovered, so this checks the paper's section-3.3 table
against human labels instead of measuring judge-human agreement."""
import csv, json, sys, collections, math
cases = {c["id"]: c for c in json.load(open("sample50_private.json", encoding="utf-8"))}
hum = {r["id"]: r["human"] for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["human"]}
paper = {"synonym": (89.3, 2143), "finer_grained": (65.6, 2156), "out_of_ontology": (12.0, 7179)}
tot = sum(v[1] for v in paper.values())
by = collections.defaultdict(list)
for i, h in hum.items(): by[h].append(cases[i])
n = len(hum); print(f"labelled {n}/50; unsure {len(by['unsure'])}")
for k, (rate, cnt) in paper.items():
    xs = by.get(k, [])
    m = sum(c["tp"] for c in xs) / len(xs) * 100 if xs else float("nan")
    print(f"{k:16s} human share {len(xs)/n*100:5.1f}%  (paper {cnt/tot*100:5.1f}%)   mean CAAP-TP {m:5.1f}%  (paper {rate}%)   n_labels {len(xs)}")
print("note: 50 labels, so rates carry wide intervals; this is a sanity check, not a re-estimate.")
