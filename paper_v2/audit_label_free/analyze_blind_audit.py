"""Analysis of the blind audit, exactly as fixed in PREREGISTRATION.md.

Usage: python analyze_blind_audit.py <answer_key_PRIVATE.json> <old_labels.csv> <out.json> <annotator1.csv> [<annotator2.csv> ...]
       python analyze_blind_audit.py --selftest        (simulated annotators; checks that the script runs and recovers a known rate)
old_labels.csv = human_review/human_labels_final.csv (columns id, set, first_pass, ...).
Annotator CSV columns: id, choice (1-5), choice_name, word, note.  Mapping: 1 -> valid; 2, 4 -> wrong; 3, 5 -> ambiguous.
"""
import csv, json, sys, os, tempfile, random, collections
import numpy as np

POOLS = {"test_no_overlap": 3940, "test_other_class": 6468}; N_FAIL = 12139
MAP = {1: "valid", 2: "wrong", 3: "ambiguous", 4: "wrong", 5: "ambiguous"}

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def load(path):
    labels, words = {}, {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["choice"].strip(): labels[r["id"]] = MAP[int(r["choice"])]; words[r["id"]] = r.get("word", "")
    return labels, words

def rogan_gladen(q, sens, spec):
    den = sens + spec - 1
    return float(np.clip((q + spec - 1) / den, 0, 1)) if den > 0 else float("nan")

def estimate(lab, key, rng=None):
    """lab: id -> 'valid'/..; returns dict of sens, spec, per-pool q and corrected p, overall share of the 12,139 failures."""
    ids = {k: [i for i, v in key.items() if v["kind"] == k and i in lab] for k in ("pos", "neg_shift", "neg_shrink", *POOLS)}
    neg = ids["neg_shift"] + ids["neg_shrink"]
    samp = (lambda l: [l[j] for j in rng.integers(0, len(l), len(l))]) if rng is not None else (lambda l: l)
    pos_s, neg_s = samp(ids["pos"]), samp(neg)
    sens = np.mean([lab[i] == "valid" for i in pos_s]); spec = np.mean([lab[i] != "valid" for i in neg_s])
    res = {"sens": float(sens), "spec": float(spec)}; raw = cor = 0.0
    for pool, w in POOLS.items():
        t = samp(ids[pool]); q = np.mean([lab[i] == "valid" for i in t]); p = rogan_gladen(q, sens, spec)
        res[pool] = {"n": len(t), "q": float(q), "p": p}; raw += w * q; cor += w * p
    res["share_raw"] = raw / N_FAIL; res["share_corrected"] = cor / N_FAIL
    return res

def kappa(a, b, cats):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")

def main(key_p, old_p, out_p, files):
    key = json.load(open(key_p)); old = {r["id"]: r for r in csv.DictReader(open(old_p, encoding="utf-8"))}
    loaded = [load(f) for f in files]; labs = [l for l, _ in loaded]; wds = [w for _, w in loaded]; rng = np.random.default_rng(20261011); res = {"annotators": []}
    for f, lab in zip(files, labs):
        e = estimate(lab, key); draws = [estimate(lab, key, rng) for _ in range(5000)]
        for k in ("share_raw", "share_corrected"):
            v = np.array([d[k] for d in draws]); e[k + "_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for k in ("sens", "spec"):
            n = sum(1 for i, v in key.items() if i in lab and v["kind"] == ("pos" if k == "sens" else None) or (k == "spec" and v["kind"].startswith("neg") and i in lab))
            kk = round(e[k] * n); e[k + "_wilson"] = list(map(float, wilson(kk, n)))
        # agreement with the earlier (label-visible) first-pass labels on the same test items
        t = [i for i, v in key.items() if v["kind"] in POOLS and v.get("old_case_id") in old and i in lab]
        ov = [old[key[i]["old_case_id"]]["first_pass"] == "valid" for i in t]; nv = [lab[i] == "valid" for i in t]
        e["vs_earlier_first_pass"] = {"n": len(t), "agree_valid_vs_not": float(np.mean([a == b for a, b in zip(ov, nv)])),
                                      "valid_share_blind": float(np.mean(nv)), "valid_share_earlier": float(np.mean(ov))}
        e["file"] = os.path.basename(f); res["annotators"].append(e)
    if len(labs) >= 2:
        a, b = labs[0], labs[1]; common = sorted(set(a) & set(b))
        for name, sel in (("all_items", common), ("test_items", [i for i in common if key[i]["kind"] in POOLS])):
            x, y = [a[i] for i in sel], [b[i] for i in sel]; po, k3 = kappa(x, y, ["valid", "wrong", "ambiguous"])
            xb, yb = [v == "valid" for v in x], [v == "valid" for v in y]
            res[f"agreement_{name}"] = {"n": len(sel), "raw": po, "kappa3": k3, "pabak_valid_vs_not": 2 * np.mean([p == q for p, q in zip(xb, yb)]) - 1}
        bounds = {}
        for pool, w in POOLS.items():
            ids = [i for i in common if key[i]["kind"] == pool]
            bounds[pool] = {"both_valid": float(np.mean([a[i] == "valid" and b[i] == "valid" for i in ids])), "either_valid": float(np.mean([a[i] == "valid" or b[i] == "valid" for i in ids]))}
        res["uncorrected_bounds"] = {k: sum(w * bounds[p][k] for p, w in POOLS.items()) / N_FAIL for k in ("both_valid", "either_valid")}
        words = [{"id": i, "model_label": key[i]["model_label"], "word_a1": wds[0][i], "word_a2": wds[1][i], "judged_a1": a[i], "judged_a2": b[i]} for i in common if key[i]["kind"] in POOLS]
        with open(out_p.replace(".json", "_naming.csv"), "w", encoding="utf-8", newline="") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(words[0])); w_.writeheader(); w_.writerows(words)
    json.dump(res, open(out_p, "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))

def selftest():
    rng = random.Random(1); tmp = tempfile.mkdtemp(); key = {}
    kinds = ["test_no_overlap"] * 65 + ["test_other_class"] * 35 + ["pos"] * 50 + ["neg_shift"] * 35 + ["neg_shrink"] * 15
    truth = {}
    for n, k in enumerate(kinds, 1):
        i = f"B{n:03d}"; key[i] = dict(kind=k, old_case_id=None, model_label="x")
        truth[i] = {"pos": 1, "neg_shift": 0, "neg_shrink": 0}.get(k, int(rng.random() < (0.70 if k == "test_no_overlap" else 0.85)))
    json.dump(key, open(f"{tmp}/key.json", "w")); open(f"{tmp}/old.csv", "w").write("id,set,first_pass\n")
    for a in (1, 2):
        with open(f"{tmp}/a{a}.csv", "w", encoding="utf-8") as fh:
            fh.write("id,choice,choice_name,word,note\n")
            for i, t in truth.items():
                said = (1 if rng.random() < 0.93 else 4) if t else (4 if rng.random() < 0.92 else 1)
                fh.write(f"{i},{said},,\"w\",\"\"\n")
    main(f"{tmp}/key.json", f"{tmp}/old.csv", f"{tmp}/out.json", [f"{tmp}/a1.csv", f"{tmp}/a2.csv"])
    true_share = (3940 * np.mean([truth[i] for i, v in key.items() if v["kind"] == "test_no_overlap"]) + 6468 * np.mean([truth[i] for i, v in key.items() if v["kind"] == "test_other_class"])) / N_FAIL
    print("true simulated share of all failures:", round(float(true_share), 3))

if __name__ == "__main__":
    selftest() if sys.argv[1] == "--selftest" else main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
