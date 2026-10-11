"""E8: how much of the 72% 'valid but unannotated' estimate survives if stuff labels are set aside?

Inputs (all in the vault): author final labels of both audits (human_review/human_labels_final.csv + cases.json).
Stuff definition is fixed BEFORE looking at the result:
  core  = the six labels the earlier AI pass flagged as stuff in audit96_labels.json:
          sky, wall, field, beach, sea, baseball field
  broad = core + tree, trees, building, house   (COCO-Stuff lists these as stuff classes; sensitivity only)
Estimator is the one behind the 72% [67, 75] in section 3.4.1: stratum sizes 3,940 (no overlap) and 6,493 (other-class overlap),
valid share per stratum from the audit, Beta(k+1, n-k+1) posterior draws, percentile interval. The first step reproduces 72% [67, 75]
as a check; if it does not, the prior assumed here is wrong and the later rows should not be trusted.
Usage: python3 e8_stuff_exclusion.py <selfreview_dir> <out.json>
"""
import csv, json, sys
import numpy as np

base, out_path = sys.argv[1], sys.argv[2]
cases = {c["id"]: c for c in json.load(open(f"{base}/human_review/cases.json"))}
rows = list(csv.DictReader(open(f"{base}/human_review/human_labels_final.csv", encoding="utf-8")))
CORE = {"sky", "wall", "field", "beach", "sea", "baseball field"}
BROAD = CORE | {"tree", "trees", "building", "house"}
N_STRATUM = {"no_overlap_n200": 3940, "other_class_n96": 6468}
N_FAIL = 12139; N_DET = 27273; N_HIT = 15134  # detection-level rule (section 3.2)

def tally(stuff):
    t = {}
    for s in N_STRATUM:
        rs = [r for r in rows if r["set"] == s]
        valid = [r for r in rs if r["final"] == "valid"]
        st = [r for r in valid if cases[r["id"]]["label"].lower().strip() in stuff]
        t[s] = dict(n=len(rs), valid=len(valid), valid_stuff=len(st), valid_nonstuff=len(valid) - len(st),
                    ambiguous=sum(r["final"] == "ambiguous" for r in rs), wrong=sum(r["final"] == "wrong" for r in rs))
    return t

rng = np.random.default_rng(20261010)
D = 200000
def est(t, key):
    """share of all failures in category `key`, point + 95% Beta-posterior interval"""
    pt = sum(N_STRATUM[s] * t[s][key] / t[s]["n"] for s in N_STRATUM) / N_FAIL
    draws = sum(N_STRATUM[s] * rng.beta(t[s][key] + 1, t[s]["n"] - t[s][key] + 1, D) for s in N_STRATUM) / N_FAIL
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return pt, float(lo), float(hi)

res = {}
t0 = tally(set())
res["check_all_valid"] = dict(zip(("point", "lo", "hi"), est(t0, "valid")))  # must be ~0.724 [0.67, 0.75]
for name, stuff in (("core", CORE), ("broad", BROAD)):
    t = tally(stuff)
    r = {"audit_counts": t}
    for key in ("valid_stuff", "valid_nonstuff"):
        r[key] = dict(zip(("point", "lo", "hi"), est(t, key)))
    v_non = r["valid_nonstuff"]["point"]; v_st = r["valid_stuff"]["point"]
    # precision of detections (hit / all):
    #   stuff counted as failures: stuff-valid boxes stay in the denominator but are not credited
    #   stuff set aside: those detections leave numerator and denominator
    r["precision_stuff_as_failures"] = (N_HIT + v_non * N_FAIL) / N_DET
    r["precision_stuff_set_aside"] = (N_HIT + v_non * N_FAIL) / (N_DET - v_st * N_FAIL)
    res[name] = r
res["precision_all_valid_credited"] = (N_HIT + res["check_all_valid"]["point"] * N_FAIL) / N_DET
res["precision_baseline"] = N_HIT / N_DET
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
