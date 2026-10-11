"""Usage: python compute_agreement.py human_labels.csv
Compares the human labels exported from index.html with the AI answer key.
n=96 (other_class_n96): per-case agreement + Cohen's kappa against the AI's per-case labels.
n=200 (no_overlap_n200): the AI's per-case labels were not saved, so only the human distribution is
compared against the AI aggregate (gt_gap 167 / ambiguous 18 / real_failure 15)."""
import csv, json, sys, collections
key = {k["id"]: k for k in json.load(open("answer_key_DO_NOT_OPEN_BEFORE_REVIEW.json", encoding="utf-8"))}
hum = {r["id"]: r["human"] for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["human"]}
cats = ["valid", "ambiguous", "wrong"]
ai_map = {"valid_unannotated": "valid", "ambiguous": "ambiguous", "wrong": "wrong"}
def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*((p*(1-p)/n + z*z/(4*n*n))**.5) / (1 + z*z/n); return round(c-h, 3), round(c+h, 3)
pairs = [(ai_map[key[i]["ai"]], h) for i, h in hum.items() if key[i]["set"] == "other_class_n96"]
if pairs:
    n = len(pairs); agree = sum(a == h for a, h in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(h == c for _, h in pairs) / n) for c in cats)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"n96: labelled {n}/96, agreement {agree:.3f}, Cohen kappa {kappa:.3f}")
    print("  confusion (AI rows x human cols):", {a: dict(collections.Counter(h for x, h in pairs if x == a)) for a in cats})
    v = sum(h == "valid" for _, h in pairs); print(f"  human valid share {v/n:.3f}, Wilson 95% CI {wilson(v, n)} (AI: 79/96 = 0.823)")
h200 = [h for i, h in hum.items() if key[i]["set"] == "no_overlap_n200"]
if h200:
    n = len(h200); c = collections.Counter(h200); v = c["valid"]
    print(f"n200: labelled {n}/200, human valid {v/n:.3f} (Wilson {wilson(v, n)}), ambiguous {c['ambiguous']/n:.3f}, wrong {c['wrong']/n:.3f}")
    print("  AI aggregate: gt_gap 0.835 [0.777, 0.880], ambiguous 0.090, real_failure 0.075")
