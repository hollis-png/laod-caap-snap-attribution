"""Text-only judge (E12): classify a LABEL against COCO's 80 categories WITHOUT the box-overlap reference category.

Why: in the section 3.3 contrast the reference class of a label is the COCO category its boxes most often overlapped, and the outcome
(CAAP-TP) also depends on box overlap, so the contrast is partly circular. Here the classification sees only the label word and the list
of 80 COCO categories; box overlap never enters the classification. The outcome rate per class is then computed from the stored
per-label CAAP-TP rates as before.

Usage: python run_textonly.py <model_id> <out.jsonl>        (prompt frozen 2026-10-10 before any output was seen)
       python run_textonly.py --analyze                      (reads out_textonly_{deepseek,mistral}.jsonl, writes results_textonly.json)
Eligible labels: the same 775 as run_judge.py (n_with_gt_match > 0 and a matched COCO category), for comparability.
The OpenRouter key is read from the llm CLI's key store; it is never written to any file.
"""
import json, sys, time, datetime, threading, collections
from concurrent.futures import ThreadPoolExecutor

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
          "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
          "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
          "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]
assert len(COCO80) == 80
SYSTEM = ("You map a word to COCO's 80 object categories.\n"
          "LABEL is a word an AI assistant used to name an object in a photo. The 80 COCO categories are: " + ", ".join(COCO80) + ".\n"
          "Decide whether LABEL names an object of one of these categories, whatever the wording (synonym, plural, or a more specific description of the "
          "same kind of object, such as woman or surfer for person, armchair for chair). If it does, give that category. If LABEL instead names a part "
          "of an object, something worn or attached to an object, a material, a scene or background region (sky, wall, field), or an object that "
          "is not one of the 80 categories, answer none.\n"
          "Judge the meaning of the word only.\n"
          'Reply with JSON only: {"category": "<one of the 80 categories exactly as written>" | "none"}')
USER = "LABEL: {label}"
PARAMS = dict(temperature=0, max_tokens=100, extra_body={"reasoning": {"enabled": False}})

def parse(txt):
    t = (txt or "").strip()
    if t.startswith("```"): t = t.strip("`").replace("json", "", 1).strip()
    try: c = json.loads(t)["category"]
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: c = json.loads(t[i:j + 1])["category"]
        except Exception: return None
    return c if (c == "none" or c in COCO80) else None

def eligible():
    raw = json.load(open("raw_full.json", encoding="utf-8"))
    return [x for x in raw if x.get("most_common_matched_gt") and x.get("n_with_gt_match", 0) > 0]

def run(model_id, out):
    import llm
    from openai import OpenAI
    api_model = model_id.replace("openrouter/", "", 1)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llm.get_key("", "openrouter"))
    items = eligible(); done = set()
    try:
        for l in open(out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("parsed") is not None: done.add(r["label"])
    except FileNotFoundError: pass
    todo = [x for x in items if x["label"] not in done]
    json.dump(dict(model=api_model, system_prompt=SYSTEM, user_template=USER, params=PARAMS,
                   started=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
              open(out.replace(".jsonl", "_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{api_model}: {len(items)} items, {len(done)} done, {len(todo)} to do", flush=True)
    lock = threading.Lock()
    def write(rec):
        with lock: open(out, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
    def work(x):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(label=x["label"])}]
        last = None
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=api_model, messages=msgs, **PARAMS)
                txt = r.choices[0].message.content or ""
                rec = dict(label=x["label"], model_requested=api_model, model_returned=r.model, raw=txt, parsed=parse(txt),
                           finish=r.choices[0].finish_reason, attempts=attempt + 1,
                           ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
                if rec["parsed"] is None and attempt < 3:
                    last = "unparsable/empty: " + repr(txt[:60]); time.sleep(1 + attempt); continue
                write(rec); return
            except Exception as e:
                last = str(e)[:200]
                if "402" in last: break          # out of credits: do not hammer
                time.sleep(2 * (attempt + 1))
        write(dict(label=x["label"], model_requested=api_model, raw=None, parsed=None, error=last))
    with ThreadPoolExecutor(8) as ex: list(ex.map(work, todo))
    print("finished", out, flush=True)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5
    return (c - h) / d, (c + h) / d

def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

def analyze():
    def load(p):
        d = {}
        for l in open(p, encoding="utf-8"):
            if l.strip(): r = json.loads(l); d[r["label"]] = r
        return d
    runs = {"deepseek": load("out_textonly_deepseek.jsonl"), "mistral": load("out_textonly_mistral.jsonl")}
    raw = {x["label"]: x for x in eligible()}
    old = load("out_deepseek_v1.jsonl")
    two_old = lambda r: None if r in (None, "unsure") else ("different" if r == "out_of_ontology" else "same")
    cls = lambda r: None if r.get("parsed") is None else ("different" if r["parsed"] == "none" else "same")
    res = {}
    for k, d in runs.items():
        a = {"same": [0, 0, 0.0], "different": [0, 0, 0.0]}; fail = 0
        for l, x in raw.items():
            t = cls(d.get(l, {}))
            if not t: fail += 1; continue
            n = x["n_with_gt_match"]; a[t][0] += 1; a[t][1] += n; a[t][2] += x["caap_tp_rate"] * n
        s, df = a["same"], a["different"]
        rs, rd = s[2] / s[1], df[2] / df[1]
        pr = [(cls(d[l]), two_old(old[l]["parsed"])) for l in raw if l in d and l in old and cls(d[l]) and two_old(old[l]["parsed"])]
        res[k] = dict(unparsed_labels=fail, same_labels=s[0], same_n=s[1], same_rate=rs, same_ci=wilson(round(s[2]), s[1]),
                      diff_labels=df[0], diff_n=df[1], diff_rate=rd, diff_ci=wilson(round(df[2]), df[1]), gap_pp=100 * (rs - rd),
                      vs_overlap_reference_judge=dict(n=len(pr), agree=sum(x == y for x, y in pr) / len(pr), kappa=kappa(pr),
                                                      crosstab={f"{x}|{y}": c for (x, y), c in collections.Counter(pr).items()}))
    names = list(runs)
    pr = [(cls(runs[names[0]][l]), cls(runs[names[1]][l])) for l in raw if cls(runs[names[0]].get(l, {})) and cls(runs[names[1]].get(l, {}))]
    res["judge_vs_judge"] = dict(n=len(pr), agree=sum(x == y for x, y in pr) / len(pr), kappa=kappa(pr))
    json.dump(res, open("results_textonly.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))

if __name__ == "__main__":
    if "--analyze" in sys.argv: analyze()
    else: run(sys.argv[1], sys.argv[2])
