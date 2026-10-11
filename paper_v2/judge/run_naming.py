"""Naming agreement (preregistration item 6 of the label-free audit): for each test box an annotator called valid, do the annotator's free-text
word and the model's label name the same kind of object?

Text only: the judge sees two words (the model's label, English; the annotator's word, Chinese or English), never an image.
Prompt frozen 2026-10-11 before any output was seen.
Usage: python run_naming.py <model_id> <out.jsonl>     (run)
       python run_naming.py --analyze                   (reads out_naming_{deepseek,mistral}.jsonl, writes results_naming.json)
The OpenRouter key is read from the llm CLI's key store; it is never written to any file.
Pairs: results_blind_audit_naming.csv rows with judged_aN == "valid" and a non-empty word. A leading digit typed into the word field
by the form's key shortcuts (1 or 2 before the word) is removed.
"""
import csv, json, sys, re, time, datetime, threading
from concurrent.futures import ThreadPoolExecutor

CSV_P = "../blind_audit/results_blind_audit_naming.csv"
SYSTEM = ("You compare two words that name the content of a box in a photo. WORD_A is the label an AI assistant gave (English). WORD_B is what a person "
          "wrote after looking at the box (Chinese or English).\n"
          "Answer same if both words name the same kind of object: a synonym, a plural or singular form, or a broader or narrower term for the same "
          "thing (for example hat and cap, shelf and cabinet, bottles and bottle).\n"
          "Answer different if they name different things, including a part and the whole (handle and cup), an object and what is worn on it, "
          "or two unrelated objects.\n"
          "Judge the meaning of the words only.\n"
          'Reply with JSON only: {"answer": "same" | "different"}')
USER = "WORD_A: {a}\nWORD_B: {b}"
PARAMS = dict(temperature=0, max_tokens=60, extra_body={"reasoning": {"enabled": False}})


def pairs():
    out = []
    for r in csv.DictReader(open(CSV_P, encoding="utf-8")):
        for n in ("a1", "a2"):
            w = re.sub(r"^[12](?=\D)", "", r["word_" + n].strip())
            if r["judged_" + n] == "valid" and w:
                out.append(dict(id=r["id"], annotator=n, label=r["model_label"], word=w))
    return out


def parse(txt):
    t = (txt or "").strip()
    if t.startswith("```"): t = t.strip("`").replace("json", "", 1).strip()
    try: c = json.loads(t)["answer"]
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: c = json.loads(t[i:j + 1])["answer"]
        except Exception: return None
    return c if c in ("same", "different") else None


def run(model_id, out):
    import llm
    from openai import OpenAI
    api_model = model_id.replace("openrouter/", "", 1)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llm.get_key("", "openrouter"))
    items = pairs(); done = set()
    try:
        for l in open(out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("parsed") is not None: done.add((r["id"], r["annotator"]))
    except FileNotFoundError: pass
    todo = [x for x in items if (x["id"], x["annotator"]) not in done]
    json.dump(dict(model=api_model, system_prompt=SYSTEM, user_template=USER, params=PARAMS,
                   started=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
              open(out.replace(".jsonl", "_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{api_model}: {len(items)} pairs, {len(done)} done, {len(todo)} to do", flush=True)
    lock = threading.Lock()

    def work(x):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(a=x["label"], b=x["word"])}]
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=api_model, messages=msgs, **PARAMS)
                txt = r.choices[0].message.content or ""
                rec = dict(x, model_returned=r.model, raw=txt, parsed=parse(txt), attempts=attempt + 1,
                           ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
                if rec["parsed"] is None and attempt < 3:
                    time.sleep(1 + attempt); continue
                break
            except Exception as e:
                rec = dict(x, raw="", parsed=None, error=repr(e)[:200], attempts=attempt + 1)
                time.sleep(2 + attempt)
        with lock: open(out, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(4) as ex: list(ex.map(work, todo))


def analyze(out="results_naming.json"):
    res = {}
    for name in ("deepseek", "mistral"):
        try: rows = [json.loads(l) for l in open(f"out_naming_{name}.jsonl", encoding="utf-8") if l.strip()]
        except FileNotFoundError: continue
        last = {}
        for r in rows: last[(r["id"], r["annotator"])] = r
        last = {k: r for k, r in last.items() if not re.fullmatch(r"\d+", r["word"])}  # a word that is only a digit is a form keystroke, not a name (B166)
        e = {}
        for n in ("a1", "a2"):
            rs = [r for (i, a), r in last.items() if a == n]
            ok = [r for r in rs if r["parsed"]]
            e[n] = dict(pairs=len(rs), parsed=len(ok), same=sum(r["parsed"] == "same" for r in ok),
                        different=[(r["id"], r["label"], r["word"]) for r in ok if r["parsed"] == "different"])
        res[name] = e
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for name, e in res.items():
        for n, v in e.items(): print(name, n, f"{v['same']}/{v['parsed']} same ({v['pairs']} pairs)")


if __name__ == "__main__":
    if sys.argv[1] == "--analyze": analyze()
    else: run(sys.argv[1], sys.argv[2])
