"""Usage: python run_judge.py <model_id> <prompt_version v1|v2> <out.jsonl>
Classifies the 775 eligible labels (n_with_gt_match>0 and a matched COCO category). Resumable; temperature 0;
reasoning disabled (OpenRouter extra_body {"reasoning": {"enabled": false}}) because the reasoning mode sometimes
spent the whole token budget and returned an empty answer. Only the label word and the COCO category name are sent.
The OpenRouter key is read from the llm CLI's key store; it is never written to any file."""
import json, sys, time, datetime, threading
from concurrent.futures import ThreadPoolExecutor
import llm
from openai import OpenAI
from prompts import V1_SYSTEM, V2_SYSTEM, V2_MAP, USER
model_id, ver, out = sys.argv[1], sys.argv[2], sys.argv[3]
api_model = model_id.replace("openrouter/", "", 1)
system = {"v1": V1_SYSTEM, "v2": V2_SYSTEM}[ver]
PARAMS = dict(temperature=0, max_tokens=100, extra_body={"reasoning": {"enabled": False}})
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llm.get_key("", "openrouter"))
raw = json.load(open("raw_full.json", encoding="utf-8"))
items = [x for x in raw if x.get("most_common_matched_gt") and x.get("n_with_gt_match", 0) > 0]
done = set()
try:
    for l in open(out, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("parsed") is not None: done.add(r["label"])
except FileNotFoundError: pass
todo = [x for x in items if x["label"] not in done]
json.dump(dict(model=api_model, prompt_version=ver, system_prompt=system, user_template=USER, params=PARAMS,
               started=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
          open(out.replace(".jsonl", "_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{api_model} {ver}: {len(items)} items, {len(done)} done, {len(todo)} to do", flush=True)
lock = threading.Lock()
VALID = {"synonym", "finer_grained", "out_of_ontology", "unsure"}
def parse(txt):
    t = (txt or "").strip()
    if t.startswith("```"): t = t.strip("`").replace("json", "", 1).strip()
    try: r = json.loads(t)["relation"]
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: r = json.loads(t[i:j+1])["relation"]
        except Exception: return None
    if ver == "v2": r = V2_MAP.get(r, r)
    return r if r in VALID else None
def work(x):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": USER.format(label=x["label"], matched=x["most_common_matched_gt"])}]
    last = None
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=api_model, messages=msgs, **PARAMS)
            txt = r.choices[0].message.content or ""
            rec = dict(label=x["label"], matched=x["most_common_matched_gt"], model_requested=api_model, model_returned=r.model,
                       prompt_version=ver, raw=txt, parsed=parse(txt), finish=r.choices[0].finish_reason,
                       completion_tokens=r.usage.completion_tokens, attempts=attempt + 1,
                       ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
            if rec["parsed"] is None and attempt < 3:
                last = "unparsable/empty: " + repr(txt[:60]); time.sleep(1 + attempt); continue
            with lock: open(out, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
            return
        except Exception as e:
            last = str(e)[:200]
            if "402" in last: break          # out of credits: do not hammer
            time.sleep(2 * (attempt + 1))
    with lock: open(out, "a", encoding="utf-8").write(json.dumps(dict(label=x["label"], matched=x["most_common_matched_gt"], model_requested=api_model,
        prompt_version=ver, raw=None, parsed=None, error=last), ensure_ascii=False) + "\n")
with ThreadPoolExecutor(8) as ex: list(ex.map(work, todo))
print("finished", out, flush=True)
