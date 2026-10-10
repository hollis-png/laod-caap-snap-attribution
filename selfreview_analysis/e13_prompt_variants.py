"""E13: does the vocabulary effect depend on the exact prompt?  Runs the Qwen2.5-VL-7B + YOLO-World X pipeline on the 5,000 COCO-Val images with
prompt variants and evaluates each with e3_eval.py (same corrected CAAP and the section-3.6 vocabulary table).

Variants (the base prompt of the main Qwen run is SYS/USR in e3_qwen_labels.py and is NOT rerun here; its numbers are in e3_eval_qwen.json):
  paraphrase : same task, different wording and a different example of how to answer
  common     : same task, but restricted to common everyday object names (tests the idea of steering the LLM towards familiar vocabulary)
Variant scripts are made by rewriting the SYS / USR / path lines of e3_qwen_labels.py and e3_detect_gpu.py, so generation settings, weights and
detector are untouched. Run on NDHU:  cd ~/laod-exp && nohup .venv/bin/python -u e13_prompt_variants.py > e13.log 2>&1 &
"""
import re, subprocess, sys, os
H = os.path.expanduser("~/laod-exp")
VARIANTS = {
    "paraphrase": ("You are an object recognizer. Reply with only a comma-separated list of the distinct objects present in the image, using one singular noun for each object.",
                   "Which objects are in this image?"),
    "common": ("Just Give the list of objects in given picture seperated by comma. Do not write anything else. Use singular name of the objects. Only use common everyday object names that most people would use.",
               "List the objects that you see in given picture."),
}
env = dict(os.environ, TORCH_DISABLE_NATIVE_JIT="1")
for name, (sysm, usr) in VARIANTS.items():
    lab = f"{H}/qwen_llm_labels_{name}.jsonl"; pred = f"{H}/qwen_yoloworld_predictions_{name}.jsonl"; ev = f"{H}/e13_eval_{name}.json"
    src = open(f"{H}/e3_qwen_labels.py").read()
    src = re.sub(r'^SYS = .*$', lambda m: 'SYS = ' + repr(sysm), src, flags=re.M); src = re.sub(r'^USR = .*$', lambda m: 'USR = ' + repr(usr), src, flags=re.M)
    src = re.sub(r'^OUT = .*$', lambda m: 'OUT = ' + repr(lab), src, flags=re.M)
    assert repr(sysm) in src and repr(lab) in src
    open(f"{H}/_e13_labels_{name}.py", "w").write(src)
    det = open(f"{H}/e3_detect_gpu.py").read()
    det = re.sub(r'LAB = os\.path\.expanduser\("[^"]*"\)', lambda m: 'LAB = ' + repr(lab), det)
    assert repr(lab) in det
    open(f"{H}/_e13_detect_{name}.py", "w").write(det)
    for cmd in ([sys.executable, "-u", f"{H}/_e13_labels_{name}.py"], [sys.executable, "-u", f"{H}/_e13_detect_{name}.py", pred], [sys.executable, "-u", f"{H}/e3_eval.py", pred, ev]):
        print("RUN", name, os.path.basename(cmd[2]), flush=True); subprocess.run(cmd, check=True, env=env, cwd=H)
print("ALL DONE", flush=True)
