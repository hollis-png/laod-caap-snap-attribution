"""E3 step 1: regenerate Qwen2.5-VL-7B-Instruct scene labels for the 5,000 COCO-Val images (same prompt and generation settings as
run_qwen_llm_gen.py of the original run, batch size 1). The original 5,000-image label file was lost when NDHU was reinstalled."""
import argparse, json, os, time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
SYS = "Just Give the list of objects in given picture seperated by comma. Do not write anything else. Use singular name of the objects."
USR = "List the objects that you see in given picture."
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGES = os.path.expanduser("~/laod-exp/coco/val2017"); SRC = os.path.expanduser("~/laod-caap-check/coco_val_subset_predictions.jsonl")
OUT = os.path.expanduser("~/laod-exp/qwen_llm_labels_ndhu.jsonl")
def gen(model, processor, image):
    messages = [{"role": "system", "content": [{"type": "text", "text": SYS}]}, {"role": "user", "content": [{"type": "text", "text": USR}, {"type": "image", "image": image}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(messages)
    inputs = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=150, repetition_penalty=1.3, no_repeat_ngram_size=3)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].lower()
    return [l.strip() for l in out.replace(", ", ",").split(",") if l.strip()]
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
ids = [json.loads(l)["image_id"] for l in open(SRC) if l.strip()]
if a.limit: ids = ids[:a.limit]
done = set()
if os.path.exists(OUT): done = {json.loads(l)["image_id"] for l in open(OUT) if l.strip()}
todo = [i for i in ids if i not in done]; print(len(ids), "ids;", len(done), "done;", len(todo), "to do", flush=True)
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"); processor = AutoProcessor.from_pretrained(MODEL_ID)
print("loaded in", round(time.time() - t0), "s; transformers/torch:", __import__("transformers").__version__, torch.__version__, flush=True)
with open(OUT, "a") as f:
    for k, iid in enumerate(todo):
        t = time.time()
        try: labels = gen(model, processor, Image.open(f"{IMAGES}/{iid:012d}.jpg").convert("RGB"))
        except Exception as e: print("ERR", iid, str(e)[:120], flush=True); continue
        f.write(json.dumps({"image_id": iid, "llm_labels": labels, "elapsed_sec": time.time() - t}) + "\n"); f.flush()
        if k % 50 == 0: print(f"[{k+1}/{len(todo)}] {iid} {labels[:6]} ({time.time()-t:.1f}s)", flush=True)
print("DONE", flush=True)
