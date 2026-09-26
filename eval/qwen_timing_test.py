"""
Timing test: load Qwen2.5-VL-7B-Instruct once, run inference over the first N
images from the validated 5000-image image_id list (read from
coco_val_subset_predictions.jsonl), report per-image timing after model is
warm, to extrapolate full-5000 runtime.
"""
import json
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

_LLM_SYSTEM_PROMPT = (
    "Just Give the list of objects in given picture seperated by comma. "
    "Do not write anything else. Use singular name of the objects."
)
_LLM_USER_PROMPT = "List the objects that you see in given picture."

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGES_DIR = r"./datasets/coco/val2017"
SRC_PREDICTIONS = r"C:\LAOD\eval_out\coco_val_subset_predictions.jsonl"
N_TIMING = 30

def vram_report(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)

# get image_ids from source predictions file (same set used by validated run)
image_ids = []
with open(SRC_PREDICTIONS, "r") as f:
    for line in f:
        if line.strip():
            image_ids.append(json.loads(line)["image_id"])
print(f"Total image_ids in source file: {len(image_ids)}", flush=True)
timing_ids = image_ids[:N_TIMING]

print("Loading model...", flush=True)
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Load time: {time.time()-t0:.1f}s", flush=True)
vram_report("after load")

def image_path(image_id):
    return f"{IMAGES_DIR}\\{image_id:012d}.jpg"

def generate_labels(image):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _LLM_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _LLM_USER_PROMPT},
                {"type": "image", "image": image},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    llm_response = output_text[0].lower()
    return [l.strip() for l in llm_response.replace(", ", ",").split(",") if l.strip()]

times = []
for i, image_id in enumerate(timing_ids):
    img = Image.open(image_path(image_id)).convert("RGB")
    t0 = time.time()
    labels = generate_labels(img)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"[{i+1}/{len(timing_ids)}] image_id={image_id} labels={labels} ({elapsed:.2f}s)", flush=True)

vram_report("after timing batch")
avg = sum(times) / len(times)
# skip first (often includes extra warmup/cudnn autotune)
avg_warm = sum(times[1:]) / len(times[1:]) if len(times) > 1 else avg
print(f"\nAvg time/image (all {len(times)}): {avg:.2f}s", flush=True)
print(f"Avg time/image (excluding first): {avg_warm:.2f}s", flush=True)
print(f"Extrapolated for 5000 images: {avg_warm*5000/3600:.2f} hours", flush=True)
print("=== timing test complete ===", flush=True)

