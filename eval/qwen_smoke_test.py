"""
Smoke test: load Qwen2.5-VL-7B-Instruct and run one inference on a single image,
using the SAME prompt as the validated Gemma pipeline (laod.py / eval/laod_infer.py).
Reports timing and, if torch.cuda available, VRAM usage before/after load and after inference.
"""
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
IMAGE_PATH = r"C:\LAOD\images\1.jpg"

def vram_report(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")
    else:
        print(f"[VRAM {tag}] CUDA not available")

print("=== Qwen2.5-VL-7B-Instruct smoke test ===")
vram_report("before load")

t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
load_time = time.time() - t0
print(f"Model+processor load time: {load_time:.1f}s")
vram_report("after load")

image = Image.open(IMAGE_PATH).convert("RGB")

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
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

t1 = time.time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=500)
gen_time = time.time() - t1
vram_report("after generate")

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(f"Generation time: {gen_time:.1f}s")
print("RAW OUTPUT:", repr(output_text[0]))

llm_response = output_text[0].lower()
llm_labels = [l.strip() for l in llm_response.replace(", ", ",").split(",") if l.strip()]
print("PARSED LABELS:", llm_labels)

print("=== smoke test complete ===")

