"""
Generate LLM object-name labels for the validated 5000-image COCO-Val subset
using Qwen2.5-VL-7B-Instruct, loaded ONCE and reused across the whole loop.

Writes incrementally + resumably to eval_out/qwen_llm_labels.jsonl
(schema: {"image_id": int, "llm_labels": [...], "elapsed_sec": float}),
skipping already-done image_ids on restart.

Usage:
    python eval/run_qwen_llm_gen.py [--limit N]
"""
import argparse
import json
import os
import sys
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
OUT_PATH = r"C:\LAOD\eval_out\qwen_llm_labels.jsonl"


def image_path(image_id):
    return f"{IMAGES_DIR}\\{image_id:012d}.jpg"


def generate_labels(model, processor, image):
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    llm_response = output_text[0].lower()
    return [l.strip() for l in llm_response.replace(", ", ",").split(",") if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N image_ids")
    args = parser.parse_args()

    image_ids = []
    with open(SRC_PREDICTIONS, "r") as f:
        for line in f:
            if line.strip():
                image_ids.append(json.loads(line)["image_id"])
    print(f"Total image_ids in source file: {len(image_ids)}", flush=True)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
        print(f"Limiting to first {len(image_ids)} image_ids", flush=True)

    done_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["image_id"])
        print(f"Resuming: {len(done_ids)} images already processed, skipping them.", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    print("Loading model...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model+processor load time: {time.time()-t0:.1f}s", flush=True)

    todo = [iid for iid in image_ids if iid not in done_ids]
    print(f"{len(todo)} images remaining to process", flush=True)

    with open(OUT_PATH, "a") as f:
        for i, image_id in enumerate(todo):
            t0 = time.time()
            try:
                img = Image.open(image_path(image_id)).convert("RGB")
                labels = generate_labels(model, processor, img)
            except Exception as e:
                print(f"[{i+1}/{len(todo)}] image_id={image_id} ERROR: {e}", flush=True)
                continue
            elapsed = time.time() - t0

            record = {"image_id": image_id, "llm_labels": labels, "elapsed_sec": elapsed}
            f.write(json.dumps(record) + "\n")
            f.flush()

            print(f"[{i+1}/{len(todo)}] image_id={image_id} labels={labels} ({elapsed:.2f}s)", flush=True)

    print("=== run_qwen_llm_gen complete ===", flush=True)


if __name__ == "__main__":
    main()

