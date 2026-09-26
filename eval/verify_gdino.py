import json

ids = set()
n = 0
bad = 0
with open("C:/LAOD/eval_out/gdino_predictions.jsonl") as f:
	for line in f:
		if not line.strip():
			continue
		n += 1
		try:
			r = json.loads(line)
			ids.add(r["image_id"])
		except Exception:
			bad += 1

print("total_lines", n, "unique_ids", len(ids), "bad_lines", bad)

