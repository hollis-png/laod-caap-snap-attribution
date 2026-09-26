import json

with open(r"./datasets/coco/annotations/instances_val2017.json") as f:
    d = json.load(f)

print("images[0] keys:", list(d["images"][0].keys()))
print("images[0]:", d["images"][0])

