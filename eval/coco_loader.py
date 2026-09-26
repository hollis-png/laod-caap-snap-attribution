import json
import os


class CocoGroundTruth:
	"""
	Minimal COCO instances-format loader producing the GT structures that
	eval.caap.compute_ca_ap and eval.snap.compute_snap expect, without
	depending on pycocotools.
	"""

	def __init__(self, annotation_path, images_dir):
		with open(annotation_path, "r") as f:
			coco = json.load(f)

		self.images_dir = images_dir
		self.category_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
		self.image_meta_by_id = {img["id"]: img for img in coco["images"]}

		self.anns_by_image_id = {}
		for ann in coco["annotations"]:
			self.anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

		self.image_ids = sorted(self.image_meta_by_id.keys())

	def __len__(self):
		return len(self.image_ids)

	def image_path(self, image_id):
		return os.path.join(self.images_dir, self.image_meta_by_id[image_id]["file_name"])

	def gt_boxes_caap(self, image_id):
		"""[x1, y1, x2, y2, class_category] per GT box, for eval.caap."""
		out = []
		for ann in self.anns_by_image_id.get(image_id, []):
			x, y, w, h = ann["bbox"]
			class_name = self.category_name_by_id[ann["category_id"]]
			out.append([x, y, x + w, y + h, class_name])
		return out

	def gt_labels_snap(self, image_id):
		"""[class_name] per GT object, for eval.snap (box geometry irrelevant to SNAP)."""
		out = []
		for ann in self.anns_by_image_id.get(image_id, []):
			out.append([self.category_name_by_id[ann["category_id"]]])
		return out

	def subset(self, n, seed=0):
		"""Returns a new CocoGroundTruth-like view restricted to n images (deterministic)."""
		import random

		rng = random.Random(seed)
		ids = self.image_ids[:]
		rng.shuffle(ids)
		selected = set(ids[:n])

		view = CocoGroundTruth.__new__(CocoGroundTruth)
		view.images_dir = self.images_dir
		view.category_name_by_id = self.category_name_by_id
		view.image_meta_by_id = {k: v for k, v in self.image_meta_by_id.items() if k in selected}
		view.anns_by_image_id = {k: v for k, v in self.anns_by_image_id.items() if k in selected}
		view.image_ids = sorted(selected)
		return view

