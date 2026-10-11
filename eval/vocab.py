"""Vocabulary categories of LLM labels relative to COCO-80 (torch-free, so scripts and tests can import it).

native             exact match to a COCO-80 name
plural_of_native   a singular candidate of the label is a COCO-80 name (grammatical form only)
out_of_vocabulary  anything else

History: the first release singularised with suffix rules ("ses" -> strip two letters, "ies" -> "y") and compared the
result with the singularised COCO names. That turned "horses" into "hors", "buses" into "bus" while "bus" itself became
"bu", "vases" into "vas", "suitcases" into "suitcas" and "ties" into "ty", so those labels were counted as
out-of-vocabulary (see the correction notice in the README). The rule now tests every plausible singular form against the
COCO-80 names.
"""

COCO80_NAMES = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
	"couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
	"toothbrush",
}

# Irregular plural/singular pairs among COCO-80 names and common LLM phrasing for them.
_IRREGULAR_SINGULARS = {
	"people": "person",
	"mice": "mouse",
	"knives": "knife",
	"loaves": "loaf",
	"shelves": "shelf",
	"leaves": "leaf",
}


def singular_candidates(word):
	"""All plausible singular forms of the (lower-cased) label, plus the plural form when that is a COCO-80 name
	("skis", "scissors"); the label itself is not included."""
	w = word.lower().strip()
	if w in _IRREGULAR_SINGULARS:
		return [_IRREGULAR_SINGULARS[w]]
	out = []
	if w + "s" in COCO80_NAMES:
		out.append(w + "s")                     # "ski" -> "skis", "scissor" -> "scissors" (COCO names that are plural)
	if w.endswith("ies") and len(w) > 3:
		out += [w[:-3] + "y", w[:-1]]          # "flies" -> "fly"; "ties" -> "tie"
	if w.endswith("es") and len(w) > 2:
		out.append(w[:-2])                      # "buses" -> "bus"; "benches" -> "bench"
	if w.endswith("s") and not w.endswith("ss") and len(w) > 1:
		out.append(w[:-1])                      # "horses" -> "horse"; "chairs" -> "chair"
	return out


def normalize_plural(word):
	"""Singular form used for matching: a COCO-80 name if any singular candidate is one, otherwise the most
	plausible candidate (or the word itself)."""
	w = word.lower().strip()
	if w in COCO80_NAMES:
		return w
	cands = singular_candidates(w)
	for c in cands:
		if c in COCO80_NAMES:
			return c
	return cands[0] if cands else w


COCO80_NAMES_SINGULAR = set(COCO80_NAMES)


def vocabulary_category(label):
	"""native / plural_of_native / out_of_vocabulary, see the module docstring."""
	raw = label.lower().strip()
	if raw in COCO80_NAMES:
		return "native"
	if any(c in COCO80_NAMES for c in singular_candidates(raw)):
		return "plural_of_native"
	return "out_of_vocabulary"
