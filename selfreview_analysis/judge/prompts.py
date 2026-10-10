"""Judge prompts (versions frozen 2026-10-09, before any judge output was seen).
V1 mirrors the definitions and examples shown to the human labeller (human_review_b1/index.html).
V2 paraphrases the definitions, reorders and renames the options, and uses different examples."""
V1_SYSTEM = """You classify the relation between two words.
LABEL is a word an AI assistant used to name an object in a photo. CATEGORY is the COCO object category whose annotated boxes the LABEL's box most often overlapped.
Choose exactly one relation of LABEL to CATEGORY:
- synonym: LABEL refers to the same kind of thing as CATEGORY, only worded differently; they are largely interchangeable. Examples: sofa/couch, phone/cell phone, television/tv, horses/horse, vehicle/car.
- finer_grained: LABEL still refers to the same object or the same group of objects as CATEGORY, but is more specific or covers a different extent. Examples: woman/person, boy/person, surfer/person, table/dining table, flowers/vase.
- out_of_ontology: LABEL refers to a different object, a part of the object, or an accessory, not CATEGORY itself. Examples: helmet/person, hat/person, shoes/person, plate/dining table, cabinet/oven, shelf/book.
- unsure: the words are unclear or you cannot decide.
Judge the meaning of the words only. For gender, age or occupation words relative to person, use your own best judgment. For clothing words such as suit or jacket relative to person, judge by their literal meaning.
Reply with JSON only: {"relation": "synonym" | "finer_grained" | "out_of_ontology" | "unsure"}"""
V2_SYSTEM = """Task: decide how a naming word relates to a COCO category.
You are given LABEL (the word a vision assistant used for an object in a photo) and CATEGORY (the COCO class that the object's box usually overlapped).
Pick exactly one option:
- different_thing: LABEL names another object, a component of one, or something worn or attached to it, rather than CATEGORY itself (for example wheel/car, handle/cup, glove/person, lid/pot).
- same_thing_other_word: LABEL denotes the same kind of object as CATEGORY under a different word, so the two can be swapped (for example bike/bicycle, tv set/tv, cellphone/cell phone).
- narrower_description: LABEL is still the same object or group as CATEGORY but describes it more specifically or over a different extent (for example puppy/dog, armchair/chair, passengers/person).
- cannot_tell: the words are unclear or you cannot decide.
Consider the meanings of the words only, not any image. For words about a person's gender, age or job, and for clothing words relative to person, use your own best judgment from the literal meaning.
Answer with JSON only: {"relation": "different_thing" | "same_thing_other_word" | "narrower_description" | "cannot_tell"}"""
V2_MAP = {"different_thing": "out_of_ontology", "same_thing_other_word": "synonym", "narrower_description": "finer_grained", "cannot_tell": "unsure"}
USER = "LABEL: {label}\nCATEGORY: {matched}"
# labels whose words appear in the V1 prompt (examples or guidance) -> excluded from the headline agreement
V1_EXAMPLE_WORDS = {"sofa", "phone", "television", "horses", "vehicle", "woman", "boy", "surfer", "table", "flowers", "helmet", "hat", "shoes", "plate", "cabinet", "shelf", "suit", "jacket"}
