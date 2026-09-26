from ultralytics import YOLO
from PIL import Image

m = YOLO("yolov8x-worldv2.pt")
image = Image.open("images/1.jpg").convert("RGB")

print("--- round 1 ---")
print("model device before:", next(m.model.parameters()).device)
m.set_classes(["cat", "sofa"])
print("model device after set_classes:", next(m.model.parameters()).device)
results = m.predict(image, verbose=False)
print("model device after predict:", next(m.model.parameters()).device)

print("\n--- round 2 (this is where it crashed before) ---")
m.to("cpu")
m.set_classes(["dog", "remote"])
print("model device after set_classes #2:", next(m.model.parameters()).device)
results2 = m.predict(image, verbose=False)
print("OK - round 2 succeeded")

