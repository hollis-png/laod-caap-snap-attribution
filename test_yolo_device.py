from ultralytics import YOLO

m = YOLO("yolov8x-worldv2.pt")
print("device before set_classes:", next(m.model.parameters()).device)

m.set_classes(["cat"])
print("device after set_classes #1:", next(m.model.parameters()).device)

m.set_classes(["dog", "sofa"])
print("device after set_classes #2:", next(m.model.parameters()).device)
print("OK - no crash")

