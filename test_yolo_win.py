import time
import laod
from PIL import Image

t0 = time.time()
image = Image.open('images/1.jpg')
print(f"Image loaded: {image.size}, mode={image.mode}")

result = laod.laod_yolo(image)
t1 = time.time()
print(f"SUCCESS - elapsed {t1-t0:.1f}s")
print(f"Result type: {type(result)}, shape: {getattr(result, 'shape', 'N/A')}")

import cv2
cv2.imwrite('test_output.jpg', result)
print("Saved to test_output.jpg")

