import yarp
import cv2
import numpy as np
import torch
from yoloworld.models import build_model
from yoloworld.util.inference import inference

# --- YARP Setup ---
yarp.Network.init()
input_port = yarp.Port()
input_port.open("/yarp/image:i")
img_data = yarp.ImageRgb()

# --- YOLO-World Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model("yoloworld_l", pretrained=True)
model = model.to(device).eval()

# Define the prompts
text_prompts = ["a person", "a bottle", "a book", "a chair"]

# --- Main loop ---
print("Waiting for images...")
while True:
    if input_port.read(img_data):
        h, w = img_data.height(), img_data.width()
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        img_data.convertToCvPixelCode()
        img_data.copyimg(img_array)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Inference
        output = inference(model, img_rgb, text_prompts, device=device)

        # Draw bounding boxes
        for det in output['predictions']:
            box = det['box']
            label = det['label']
            conf = det['score']
            cv2.rectangle(img_array, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(img_array, f"{label} {conf:.2f}", (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Show result
        cv2.imshow("YOLO-World Detection", img_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
input_port.close()
yarp.Network.fini()

