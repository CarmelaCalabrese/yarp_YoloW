import torch
import cv2
from yoloworld.models import build_model
from yoloworld.util.inference import inference

# --- Configuration ---
image_path = "sample.jpg"  # Replace with your test image path
device = "cuda" if torch.cuda.is_available() else "cpu"
text_prompts = ["a person", "a bottle", "a book", "a chair"]

# --- Load image ---
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Load YOLO-World model ---
print("Loading YOLO-World model...")
model = build_model("yoloworld_l", pretrained=True)
model = model.to(device).eval()

# --- Inference ---
print("Running inference...")
output = inference(model, img_rgb, text_prompts, device=device)

# --- Visualization ---
for det in output['predictions']:
    box = det['box']
    label = det['label']
    conf = det['score']
    cv2.rectangle(img_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
    cv2.putText(img_bgr, f"{label} {conf:.2f}", (int(box[0]), int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("YOLO-World Test", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
