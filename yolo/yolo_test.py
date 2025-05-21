from ultralytics import YOLO
import cv2
import os

# ----------- Config -----------
image_path = './sample.jpg'  # Change this to your image path
classes = ['person', 'book', 'chair', 'dog']  # Text prompts for detection
model_name = 'yolov8s-world.pt'  # or yolo_world_m, yolo_world_l, etc.
confidence_threshold = 0.25
show_image = False  # Set to False to skip OpenCV display
# ------------------------------

def main():
    # 1. Load the model
    print(f'Loading model: {model_name}')
    model = YOLO(model_name)

    print(model.names)

    # 2. Set the text classes
    print(f'Setting detection classes: {classes}')
    model.set_classes(classes)

    print(model.names)

    # 3. Run prediction
    print(f'Running prediction on {image_path}')
    results = model.predict(
        image_path,
        conf=confidence_threshold,
        save_txt=True  # Saves results in runs/detect/predict/labels
    )

    # 4. Print detections
    for result in results:
        print('--- Detected Boxes ---')
        print(result.boxes)

    # 5. Optional: Show image with detections
    if show_image:
        pred_dir = results[0].save_dir
        filename = os.path.basename(image_path)
        pred_path = os.path.join(pred_dir, filename)

        img = cv2.imread(str(pred_path))
        if img is not None:
            cv2.imshow('YOLO-World Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f'Could not load image from {pred_path}')


if __name__ == '__main__':
    main()
