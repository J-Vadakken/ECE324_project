import cv2
import random
from pathlib import Path
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT

def test_on_calibration(num_samples=10):
    # 1. PATHS
    model_path = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"
    # Testing specifically on your manually annotated calibration IMAGES
    image_dir = PROJ_ROOT / "data/SoccerNet/calibration-2023/test"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return

    # 2. LOAD MODEL
    model = YOLO(str(model_path))

    # 3. GET IMAGES
    all_images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if not all_images:
        print(f"❌ No images found in {image_dir}. Did you run the sync script?")
        return
        
    samples = random.sample(all_images, min(num_samples, len(all_images)))

    print("🚀 Running Object Detection on Calibration Frames...")
    print("👉 Press 'q' to quit | Any other key for next image")

    for img_path in samples:
        # Run inference
        results = model.predict(source=str(img_path), conf=0.3, imgsz=960)[0]

        # --- PLOT SETTINGS ---
        # labels=False: Removes the "player 0.85" text
        # conf=False: Removes the confidence percentage
        # boxes=True: Keeps the actual bounding boxes
        annotated_frame = results.plot(labels=False, conf=False)

        # Resize for display on your M2 Max
        h, w = annotated_frame.shape[:2]
        display_frame = cv2.resize(annotated_frame, (w // 2, h // 2))

        # UI Overlay
        count = len(results.boxes)
        color = (0, 0, 255) if count > 0 else (0, 255, 0)

        cv2.imshow("Detection Consistency Check", display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_on_calibration()