import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT, logger

def generate_eval_comparison(num_images=5):
    # 1. PATHS
    model_path = PROJ_ROOT / "models/runs/calibration_finetuned/weights/best.pt"
    # Use the SynLoc images specifically
    synloc_img_dir = PROJ_ROOT / "data/processed/yolo-synloc/images/train"
    output_dir = PROJ_ROOT / "models/runs/calibration/synloc_alignment_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    img_files = list(synloc_img_dir.glob("*.jpg"))[:num_images]

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] # This gets the ACTUAL size (e.g., 1080p)
        
        # 2. INFERENCE (Run at 960 to match training)
        results = model(img_path, imgsz=960, conf=0.1, verbose=False)[0]

        if results.keypoints is not None:
            # .xyn is the key: it returns coordinates from 0.0 to 1.0
            kpts_norm = results.keypoints.xyn[0].cpu().numpy()
            confs = results.keypoints.conf[0].cpu().numpy()

            for idx, (xn, yn) in enumerate(kpts_norm):
                # 3. CONVERT NORMALIZED -> ACTUAL PIXELS
                px = int(xn * w)
                py = int(yn * h)
                
                conf = confs[idx]
                if conf > 0.1: # Low threshold to see WHERE it's guessing
                    color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
                    cv2.circle(img, (px, py), 6, color, -1)
                    cv2.putText(img, f"{idx}({conf:.2f})", (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        save_path = output_dir / f"align_{img_path.name}"
        cv2.imwrite(str(save_path), img)
        logger.info(f"✅ Alignment check saved: {save_path.name}")

if __name__ == "__main__":
    generate_eval_comparison()