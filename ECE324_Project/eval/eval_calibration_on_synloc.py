import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT, logger

def generate_eval_comparison(num_images=5):
    # 1. PATHS
    model_path = PROJ_ROOT / "models/runs/synloc_pixel_refinement_1920/weights/best.pt"

    # Use the SynLoc images specifically
    synloc_img_dir = PROJ_ROOT / "data/processed/yolo-calibration/images"
    output_dir = PROJ_ROOT / "debug_preds" / "synloc_alignment_check"
    output_dir.mkdir(parents=True, exist_ok=True)


    model = YOLO(model_path)
    img_files = list(synloc_img_dir.glob("*.jpg"))[:num_images]

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] # This gets the ACTUAL size (e.g., 1080p)
        
        # 2. INFERENCE (Run at 960 to match training)
        results = model(img_path, imgsz=1920, conf=0.1, verbose=False)[0]

        if len(results.boxes) > 0 and results.keypoints is not None:
            # .xyn returns [num_objs, 14, 2] in 0.0-1.0 range
            kpts_norm = results.keypoints.xyn[0].cpu().numpy()
            confs = results.keypoints.conf[0].cpu().numpy()

            for idx, (xn, yn) in enumerate(kpts_norm):
                # 3. CONVERT NORMALIZED -> ACTUAL PIXELS
                # YOLO xyn is [x, y], so xn*w and yn*h is correct
                px = int(xn * w)
                py = int(yn * h)

                if (confs[idx] < 0.1):
                    continue
                
                # Filter out the (0,0) padding points if any
                if xn == 0 and yn == 0: continue

                conf = confs[idx]
                # Draw green for high confidence, red for low
                color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
                cv2.circle(img, (px, py), 8, color, -1)
                cv2.putText(img, f"ID:{idx} ({conf:.2f})", (px + 12, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            logger.warning(f"⚠️ No pitch detected in {img_path.name}")

        cv2.imwrite(str(output_dir / f"align_{img_path.name}"), img)
        logger.info(f"✅ Saved: align_{img_path.name}")

if __name__ == "__main__":
    generate_eval_comparison()