import cv2
import random
import numpy as np
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT, logger

def verify_labels(num_samples=3):
    # 1. Point specifically to your PROCESSED data (The stuff you just fixed)
    img_dir = PROJ_ROOT / "data" / "processed" / "yolo-synloc-10k" / "images" / "val"
    lbl_dir = PROJ_ROOT / "data" / "processed" / "yolo-synloc-10k" / "labels" / "val"
    output_dir = PROJ_ROOT / "debug_labels" / "final_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(img_dir.glob("*.jpg"))
    if not img_files:
        logger.error(f"❌ No images found in {img_dir}. Check your paths!")
        return

    # Select 3 random images to check coverage across the pitch
    samples = random.sample(img_files, min(num_samples, len(img_files)))

    for img_path in samples:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        if not lbl_path.exists():
            logger.warning(f"⚠️ Missing label for: {img_path.name}")
            continue

        with open(lbl_path, 'r') as f:
            for line in f:
                # YOLO: cls, cx, cy, bw, bh
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5: continue
                
                cls, cx, cy, bw, bh = parts[:5]

                # DYNAMIC SCALING: This uses the ACTUAL image width/height 
                # to turn the 0.0-1.0 percentages back into pixels.
                u1 = int((cx - bw/2) * w)
                v1 = int((cy - bh/2) * h)
                u2 = int((cx + bw/2) * w)
                v2 = int((cy + bh/2) * h)

                # Use a thick, bright color (Cyan) to verify
                cv2.rectangle(img, (u1, v1), (u2, v2), (255, 255, 0), 2)
                cv2.putText(img, f"ID:{int(cls)}", (u1, v1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        save_path = output_dir / f"verify_{img_path.name}"
        cv2.imwrite(str(save_path), img)
        logger.info(f"✅ Verification image saved: {save_path.name}")

if __name__ == "__main__":
    verify_labels(3)