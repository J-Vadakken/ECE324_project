import cv2
import random
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT, logger

def generate_side_by_side_validation(num_samples=3):
    # 1. EXPLICIT PATHS
    model_path = PROJ_ROOT / "models" / "runs" / "synloc_50" / "weights" / "best.pt"
    img_dir = PROJ_ROOT / "data" / "processed" / "yolo-synloc-10k" / "images" / "val"
    lbl_dir = PROJ_ROOT / "data" / "processed" / "yolo-synloc-10k" / "labels" / "val"
    output_dir = PROJ_ROOT / "debug_preds" / "final_side_by_side"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}")
        return

    model = YOLO(model_path)
    img_files = list(img_dir.glob("*.jpg"))
    samples = random.sample(img_files, min(num_samples, len(img_files)))

    for img_path in samples:
        # Load Original
        img_orig = cv2.imread(str(img_path))
        h, w = img_orig.shape[:2]
        
        # Create two canvases
        canvas_gt = img_orig.copy()
        canvas_pred = img_orig.copy()

        # Initialize Counters
        gt_count = 0
        pred_count = 0

        # 2. DRAW GROUND TRUTH
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5: continue
                    
                    gt_count += 1 # Increment GT counter
                    cls, cx, cy, bw, bh = parts[:5]
                    
                    u1 = int((cx - bw/2) * w)
                    v1 = int((cy - bh/2) * h)
                    u2 = int((cx + bw/2) * w)
                    v2 = int((cy + bh/2) * h)
                    
                    cv2.rectangle(canvas_gt, (u1, v1), (u2, v2), (0, 0, 255), 3) # RED GT
                    cv2.putText(canvas_gt, f"GT:{int(cls)}", (u1, v1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. DRAW PREDICTIONS
        results = model.predict(source=str(img_path), conf=0.35, imgsz=960, verbose=False)[0]
        
        pred_count = len(results.boxes) # Increment Pred counter

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            cv2.rectangle(canvas_pred, (x1, y1), (x2, y2), (0, 255, 0), 2) # GREEN PRED
            cv2.putText(canvas_pred, f"ID:{cls}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. ASSEMBLE & STYLE
        combined = np.hstack((canvas_gt, canvas_pred))
        
        # Add Header Labels with Counts
        cv2.rectangle(combined, (0, 0), (w*2, 60), (0, 0, 0), -1)
        
        # Format the header text to include the counts
        header_text = f"FILE: {img_path.name} | GROUND TRUTH (Count: {gt_count}) | PREDICTION (Count: {pred_count})"
        cv2.putText(combined, header_text, (50, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Save
        save_path = output_dir / f"compare_{img_path.name}"
        cv2.imwrite(str(save_path), combined)
        logger.info(f"✅ Side-by-side saved: {save_path.name} (GT: {gt_count}, Pred: {pred_count})")

if __name__ == "__main__":
    generate_side_by_side_validation(5)