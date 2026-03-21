import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT, logger

def generate_eval_comparison(num_images=5):
    model_path = PROJ_ROOT / "models/runs/calibration/weights/best.pt"
    valid_img_dir = PROJ_ROOT / "data" / "processed" / "yolo-calibration-2023" / "images" / "valid"
    output_dir = PROJ_ROOT / "models" / "runs" / "calibration" / "eval_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        logger.error(f"❌ Model weights not found at {model_path}")
        return

    img_files = list(valid_img_dir.glob("*.jpg"))[:num_images]
    model = YOLO(model_path)
    all_errors = []

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        img_out = img.copy()
        img_gt = img.copy()

        # 1. PREDICTIONS + ID LABELS
        results = model(img_path, conf=0.3, verbose=False)[0]
        pred_kpts = []
        if results.keypoints is not None:
            # kpts structure: [num_keypoints, 3] -> (x, y, conf)
            kpts = results.keypoints.data[0].cpu().numpy()
            for idx, (x, y, conf) in enumerate(kpts):
                margin = 5
                if conf > 0.4 and (margin < x < w-margin) and (margin < y < h-margin):
                    px, py = int(x), int(y)
                    pred_kpts.append((px, py))
                    # Smaller dot (radius 6)
                    cv2.circle(img_out, (px, py), 6, (0, 255, 0), -1)
                    # Label the ID next to the dot
                    cv2.putText(img_out, str(idx), (px + 8, py - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. LABELS (GT) + ID LABELS
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        gt_kpts = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.split()))
                    # Pose labels: class, x, y, w, h, k1_x, k1_y, k1_v, k2_x, k2_y, k2_v...
                    # In YOLO Pose, visibility is usually the 3rd element in the triplet
                    kpt_idx = 0
                    for j in range(5, len(parts), 3):
                        gx, gy = int(parts[j] * w), int(parts[j+1] * h)
                        if gx > 0 and gy > 0:
                            gt_kpts.append((gx, gy))
                            cv2.circle(img_gt, (gx, gy), 6, (0, 0, 255), -1)
                            cv2.putText(img_gt, str(kpt_idx), (gx + 8, gy - 8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        kpt_idx += 1

        # 3. HELPER FOR STYLED TITLES
        def draw_styled_title(canvas, text, color):
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 1.0
            thick = 2
            (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(canvas, (10, 10), (10 + t_w + 20, 10 + t_h + 20), (255, 255, 255), -1)
            cv2.putText(canvas, text, (20, 15 + t_h), font, scale, color, thick)

        draw_styled_title(img_gt, "GT (Manual Labels)", (0, 0, 255))
        draw_styled_title(img_out, "PRED (YOLOv8 Pose)", (0, 180, 0))

        # 4. ASSEMBLE & SAVE
        combined = np.hstack((img_gt, img_out))
        save_path = output_dir / f"id_eval_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), combined)
        logger.info(f"✅ Comparison saved with IDs: {save_path.name}")

    if all_errors:
        logger.info(f"📊 Median Error: {np.median(all_errors):.2f} px")

if __name__ == "__main__":
    generate_eval_comparison()