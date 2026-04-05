import cv2
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from ECE324_Project.config import PROJ_ROOT, logger

def parse_yolo_pose_label(label_path, num_kpts=14):
    """
    Parses a YOLO pose .txt file.
    Returns a list of keypoints: [[x1, y1, v1], [x2, y2, v2], ...]
    """
    if not label_path.exists():
        return None
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return None

    # We assume the first detection is the pitch (class 0)
    line = lines[0].strip().split()
    # YOLO format: [class, cx, cy, w, h, k1_x, k1_y, k1_v, ...]
    # Keypoints start at index 5
    kpts_raw = list(map(float, line[5:]))
    
    # Reshape into [num_kpts, 3] -> (x, y, visibility)
    return np.array(kpts_raw).reshape(-1, 3)

def run_manual_calibration_eval(split="train"):
    # 1. SETUP PATHS
    model_path = PROJ_ROOT / "models/runs/calibration_synloc/weights/best.pt"
    img_dir = PROJ_ROOT / "data/processed/yolo-calibration/images" 
    lbl_dir = PROJ_ROOT / "data/processed/yolo-calibration/labels" 
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(model_path)
    
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    all_errors = []
    total_gt_visible = 0
    total_detected = 0

    logger.info(f"🚀 Evaluating on MANUALLY ANNOTATED {split} set ({len(img_files)} images)...")

    for img_path in tqdm(img_files):
        # Match image to label file
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_kpts = parse_yolo_pose_label(lbl_path)
        
        if gt_kpts is None:
            continue
            
        # Inference at 1920px
        results = model.predict(img_path, imgsz=1920, conf=0.1, verbose=False, device=device)[0]
        h, w = results.orig_shape
        
        # Check for detections
        if results.keypoints is not None and len(results.keypoints.xyn) > 0:
            pred_xyn = results.keypoints.xyn[0].cpu().numpy() # [14, 2] normalized
            pred_conf = results.keypoints.conf[0].cpu().numpy() # [14]

            for i in range(len(gt_kpts)):
                # GT is already normalized in YOLO .txt [0.0 - 1.0]
                xn_gt, yn_gt, vis = gt_kpts[i]
                
                if vis > 0: # Only evaluate if you annotated it as visible
                    total_gt_visible += 1
                    
                    if pred_conf[i] > 0.1:
                        total_detected += 1
                        
                        # Convert both to pixel space for MRE
                        u_p, v_p = pred_xyn[i][0] * w, pred_xyn[i][1] * h
                        u_gt, v_gt = xn_gt * w, yn_gt * h
                        
                        dist = np.sqrt((u_p - u_gt)**2 + (v_p - v_gt)**2)
                        all_errors.append(dist)
        else:
            # Count the missed visible points
            total_gt_visible += (gt_kpts[:, 2] > 0).sum()
            logger.warning(f"⚠️ No pitch detected in {img_path.name}")

    # --- METRICS ---
    err_arr = np.array(all_errors)
    mre = np.mean(err_arr) if len(err_arr) > 0 else 0
    rmse = np.sqrt(np.mean(err_arr**2)) if len(err_arr) > 0 else 0
    recall = total_detected / total_gt_visible if total_gt_visible > 0 else 0
    pck_10 = (err_arr < 10).sum() / total_gt_visible if total_gt_visible > 0 else 0

    print("\n" + "="*50)
    print(f"      CALIBRATION REPORT: MANUAL {split.upper()}")
    print("="*50)
    print(f"Mean Radial Error (MRE):  {mre:.3f} px")
    print(f"RMSE:                     {rmse:.3f} px")
    print(f"Keypoint Recall:          {recall:.2%}")
    print(f"PCK @ 10px:               {pck_10:.2%}")
    print("="*50)

if __name__ == "__main__":
    run_manual_calibration_eval("train")