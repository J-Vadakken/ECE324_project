import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment 
from ECE324_Project.config import PROJ_ROOT, logger

# --- 1. THE MISSING CONSTANTS (Pitch Geometry) ---
P_LENGTH, P_WIDTH = 103.0, 68.0
HALF_L, HALF_W = P_LENGTH / 2, P_WIDTH / 2

# Dictionary mapping YOLO Keypoint IDs to World-Space Meters
PITCH_MODEL_METERS = {
    4: [HALF_L, -HALF_W], 5: [HALF_L, HALF_W],             # Near Goal Line Corners
    0: [HALF_L - 16.5, -20.16], 1: [HALF_L - 16.5, 20.16], # Penalty Box line
    2: [HALF_L - 5.5, -9.16],  3: [HALF_L - 5.5, 9.16],   # Goalie Box line
    6: [0.0, -HALF_W], 7: [0.0, HALF_W],                   # Halfway Line Corners
    12: [-HALF_L, -HALF_W], 13: [-HALF_L, HALF_W],         # Far Goal Line Corners
    8: [-(HALF_L - 16.5), -20.16], 9: [-(HALF_L - 16.5), 20.16],
    10: [-(HALF_L - 5.5), -9.16], 11: [-(HALF_L - 5.5), 9.16]
}

def calculate_locsim(d, tau=1.0):
    if d > 4.0: return 0.0 
    return np.exp(np.log(0.05) * (d**2 / tau**2))

# --- 2. THE EVALUATOR CLASS ---
class ECE324Evaluator:
    def __init__(self, pitch_model_path, player_model_path):
        self.pitch_yolo = YOLO(pitch_model_path)
        self.player_yolo = YOLO(player_model_path)

    def undistort_pts(self, pts, K, D):
        pts_arr = np.array(pts, dtype='float32').reshape(-1, 1, 2)
        return cv2.undistortPoints(pts_arr, K, D, P=K).reshape(-1, 2)

    def evaluate_set(self, set_name, json_path, img_dir, limit=None):
        logger.info(f"--- Running Global Evaluation: {set_name} Set ---")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_meta = {img['file_name']: img for img in data['images']}
        gt_lookup = {ann['image_id']: [] for ann in data['annotations']}
        for ann in data['annotations']:
            jx, jy = ann['position_on_pitch'][:2]
            gt_lookup[ann['image_id']].append([-jy, -jx]) 

        # Accumulators
        dist_errors, locsim_scores = [], []
        skipped_frames, total_frames = 0, 0
        
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        if limit: img_files = img_files[:limit]

        for i, img_name in enumerate(img_files):
            meta = img_meta.get(img_name)
            if not meta: continue
            total_frames += 1
            
            img = cv2.imread(str(Path(img_dir) / img_name))
            if img is None: continue
            h, w = img.shape[:2]
            
            # Camera Props
            K = np.array(meta['camera_matrix'], dtype='float32')[:, :3]
            D = np.array(meta['dist_poly'][:5], dtype='float32')
            
            # 1. Pitch Detection
            p_res = self.pitch_yolo.predict(img, imgsz=1920, conf=0.1, verbose=False)[0]
            src_pts, dst_pts = [], []
            if p_res.keypoints is not None and len(p_res.keypoints.xyn) > 0:
                kpts = p_res.keypoints.xyn[0].cpu().numpy()
                confs = p_res.keypoints.conf[0].cpu().numpy()
                for idx, (xn, yn) in enumerate(kpts):
                    if confs[idx] > 0.15 and idx in PITCH_MODEL_METERS:
                        src_pts.append([xn * w, yn * h])
                        dst_pts.append(PITCH_MODEL_METERS[idx])

            if len(src_pts) < 4:
                skipped_frames += 1
                continue

            # 2. Homography Math
            flat_src = self.undistort_pts(src_pts, K, D)
            H, _ = cv2.findHomography(flat_src, np.array(dst_pts, dtype='float32'), cv2.RANSAC, 5.0)

            # 3. Player Detections
            player_res = self.player_yolo.predict(img, imgsz=960, conf=0.5, verbose=False)[0]
            if len(player_res.boxes) == 0: continue
            
            feet = [[int((b.xyxy[0][0]+b.xyxy[0][2])/2), int(b.xyxy[0][3])] for b in player_res.boxes]
            flat_feet = self.undistort_pts(feet, K, D)
            proj = cv2.perspectiveTransform(flat_feet.reshape(-1, 1, 2), H).reshape(-1, 2)

            # 4. Global Optimal Matching (Hungarian)
            gt_pts = gt_lookup.get(meta['id'], [])
            if len(gt_pts) > 0 and len(proj) > 0:
                distances = cdist(proj, gt_pts)
                pred_idx, gt_idx = linear_sum_assignment(distances)
                best_match_dists = distances[pred_idx, gt_idx]
                
                dist_errors.extend(best_match_dists)
                locsim_scores.extend([calculate_locsim(d) for d in best_match_dists])

        # --- 5. REPORT GENERATION ---
        err_arr = np.array(dist_errors)
        report = {
            "Mean Dist": np.mean(err_arr) if len(err_arr) > 0 else 0,
            "Median Dist": np.median(err_arr) if len(err_arr) > 0 else 0,
            "RMSE": np.sqrt(np.mean(err_arr**2)) if len(err_arr) > 0 else 0,
            "E90 (90th Pctl)": np.percentile(err_arr, 90) if len(err_arr) > 0 else 0,
            "PCK@2m": (err_arr < 2.0).sum() / len(err_arr) if len(err_arr) > 0 else 0,
            "PCK@5m": (err_arr < 5.0).sum() / len(err_arr) if len(err_arr) > 0 else 0,
            "LocSim": np.mean(locsim_scores) if locsim_scores else 0,
            "Calib Fail Rate": (skipped_frames / total_frames) * 100 if total_frames > 0 else 0
        }
        self.print_summary(set_name, report)

    def print_summary(self, name, r):
        print("\n" + "="*50)
        print(f"      ECE324 FINAL SYSTEM REPORT: {name}")
        print("="*50)
        for key, val in r.items():
            if "PCK" in key or "Rate" in key:
                print(f"{key:<22} | {val:>15.2%}")
            else:
                print(f"{key:<22} | {val:>15.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    P_MODEL = PROJ_ROOT / "models/runs/synloc_pixel_refinement_1920/weights/best.pt"
    A_MODEL = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"
    
    evaluator = ECE324Evaluator(P_MODEL, A_MODEL)

    # TRAIN Set Eval
    t_json = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/annotations/train.json"
    t_dir = PROJ_ROOT / "data/processed/yolo-calibration/images"
    evaluator.evaluate_set("TRAIN", t_json, t_dir, limit=150)

    # TEST Set Eval
    v_json = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/annotations/test.json"
    v_dir = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/test/"
    evaluator.evaluate_set("TEST", v_json, v_dir, limit=300)