import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ECE324_Project.config import PROJ_ROOT, logger

# --- 1. PITCH GEOMETRY CONSTANTS ---
P_LENGTH, P_WIDTH = 103.0, 68.0
HALF_L, HALF_W = P_LENGTH / 2, P_WIDTH / 2

# Mapping YOLO Keypoint IDs to World-Space Meters
PITCH_MODEL_METERS = {
    4: [HALF_L, -HALF_W], 5: [HALF_L, HALF_W],
    0: [HALF_L - 16.5, -20.16], 1: [HALF_L - 16.5, 20.16],
    2: [HALF_L - 5.5, -9.16],  3: [HALF_L - 5.5, 9.16],
    6: [0.0, -HALF_W], 7: [0.0, HALF_W],
    12: [-HALF_L, -HALF_W], 13: [-HALF_L, HALF_W],
    8: [-(HALF_L - 16.5), -20.16], 9: [-(HALF_L - 16.5), 20.16],
    10: [-(HALF_L - 5.5), -9.16], 11: [-(HALF_L - 5.5), 9.16]
}

# --- LocSim settings ---
LOCSIM_TAU = 1.0
LOCSIM_TP_THRESH = 0.5

def calculate_locsim(d, tau=LOCSIM_TAU):
    """
    LocSim(d) = 0.05^(d^2 / tau^2)
    With tau = 1m.
    """
    return 0.05 ** ((d ** 2) / (tau ** 2))


class ECE324Evaluator:
    def __init__(self, pitch_model_path, player_model_path):
        self.pitch_yolo = YOLO(pitch_model_path)
        self.player_yolo = YOLO(player_model_path)

    def undistort_pts(self, pts, K, D):
        pts_arr = np.array(pts, dtype="float32").reshape(-1, 1, 2)
        return cv2.undistortPoints(pts_arr, K, D, P=K).reshape(-1, 2)

    def evaluate_set(self, set_name, json_path, img_dir, limit=None):
        logger.info(f"--- Running Global Evaluation: {set_name} Set ---")

        with open(json_path, "r") as f:
            data = json.load(f)

        img_meta = {img["file_name"]: img for img in data["images"]}
        gt_lookup = {ann["image_id"]: [] for ann in data["annotations"]}
        for ann in data["annotations"]:
            jx, jy = ann["position_on_pitch"][:2]
            gt_lookup[ann["image_id"]].append([-jy, -jx])

        # Accumulators
        dist_errors = []
        matched_locsim_scores = []
        skipped_frames, total_frames = 0, 0
        tp_total, fp_total, fn_total = 0, 0, 0
        correct_frames = 0

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
        if limit:
            img_files = img_files[:limit]

        for img_name in img_files:
            meta = img_meta.get(img_name)
            if not meta:
                continue

            total_frames += 1

            img = cv2.imread(str(Path(img_dir) / img_name))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Camera intrinsics
            K = np.array(meta["camera_matrix"], dtype="float32")[:, :3]
            D = np.array(meta["dist_poly"][:5], dtype="float32")

            # 1. Pitch detection
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

            # 2. Homography computation
            flat_src = self.undistort_pts(src_pts, K, D)
            H, _ = cv2.findHomography(
                flat_src,
                np.array(dst_pts, dtype="float32"),
                cv2.RANSAC,
                5.0
            )

            if H is None:
                skipped_frames += 1
                continue

            # 3. Player detections
            player_res = self.player_yolo.predict(img, imgsz=960, conf=0.5, verbose=False)[0]
            feet = [[int((b.xyxy[0][0] + b.xyxy[0][2]) / 2), int(b.xyxy[0][3])] for b in player_res.boxes]

            gt_pts = gt_lookup.get(meta["id"], [])

            # No predictions
            if not feet:
                fn_total += len(gt_pts)
                continue

            flat_feet = self.undistort_pts(feet, K, D)
            proj = cv2.perspectiveTransform(flat_feet.reshape(-1, 1, 2), H).reshape(-1, 2)

            # No GT
            if len(gt_pts) == 0:
                fp_total += len(proj)
                continue

            # 4. Matching predicted to GT
            distances = cdist(proj, gt_pts)
            pred_idx, gt_idx = linear_sum_assignment(distances)
            matched_dists = distances[pred_idx, gt_idx]
            matched_locsims = np.array([calculate_locsim(d) for d in matched_dists])

            # TP threshold is LocSim(d) > 0.5
            tp_mask = matched_locsims > LOCSIM_TP_THRESH
            tps = int(np.sum(tp_mask))
            fps = len(proj) - tps
            fns = len(gt_pts) - tps

            tp_total += tps
            fp_total += fps
            fn_total += fns

            dist_errors.extend(matched_dists.tolist())
            matched_locsim_scores.extend(matched_locsims.tolist())

            # Frame accuracy uses the same TP rule
            if fns == 0 and fps == 0:
                correct_frames += 1

        # --- 5. REPORT GENERATION ---
        err_arr = np.array(dist_errors, dtype=float)
        locsim_arr = np.array(matched_locsim_scores, dtype=float)

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # mAP-LocSim:
        # using the requested correctness condition LocSim(d) >= 0.5
        # this is effectively the fraction of matched assignments satisfying that threshold
        mAP_locsim = np.mean(locsim_arr >= LOCSIM_TP_THRESH) if len(locsim_arr) > 0 else 0.0

        report = {
            "mAP-LocSim": mAP_locsim,
            "Mean LocSim": np.mean(locsim_arr) if len(locsim_arr) > 0 else 0.0,
            "FrameAcc": correct_frames / total_frames if total_frames > 0 else 0.0,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Mean Dist (m)": np.mean(err_arr) if len(err_arr) > 0 else 0.0,
            "Median Dist (m)": np.median(err_arr) if len(err_arr) > 0 else 0.0,
            "PCK@2m": (err_arr < 2.0).sum() / len(err_arr) if len(err_arr) > 0 else 0.0,
            "Calib Fail Rate": (skipped_frames / total_frames) * 100 if total_frames > 0 else 0.0
        }

        self.print_summary(set_name, report)

    def print_summary(self, name, r):
        print("\n" + "=" * 60)
        print(f"      ECE324 SYSTEM PERFORMANCE REPORT: {name}")
        print("=" * 60)
        for key, val in r.items():
            if any(x in key for x in ["PCK", "Rate", "Acc", "Precision", "Recall", "F1", "mAP"]):
                print(f"{key:<25} | {val:>15.2%}")
            else:
                print(f"{key:<25} | {val:>15.4f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    P_MODEL = PROJ_ROOT / "models/runs/calibration_synloc/weights/best.pt"
    A_MODEL = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"

    evaluator = ECE324Evaluator(P_MODEL, A_MODEL)

    t_json = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/annotations/train.json"
    t_dir = PROJ_ROOT / "data/processed/yolo-calibration/images"
    evaluator.evaluate_set("TRAIN", t_json, t_dir, limit=150)

    v_json = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/annotations/test.json"
    v_dir = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/test/"
    evaluator.evaluate_set("TEST", v_json, v_dir, limit=300)
