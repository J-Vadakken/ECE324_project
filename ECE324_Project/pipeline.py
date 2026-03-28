import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from ECE324_Project.config import PROJ_ROOT, logger

# --- DIMENSIONS & ANCHORS (103x68) ---
P_LENGTH, P_WIDTH = 103.0, 68.0
HALF_L, HALF_W = P_LENGTH / 2, P_WIDTH / 2
MARGIN, SCALE = 5, 10   

PENALTY_W, PENALTY_D = 40.32, 16.5
GOALIE_W, GOALIE_D = 18.32, 5.5

PITCH_MODEL_METERS = {
    4: [HALF_L, -HALF_W], 5: [HALF_L, HALF_W], 
    0: [HALF_L - 16.5, -20.16], 1: [HALF_L - 16.5, 20.16],
    2: [HALF_L - 5.5, -9.16],  3: [HALF_L - 5.5, 9.16],
    6: [0.0, -HALF_W], 7: [0.0, HALF_W],
    12: [-HALF_L, -HALF_W], 13: [-HALF_L, HALF_W], 
    8: [-(HALF_L - 16.5), -20.16], 9: [-(HALF_L - 16.5), 20.16],
    10: [-(HALF_L - 5.5), -9.16], 11: [-(HALF_L - 5.5), 9.16]
}

def draw_blank_radar():
    w_px, h_px = int((P_LENGTH + 2 * MARGIN) * SCALE), int((P_WIDTH + 2 * MARGIN) * SCALE)
    radar = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    track_gray, pitch_green, white = (50, 50, 50), (34, 139, 34), (255, 255, 255)
    radar[:] = track_gray 
    
    off = int(MARGIN * SCALE)
    p_w, p_h = int(P_LENGTH * SCALE), int(P_WIDTH * SCALE)
    
    # 1. Pitch Surface & Main Lines
    cv2.rectangle(radar, (off, off), (off + p_w, off + p_h), pitch_green, -1)
    cv2.rectangle(radar, (off, off), (off + p_w, off + p_h), white, 2)
    cv2.line(radar, (off + p_w // 2, off), (off + p_w // 2, off + p_h), white, 2)
    cv2.circle(radar, (off + p_w // 2, off + p_h // 2), int(9.15 * SCALE), white, 2)
    
    # 2. Penalty & Goalie Box Geometry
    # Y-offsets centered on the width
    p_y1 = int((HALF_W - (PENALTY_W/2) + MARGIN) * SCALE)
    p_y2 = int((HALF_W + (PENALTY_W/2) + MARGIN) * SCALE)
    g_y1 = int((HALF_W - (GOALIE_W/2) + MARGIN) * SCALE)
    g_y2 = int((HALF_W + (GOALIE_W/2) + MARGIN) * SCALE)

    # Left Side
    cv2.rectangle(radar, (off, p_y1), (off + int(PENALTY_D*SCALE), p_y2), white, 2)
    cv2.rectangle(radar, (off, g_y1), (off + int(GOALIE_D*SCALE), g_y2), white, 2)
    # Right Side
    cv2.rectangle(radar, (off + p_w - int(PENALTY_D*SCALE), p_y1), (off + p_w, p_y2), white, 2)
    cv2.rectangle(radar, (off + p_w - int(GOALIE_D*SCALE), g_y1), (off + p_w, g_y2), white, 2)
    
    return radar

class ECE324Pipeline:
    def __init__(self, pitch_model_path, player_model_path, json_path):
        logger.info("Initializing Header-Visualizer Pipeline...")
        self.pitch_detector = YOLO(str(pitch_model_path))
        self.player_detector = YOLO(str(player_model_path))
        self.ui_colors = [(0,0,255), (255,0,0), (0,255,255), (255,255,255), (0,0,0), (255,0,255)]
        self.out_dir = PROJ_ROOT / "results/visuals"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.img_meta = {img['file_name']: img for img in data['images']}
            self.gt_data = {ann['image_id']: [] for ann in data['annotations']}
            for ann in data['annotations']:
                jx, jy = ann['position_on_pitch'][:2]
                self.gt_data[ann['image_id']].append([-jy, -jx])

    def calculate_locsim(self, d, tau=1.0):
        if d > 4.0: return 0.0
        return np.exp(np.log(0.05) * (d**2 / tau**2))

    def undistort_pts(self, pts, K, D):
        pts_arr = np.array(pts, dtype='float32').reshape(-1, 1, 2)
        return cv2.undistortPoints(pts_arr, K, D, P=K).reshape(-1, 2)

    def run_visualizer(self, img_dir, limit=15):
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])[:limit]

        for img_name in img_files:
            img = cv2.imread(str(Path(img_dir) / img_name))
            meta = self.img_meta.get(img_name)
            if img is None or not meta: continue
            h_img, w_img = img.shape[:2]
            
            K = np.array(meta['camera_matrix'], dtype='float32')[:, :3]
            D = np.array(meta['dist_poly'][:5], dtype='float32')

            # 1. Calibration
            p_res = self.pitch_detector.predict(img, imgsz=1920, conf=0.1, verbose=False)[0]
            raw_pts, dst_pts = [], []
            if p_res.keypoints is not None:
                kpts = p_res.keypoints.xyn[0].cpu().numpy()
                confs = p_res.keypoints.conf[0].cpu().numpy()
                for i, (xn, yn) in enumerate(kpts):
                    if confs[i] > 0.15 and i in PITCH_MODEL_METERS:
                        raw_pts.append([xn * w_img, yn * h_img])
                        dst_pts.append(PITCH_MODEL_METERS[i])

            H = None
            if len(raw_pts) >= 4:
                flat_src = self.undistort_pts(raw_pts, K, D)
                H, _ = cv2.findHomography(flat_src, np.array(dst_pts, dtype='float32'), cv2.RANSAC, 5.0)

            # 2. Player Detection
            player_res = self.player_detector.predict(img, imgsz=960, conf=0.5, verbose=False)[0]
            gt_points = self.gt_data.get(meta['id'], [])
            
            player_data, features = [], []
            for box in player_res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                torso = img[int(y1 + (y2-y1)*0.15):int(y1 + (y2-y1)*0.55), x1:x2]
                feat = None
                if torso.size > 0:
                    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                    mask = cv2.bitwise_not(cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255])))
                    sig = torso[mask > 0]
                    if len(sig) > 15: feat = np.median(sig, axis=0)
                
                player_data.append({'feet': (int((x1+x2)/2), y2), 'feat': feat, 'box': (x1,y1,x2,y2)})
                if feat is not None: features.append(feat)

            labels = []
            if len(features) >= 2:
                labels = KMeans(n_clusters=2, n_init=10).fit(np.array(features)).labels_

            # 3. Projection & Radar
            radar = draw_blank_radar()
            for gx, gy in gt_points:
                rx, ry = int((gx + HALF_L + MARGIN) * SCALE), int((gy + HALF_W + MARGIN) * SCALE)
                cv2.drawMarker(radar, (rx, ry), (255, 255, 255), cv2.MARKER_STAR, 12, 1)

            pred_meters = []
            if H is not None and len(player_data) > 0:
                flat_feet = self.undistort_pts([p['feet'] for p in player_data], K, D)
                proj = cv2.perspectiveTransform(flat_feet.reshape(-1, 1, 2), H).reshape(-1, 2)
                f_idx = 0
                for i, pt in enumerate(proj):
                    mx, my = pt[0], pt[1]
                    pred_meters.append([mx, my])
                    p = player_data[i]
                    color = self.ui_colors[labels[f_idx] % 6] if (p['feat'] is not None and len(labels) > 0) else (128,128,128)
                    if p['feat'] is not None and len(labels) > 0: f_idx += 1
                    cv2.rectangle(img, p['box'][:2], p['box'][2:], color, 4)
                    rx, ry = int((mx + HALF_L + MARGIN) * SCALE), int((my + HALF_W + MARGIN) * SCALE)
                    if 0 <= rx < radar.shape[1] and 0 <= ry < radar.shape[0]:
                        cv2.circle(radar, (rx, ry), 8, color, -1)
                        cv2.circle(radar, (rx, ry), 8, (0, 0, 0), 1)

            # --- CONSTRUCT THE PRO HEADER ---
            view_h = 720
            img_res = cv2.resize(img, (int(img.shape[1] * (view_h / img.shape[0])), view_h))
            radar_res = cv2.resize(radar, (int(radar.shape[1] * (view_h / radar.shape[0])), view_h))
            combined = np.hstack((img_res, radar_res))
            
            # Create a white header bar
            header_h = 100
            header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8) + 255
            
            # Title Text
            title = f"Frame: {img_name} | Detections: {len(player_res.boxes)} | GT: {len(gt_points)}"
            cv2.putText(header, title, (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Metrics on the right side of the header
            if len(pred_meters) > 0 and len(gt_points) > 0:
                dists = cdist(pred_meters, gt_points)
                p_idx, g_idx = linear_sum_assignment(dists)
                matched_d = dists[p_idx, g_idx]
                metrics = f"mEuclidean: {np.mean(matched_d):.2f}m | mLocSim: {np.mean([self.calculate_locsim(d) for d in matched_d]):.3f}"
                cv2.putText(header, metrics, (combined.shape[1] - 800, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

            # Stack Header and Content
            final_frame = np.vstack((header, combined))
            
            cv2.imwrite(str(self.out_dir / f"header_{img_name}"), final_frame)
            cv2.imshow("ECE324 Pro Visualizer", final_frame)
            if cv2.waitKey(0) == ord('q'): break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    P_M = PROJ_ROOT / "models/runs/synloc_pixel_refinement_1920/weights/best.pt"
    A_M = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"
    JSON_P = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/annotations/train.json"
    DATA_D = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/train/"
    
    pipeline = ECE324Pipeline(P_M, A_M, JSON_P)
    pipeline.run_visualizer(DATA_D, limit=15)