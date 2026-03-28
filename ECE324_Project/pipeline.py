import cv2
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from ultralytics import YOLO
import os
import json
from scipy.spatial.distance import cdist
from ECE324_Project.config import PROJ_ROOT, logger

# Standard FIFA Pitch. Origin (0,0) is CENTER FIELD.
PITCH_MODEL_METERS = {
    4: [52.5, -34.0], 5: [52.5, 34.0], 0: [36.0, -20.16], 1: [36.0, 20.16],
    2: [47.0, -9.16], 3: [47.0, 9.16], 6: [0.0, -34.0], 7: [0.0, 34.0],
    12: [-52.5, -34.0], 13: [-52.5, 34.0], 8: [-36.0, -20.16], 
    9: [-36.0, 20.16], 10: [-47.0, -9.16], 11: [-47.0, 9.16]
}

def draw_blank_radar(scale=10):
    w, h = int(105 * scale), int(68 * scale)
    radar = np.zeros((h, w, 3), dtype=np.uint8)
    radar[:] = (34, 139, 34) 
    white = (255, 255, 255)
    line_thick = 2
    
    # 1. Boundary & Halfway
    cv2.rectangle(radar, (0, 0), (w, h), white, line_thick)
    cv2.line(radar, (w//2, 0), (w//2, h), white, line_thick)
    cv2.circle(radar, (w//2, h//2), int(9.15 * scale), white, line_thick)

    # 2. Left Side Boxes
    cv2.rectangle(radar, (0, int((34-20.15)*scale)), (int(16.5*scale), int((34+20.15)*scale)), white, line_thick)
    cv2.rectangle(radar, (0, int((34-9.15)*scale)), (int(5.5*scale), int((34+9.15)*scale)), white, line_thick)

    # 3. Right Side Boxes
    cv2.rectangle(radar, (w - int(16.5*scale), int((34-20.15)*scale)), (w, int((34+20.15)*scale)), white, line_thick)
    cv2.rectangle(radar, (w - int(5.5*scale), int((34-9.15)*scale)), (w, int((34+9.15)*scale)), white, line_thick)
    
    return radar

class ECE324Pipeline:
    def __init__(self, pitch_model_path, player_model_path, json_path):
        logger.info("Initializing Robust Pipeline (No Refinement)...")
        self.pitch_detector = YOLO(str(pitch_model_path))
        self.player_detector = YOLO(str(player_model_path))
        self.ui_colors = [(0,0,255), (255,0,0), (0,255,255), (255,255,255), (0,0,0), (255,0,255)]
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.img_meta = {img['file_name']: img for img in data['images']}
            
            # Index Ground Truth by image_id
            self.gt_data = {}
            # --- THE SOCCERNET-TO-FIFA TRANSFORM ---
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.gt_data: 
                    self.gt_data[img_id] = []
                
                # 1. Grab raw values from JSON
                json_x, json_y = ann['position_on_pitch'][:2]
                
                # 2. Map them to your X (Length) and Y (Width) model
                # Based on the data:
                # Our X_model = JSON -Y
                # Our Y_model = JSON -X
                gx = -json_y
                gy = -json_x
                
                self.gt_data[img_id].append([gx, gy])

    def undistort_pts(self, pts, K, D):
        pts_arr = np.array(pts, dtype='float32').reshape(-1, 1, 2)
        return cv2.undistortPoints(pts_arr, K, D, P=K).reshape(-1, 2)

    def get_jersey_feature(self, crop):
        if crop.size == 0: return None
        h, w = crop.shape[:2]
        torso = crop[int(h*0.15):int(h*0.55), :]
        if torso.size == 0: return None
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_not(cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255])))
        significant = torso[mask > 0]
        return np.median(significant, axis=0) if len(significant) > 15 else None

    def run_popups(self, img_dir, limit=10):
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])[:limit]

        for img_name in img_files:
            img = cv2.imread(str(Path(img_dir) / img_name))
            if img is None: continue
            h_img, w_img = img.shape[:2]
            meta = self.img_meta.get(img_name)
            if not meta: continue
            
            K = np.array(meta['camera_matrix'], dtype='float32')[:, :3]
            D = np.array(meta['dist_poly'][:5], dtype='float32')

            # --- 1. PITCH CALIBRATION ---
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
                # No refinement: Use raw YOLO points but undistort them first
                flat_src = self.undistort_pts(raw_pts, K, D)
                H, _ = cv2.findHomography(flat_src, np.array(dst_pts, dtype='float32'), cv2.RANSAC, 5.0)

            # --- 2. GROUND TRUTH PLOTTING ---
            radar = draw_blank_radar()
            gt_points = self.gt_data.get(meta['id'], [])
            for gx, gy in gt_points:
                rx, ry = int((gx + 52.5) * 10), int((gy + 34.0) * 10)
                if 0 <= rx < 1050 and 0 <= ry < 680:
                    cv2.drawMarker(radar, (rx, ry), (255, 255, 255), cv2.MARKER_STAR, 12, 1)

            # --- 3. PLAYER DETECTION & PROJECTION ---
            player_res = self.player_detector.predict(img, imgsz=960, conf=0.5, verbose=False)[0]
            player_data, features = [], []
            for box in player_res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                feat = self.get_jersey_feature(img[y1:y2, x1:x2])
                player_data.append({'feet': (int((x1+x2)/2), y2), 'feat': feat, 'box': (x1,y1,x2,y2)})
                if feat is not None: features.append(feat)

            labels = []
            if len(features) >= 3:
                kmeans = KMeans(n_clusters=min(5, len(features)), n_init=10).fit(np.array(features))
                labels = kmeans.labels_

            pred_meters = []
            if H is not None and len(player_data) > 0:
                flat_feet = self.undistort_pts([p['feet'] for p in player_data], K, D)
                proj = cv2.perspectiveTransform(flat_feet.reshape(-1, 1, 2), H)
                
                f_idx = 0
                for i, pt in enumerate(proj):
                    mx, my = pt[0][0], pt[0][1]
                    pred_meters.append([mx, my])
                    
                    p = player_data[i]
                    color = (128, 128, 128)
                    if p['feat'] is not None and len(labels) > 0:
                        color = self.ui_colors[labels[f_idx] % len(self.ui_colors)]
                        f_idx += 1
                    
                    cv2.rectangle(img, p['box'][:2], p['box'][2:], color, 2)
                    rx, ry = int((mx + 52.5) * 10), int((my + 34.0) * 10)
                    if 0 <= rx < 1050 and 0 <= ry < 680:
                        cv2.circle(radar, (rx, ry), 8, color, -1)
                        cv2.circle(radar, (rx, ry), 8, (0, 0, 0), 1)

            # --- 4. ERROR METRIC ---
            if len(pred_meters) > 0 and len(gt_points) > 0:
                distances = cdist(pred_meters, gt_points)
                avg_err = np.mean(np.min(distances, axis=1))
                cv2.putText(radar, f"Avg Error: {avg_err:.2f}m", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Main Feed", cv2.resize(img, (960, 540)))
            cv2.imshow("Radar (Stars=GT, Dots=Pred)", radar)
            if cv2.waitKey(0) == ord('q'): break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    P_M = PROJ_ROOT / "models/runs/synloc_pixel_refinement_1920/weights/best.pt"
    A_M = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"
    JSON_P = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/annotations/test.json"
    DATA_D = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/test/"
    
    pipeline = ECE324Pipeline(P_M, A_M, JSON_P)
    pipeline.run_popups(DATA_D)