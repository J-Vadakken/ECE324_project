import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT

def generate_pro_storyboard(image_path, model_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not load image at {image_path}")
        return

    model = YOLO(model_path)
    # Detect players
    results = model.predict(img, imgsz=960, conf=0.5, verbose=False)[0]
    
    # Storage for the flow
    player_data = []
    max_players = 10 # Keep width manageable for a PDF/Report
    
    # Standard dimensions for each "cell" in the grid
    C_H, C_W = 220, 140

    # 1. Feature Extraction Loop
    for box in results.boxes[:max_players]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_crop = cv2.resize(img[y1:y2, x1:x2], (C_W, C_H))
        
        # --- Torso Logic (Middle 40% of height) ---
        t_y1, t_y2 = int(C_H * 0.15), int(C_H * 0.55)
        torso = raw_crop[t_y1:t_y2, :]
        
        # --- HSV Masking (Remove Green Pitch) ---
        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        mask_small = cv2.bitwise_not(cv2.inRange(hsv_torso, np.array([30, 40, 40]), np.array([95, 255, 255])))
        
        # Reconstruct full-cell mask and cutout
        full_mask = np.zeros((C_H, C_W), dtype=np.uint8)
        full_mask[t_y1:t_y2, :] = mask_small
        
        mask_bgr = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        cutout = cv2.bitwise_and(raw_crop, raw_crop, mask=full_mask)
        
        # Extract Median Color for K-Means
        jersey_pixels = torso[mask_small > 0]
        feature = np.median(jersey_pixels, axis=0) if len(jersey_pixels) > 10 else [128, 128, 128]
        
        player_data.append({
            'raw': raw_crop,
            'mask': mask_bgr,
            'cutout': cutout,
            'feat': feature
        })

    # 2. Team Clustering (The Brain)
    features = np.array([p['feat'] for p in player_data])
    labels = KMeans(n_clusters=2, n_init=10).fit(features).labels_
    
    # Assign UI Colors (Team 0 = Red, Team 1 = Blue)
    team_colors = [(0, 0, 255), (255, 0, 0)] 

    # 3. Build the Horizontal Strips
    def create_strip(data_key, is_label=False):
        cells = []
        spacer = np.zeros((C_H, 15, 3), dtype=np.uint8) + 60 # Dark divider
        for i, p in enumerate(player_data):
            if is_label:
                # Create a solid color block representing the assigned Team
                cell = np.zeros((60, C_W, 3), dtype=np.uint8)
                cell[:] = team_colors[labels[i]]
                cv2.putText(cell, f"TEAM {labels[i]}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                cell = p[data_key]
            cells.append(cell)
            cells.append(spacer if not is_label else np.zeros((60, 15, 3), dtype=np.uint8))
        return np.hstack(cells)

    raw_strip = create_strip('raw')
    mask_strip = create_strip('mask')
    cutout_strip = create_strip('cutout')
    label_strip = create_strip(None, is_label=True)

    # 4. Final Assembly with Headers
    W = raw_strip.shape[1]
    def make_header(txt, h, bg=(255,255,255), fg=(0,0,0), size=1.0):
        bar = np.zeros((h, W, 3), dtype=np.uint8) + np.array(bg, dtype=np.uint8)
        cv2.putText(bar, txt, (40, h-25), cv2.FONT_HERSHEY_SIMPLEX, size, fg, 2)
        return bar

    main_header = make_header("TEAM CLASSIFICATION", 120, size=1.5)
    h1 = make_header("STEP 1: RAW YOLOv8 DETECTIONS", 60, (40,40,40), (255,255,255), 0.8)
    h2 = make_header("STEP 2: BINARY JERSEY SEGMENTATION (HSV)", 60, (40,40,40), (255,255,255), 0.8)
    h3 = make_header("STEP 3: FEATURE EXTRACTION (GRASS REMOVAL)", 60, (40,40,40), (255,255,255), 0.8)
    h4 = make_header("STEP 4: K-MEANS TEAM ASSIGNMENT", 60, (40,40,40), (255,255,255), 0.8)

    gap = np.zeros((20, W, 3), dtype=np.uint8)
    final_stack = np.vstack((main_header, h1, raw_strip, gap, h2, mask_strip, gap, h3, cutout_strip, gap, h4, label_strip))

    # 5. Output
    output_dir = PROJ_ROOT / "results/visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the final stacked image
    save_path = output_dir / "team_classification.png"
    cv2.imwrite(str(save_path), final_stack)

    print(f"✅ Storyboard saved to: {save_path}")
    cv2.imshow("ECE324 Pipeline Storyboard", final_stack)
    cv2.imwrite(str(PROJ_ROOT / "results/visuals/team_classification.png"), final_stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TEST_IMG = PROJ_ROOT / "data/processed/yolo-calibration/images/000024.jpg"
    P_MODEL = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"
    generate_pro_storyboard(TEST_IMG, P_MODEL)