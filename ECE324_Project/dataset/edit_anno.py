import cv2
import os
from pathlib import Path
import numpy as np
from ECE324_Project.config import PROJ_ROOT, SYNLOC_IMG_DIR

# --- CONFIG ---
IMG_DIR = SYNLOC_IMG_DIR
LABEL_DIR = PROJ_ROOT / "data/processed/yolo-calibration/labels"
KP_NAMES = [
    "0: R-Pen Top", "1: R-Pen Bot", "2: R-Goal Top", "3: R-Goal Bot", "4: R-Corn Top", "5: R-Corn Bot",
    "6: Mid Top", "7: Mid Bot",
    "8: L-Pen Top", "9: L-Pen Bot", "10: L-Goal Top", "11: L-Goal Bot", "12: L-Corn Top", "13: L-Corn Bot"
]

current_kps = [] 
mouse_pos = (0, 0)
edit_idx = 0 # Which point are we currently moving?

def click_event(event, x, y, flags, params):
    global current_kps, mouse_pos, edit_idx
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if edit_idx < 14:
            current_kps[edit_idx] = (x, y, 2)
            print(f"✅ Updated {KP_NAMES[edit_idx]}")
            edit_idx += 1 # Move to the next point in the sequence

def edit_annotations():
    global current_kps, mouse_pos, edit_idx
    
    # Get only images that ALREADY have labels
    labeled_stems = [f.stem for f in LABEL_DIR.glob("*.txt")]
    images = sorted([f for f in os.listdir(IMG_DIR) if Path(f).stem in labeled_stems])
    
    if not images:
        print("❌ No labels found to edit!")
        return

    cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Editor", click_event)

    for img_name in images:
        label_path = LABEL_DIR / f"{Path(img_name).stem}.txt"
        img = cv2.imread(str(IMG_DIR / img_name))
        h, w = img.shape[:2]
        
        # --- LOAD EXISTING DATA ---
        with open(label_path, 'r') as f:
            data = f.readline().strip().split()
            kpts_raw = data[5:]
            current_kps = []
            for i in range(0, len(kpts_raw), 3):
                kx, ky = float(kpts_raw[i]) * w, float(kpts_raw[i+1]) * h
                vis = int(kpts_raw[i+2])
                current_kps.append((int(kx), int(ky), vis))
        
        edit_idx = 0 # Start editing from point 0 for each image

        while True:
            temp_img = img.copy()
            
            # Draw all points
            for i, (px, py, vis) in enumerate(current_kps):
                color = (0, 255, 0) if i != edit_idx else (0, 0, 255) # Red for the one being edited
                if vis == 2:
                    cv2.circle(temp_img, (px, py), 4, color, -1)
                    cv2.putText(temp_img, str(i), (px+8, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)



            # UI Text
            msg = f"Editing: {KP_NAMES[edit_idx]}" if edit_idx < 14 else "All 14 checked! 'S' to Overwrite"
            cv2.putText(temp_img, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(temp_img, "'n': Skip to next point | 'r': Reset this image | 's': Save | 'q': Quit", 
                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Editor", temp_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): return
            if key == ord('n'): edit_idx = (edit_idx + 1) % 15 # Cycle through points
            if key == ord('r'): edit_idx = 0 # Restart editing points
            
            if key == ord('s'):
                with open(label_path, "w") as f:
                    line = ["0 0.5 0.5 1.0 1.0"]
                    for kx, ky, kv in current_kps:
                        line.append(f"{kx/w:.6f} {ky/h:.6f} {kv}")
                    f.write(" ".join(line))
                print(f"💾 Overwrote {img_name}")
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    edit_annotations()