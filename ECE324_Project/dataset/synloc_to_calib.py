import cv2
import os
import shutil
from pathlib import Path
import numpy as np
from ECE324_Project.config import PROJ_ROOT, SYNLOC_IMG_DIR

# --- CONFIG ---
IMG_DIR = SYNLOC_IMG_DIR
LABEL_DIR = PROJ_ROOT / "data/processed/yolo-calibration/labels"
IMAGE_DEST_DIR = PROJ_ROOT / "data/processed/yolo-calibration/images"

KP_NAMES = [
    "0: R-Pen Top", "1: R-Pen Bot", "2: R-Goal Top", "3: R-Goal Bot", "4: R-Corn Top", "5: R-Corn Bot",
    "6: Mid Top", "7: Mid Bot",
    "8: L-Pen Top", "9: L-Pen Bot", "10: L-Goal Top", "11: L-Goal Bot", "12: L-Corn Top", "13: L-Corn Bot"
]

# Create directories if they don't exist
LABEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DEST_DIR.mkdir(parents=True, exist_ok=True)

current_kps = [] 

def click_event(event, x, y, flags, params):
    global current_kps
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_kps) < 14:
            current_kps.append((x, y, 2))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_kps) < 14:
            current_kps.append((0, 0, 0)) # Record as invisible/skipped

def annotate():
    global current_kps
    # Get all images and sort them alphabetically
    images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))])
    
    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotator", click_event)

    for idx, img_name in enumerate(images):
        save_path = LABEL_DIR / f"{Path(img_name).stem}.txt"
        img_dest_path = IMAGE_DEST_DIR / img_name
        
        # SKIP images already annotated
        if save_path.exists(): 
            continue
        
        img = cv2.imread(str(IMG_DIR / img_name))
        if img is None: continue
        
        h, w = img.shape[:2]
        current_kps = []

        while True:
            temp_img = img.copy()
            
            # --- DRAW CURRENT POINTS ---
            for i, (px, py, vis) in enumerate(current_kps):
                if vis == 2:
                    cv2.circle(temp_img, (px, py), 5, (0, 255, 255), -1)
                    cv2.putText(temp_img, str(i), (px+10, py-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # --- STATUS BAR ---
            count_msg = f"Image {idx+1}/{len(images)}"
            kpt_msg = f"Click: {KP_NAMES[len(current_kps)]}" if len(current_kps) < 14 else "DONE (S to Save)"
            
            # Simple UI overlay at the top
            cv2.rectangle(temp_img, (0, 0), (w, 130), (0, 0, 0), -1) # Dark background for text
            cv2.putText(temp_img, f"{count_msg} | {kpt_msg}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(temp_img, "'u': Undo | 'r': Reset | 's': Save & Copy | 'q': Quit", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            cv2.imshow("Annotator", temp_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): 
                print("👋 Quitting annotator.")
                return
            if key == ord('r'): 
                current_kps = []
            if key == ord('u') and len(current_kps) > 0: 
                current_kps.pop()
            
            if key == ord('s') and len(current_kps) == 14:
                # 1. SAVE YOLO LABEL (.txt)
                with open(save_path, "w") as f:
                    # YOLO: class x_center y_center width height
                    yolo_data = ["0 0.5 0.5 1.0 1.0"] 
                    for kx, ky, kv in current_kps:
                        yolo_data.append(f"{kx/w:.6f} {ky/h:.6f} {kv}")
                    f.write(" ".join(yolo_data))
                
                # 2. COPY IMAGE TO DATASET FOLDER (.jpg)
                shutil.copy2(IMG_DIR / img_name, img_dest_path)
                
                print(f"💾 Saved label and copied image: {img_name}")
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    annotate()