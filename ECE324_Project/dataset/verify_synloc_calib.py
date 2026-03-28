import cv2
import numpy as np
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT

def verify_labels():
    img_dir = PROJ_ROOT / "data/processed/yolo-calibration/images"
    lbl_dir = PROJ_ROOT / "data/processed/yolo-calibration/labels"
    
    # Get all label files
    labels = list(lbl_dir.glob("*.txt"))
    
    for lbl_path in labels:
        img_path = img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{lbl_path.stem}.png"
            
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        with open(lbl_path, 'r') as f:
            # Format: class cx cy w h k1_x k1_y k1_v ...
            content = f.read().split()
            kpts = content[5:] # Skip the bounding box info
            
        for i in range(0, len(kpts), 3):
            idx = i // 3
            kx = float(kpts[i]) * w
            ky = float(kpts[i+1]) * h
            vis = int(kpts[i+2])
            
            if vis > 0:
                color = (0, 255, 0) if vis == 2 else (0, 0, 255)
                cv2.circle(img, (int(kx), int(ky)), 5, color, -1)
                cv2.putText(img, str(idx), (int(kx)+5, int(ky)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Label Verification", img)
        print(f"Checking: {lbl_path.name} | Press 'q' to quit, any other key for next.")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_labels()