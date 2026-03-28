import shutil
import os
import cv2
import random
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT, SYNLOC_IMG_DIR

def sync_and_verify(num_samples=5):
    # 1. SETUP PATHS
    base_dir = PROJ_ROOT / "data/processed/yolo-calibration"
    label_dir = base_dir / "labels" 
    image_dest_dir = base_dir / "images" 
    
    image_dest_dir.mkdir(parents=True, exist_ok=True)

    # --- PART 1: SYNCING ---
    print("🔄 Step 1: Syncing images to manual folder...")
    
    # Get the sorted list of source images (Exact same logic as your annotator)
    all_source_images = sorted([
        f for f in os.listdir(SYNLOC_IMG_DIR) 
        if f.endswith(('.jpg', '.png'))
    ])

    # Get labels currently in your manual folder
    existing_labels = {f.stem: f for f in label_dir.glob("*.txt")}
    
    if not existing_labels:
        print("❌ No manual labels found in labels/manual. Run the annotator first!")
        return

    copy_count = 0
    for img_name in all_source_images:
        img_stem = Path(img_name).stem
        if img_stem in existing_labels:
            src_path = SYNLOC_IMG_DIR / img_name
            dst_path = image_dest_dir / img_name
            
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copy_count += 1

    print(f"✅ Sync complete. {copy_count} new images copied.")
    print(f"📂 Total manual images ready: {len(list(image_dest_dir.glob('*')))}")
    print("-" * 30)

    # --- PART 2: VERIFICATION POP-UP ---
    print("🖼️  Step 2: Opening Verification Window...")
    print("👉 Press ANY KEY for next image | Press 'q' to QUIT")

    label_files = list(label_dir.glob("*.txt"))
    # Sample from what we actually have labels for
    samples = random.sample(label_files, min(num_samples, len(label_files)))

    for lbl_path in samples:
        img_path = image_dest_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            img_path = image_dest_dir / f"{lbl_path.stem}.png"
        
        if not img_path.exists():
            print(f"⚠️ Image file missing for {lbl_path.name}")
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        with open(lbl_path, 'r') as f:
            data = f.readline().strip().split()
        
        # YOLO: [class, cx, cy, w, h, x1, y1, v1, ... x14, y14, v14]
        # Keypoints start at index 5
        kpts = data[5:]

        for i in range(0, len(kpts), 3):
            idx = i // 3
            kx, ky = float(kpts[i]) * w, float(kpts[i+1]) * h
            vis = int(kpts[i+2])

            if vis == 2: # Visible/Labeled
                # Green dot for the intersection
                cv2.circle(img, (int(kx), int(ky)), 6, (0, 255, 0), -1)
                # Label ID (0-13)
                cv2.putText(img, str(idx), (int(kx)+12, int(ky)-12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Manual Calibration Check", img)
        
        # Wait for key press. If 'q', exit.
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("👋 Quitting verification.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change '5' to the number of images you want to see per run
    sync_and_verify(num_samples=10)