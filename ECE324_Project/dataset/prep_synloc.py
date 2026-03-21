import json
import os
import shutil
from pathlib import Path
from ECE324_Project.config import SYNLOC_ANNO_DIR, SYNLOC_IMG_DIR, PROCESSED_DATA_DIR, logger

# Output folder for YOLO training
YOLO_DIR = PROCESSED_DATA_DIR / "yolo-synloc"

def prep_synloc(split="train"):
    json_path = SYNLOC_ANNO_DIR / f"{split}.json"
    if not json_path.exists() and split == "val":
        json_path = SYNLOC_ANNO_DIR / "valid.json"
        
    if not json_path.exists():
        logger.error(f"❌ Could not find annotation file for {split} at {json_path}")
        return

    logger.info(f"🚀 Processing {split} split using native JSON dimensions...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Setup YOLO directory structure
    img_out_dir = YOLO_DIR / "images" / split
    lbl_out_dir = YOLO_DIR / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Map annotations to image IDs for faster lookup
    anno_dict = {img['id']: [] for img in data['images']}
    for anno in data['annotations']:
        if 'image_id' in anno:
            anno_dict[anno['image_id']].append(anno)

    processed_count = 0

    for img_info in data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Extract the native dimensions directly from the JSON
        json_w = img_info['width']
        json_h = img_info['height']
        
        src_img_path = (SYNLOC_IMG_DIR / file_name).resolve()
        if not src_img_path.exists():
            continue

        # Create symlink for the image
        dst_img_path = img_out_dir / Path(file_name).name
        if not dst_img_path.exists():
            os.symlink(src_img_path, dst_img_path)

        txt_name = Path(file_name).stem + ".txt"
        lbl_out_path = lbl_out_dir / txt_name
        
        with open(lbl_out_path, 'w') as f:
            for anno in anno_dict.get(img_id, []):
                bbox = anno.get('bbox') # [x, y, w, h]
                if not bbox: continue

                bx, by, bw, bh = bbox

                # 1. YOLO Normalization using the JSON's native dimensions
                cx_norm = (bx + (bw / 2.0)) / json_w
                cy_norm = (by + (bh / 2.0)) / json_h
                w_norm = bw / json_w
                h_norm = bh / json_h

                # 2. Clamp values just in case they bleed slightly off the 4K canvas
                cx_norm = max(0.0, min(1.0, cx_norm))
                cy_norm = max(0.0, min(1.0, cy_norm))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                # 3. Class ID fix (JSON is 1, YOLO expects 0)
                # Since we are only tracking players, we can hardcode 0
                class_id = 0

                f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        processed_count += 1
        if processed_count % 500 == 0:
            logger.info(f"⏳ Processed {processed_count} images...")

    logger.info(f"✅ {split} split complete. Total: {processed_count}")

if __name__ == "__main__":
    # Clean the old corrupted directory first
    if YOLO_DIR.exists():
        logger.info("🧹 Cleaning old YOLO directory...")
        shutil.rmtree(YOLO_DIR)
    
    for s in ["train", "val"]:
        prep_synloc(s)