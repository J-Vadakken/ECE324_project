import json
import os
import pandas as pd
from pathlib import Path
from ECE324_Project.config import SYNLOC_ANNO_DIR, SYNLOC_IMG_DIR, PROCESSED_DATA_DIR, logger

# The output folder for YOLO training
YOLO_DIR = PROCESSED_DATA_DIR / "yolo-synloc"

# Verified HD Constants for your 1080p version
VERIFIED_W = 1920
VERIFIED_H = 1080

def convert_to_yolo(split="train"):
    # SoccerNet sometimes uses 'valid.json' or 'val.json' - checking both
    json_path = SYNLOC_ANNO_DIR / f"{split}.json"
    if not json_path.exists() and split == "val":
        json_path = SYNLOC_ANNO_DIR / "valid.json"
        
    if not json_path.exists():
        logger.error(f"Could not find annotation file for {split} at {json_path}")
        return

    logger.info(f"Processing {split} split...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Setup YOLO directory structure
    img_out_dir = YOLO_DIR / "images" / split
    lbl_out_dir = YOLO_DIR / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Map annotations to image IDs
    anno_dict = {img['id']: [] for img in data['images']}
    for anno in data['annotations']:
        if anno['image_id'] in anno_dict:
            anno_dict[anno['image_id']].append(anno)

    processed_count = 0

    for img_info in data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # 1. Use .resolve() to create absolute path symlinks (Fixes NoneType errors)
        src_img_path = (SYNLOC_IMG_DIR / file_name).resolve()
        if not src_img_path.exists():
            continue

        dst_img_path = img_out_dir / Path(file_name).name
        if not dst_img_path.exists():
            os.symlink(src_img_path, dst_img_path)

        # 2. Prepare label file
        txt_name = Path(file_name).stem + ".txt"
        lbl_out_path = lbl_out_dir / txt_name
        
        with open(lbl_out_path, 'w') as f:
            for anno in anno_dict.get(img_id, []):
                bbox = anno.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue

                # COCO: [top_left_x, top_left_y, width, height]
                bx, by, bw, bh = bbox

                # 3. Scale Check: If labels are 4K but images are 1080p
                # We check if coordinates exceed 1920
                if bx > 2000 or (bx + bw) > 2000:
                    bx /= 2.0  # 3840 -> 1920
                    by /= (1504 / 1080) # 1504 -> 1080
                    bw /= 2.0
                    bh /= (1504 / 1080)

                # YOLO: [center_x, center_y, width, height] (Normalized)
                cx_norm = (bx + (bw / 2.0)) / VERIFIED_W
                cy_norm = (by + (bh / 2.0)) / VERIFIED_H
                w_norm = bw / VERIFIED_W
                h_norm = bh / VERIFIED_H

                # Clamp to [0.0, 1.0]
                cx_norm = max(0.0, min(1.0, cx_norm))
                cy_norm = max(0.0, min(1.0, cy_norm))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                # class_id 0 = 'player'
                f.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        processed_count += 1
        if processed_count % 500 == 0:
            logger.info(f"Progress: {processed_count} images linked and labeled...")

    logger.info(f"Done! {split} split complete with {processed_count} images.")

if __name__ == "__main__":
    # Process both splits to ensure you have a valid val set for the trainer
    for s in ["train", "val"]:
        convert_to_yolo(s)