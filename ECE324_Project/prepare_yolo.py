import json
import os
from pathlib import Path
from ECE324_Project.config import SYNLOC_ANNO_DIR, SYNLOC_IMG_DIR, PROCESSED_DATA_DIR, logger

# The output folder where your YOLO training pipeline will look
YOLO_DIR = PROCESSED_DATA_DIR / "yolo_synloc"

def convert_to_yolo(split="train"):
    json_path = SYNLOC_ANNO_DIR / f"{split}.json"
    if not json_path.exists():
        logger.error(f"Could not find {json_path}")
        return

    logger.info(f"Loading {split} data...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Setup YOLO directory structure: yolo_synloc/images/train and yolo_synloc/labels/train
    img_out_dir = YOLO_DIR / "images" / split
    lbl_out_dir = YOLO_DIR / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image_id for fast lookup
    anno_dict = {}
    for anno in data['annotations']:
        img_id = anno['image_id']
        if img_id not in anno_dict:
            anno_dict[img_id] = []
        anno_dict[img_id].append(anno)

    processed_count = 0

    for img_info in data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Original Image Path
        src_img_path = SYNLOC_IMG_DIR / file_name
        if not src_img_path.exists():
            continue

        # 1. Symlink the image (Saves disk space and time)
        dst_img_path = img_out_dir / Path(file_name).name
        if not dst_img_path.exists():
            os.symlink(src_img_path, dst_img_path)

        # 2. Generate the corresponding .txt label file
        txt_name = Path(file_name).stem + ".txt"
        lbl_out_path = lbl_out_dir / txt_name
        
        # Get image dimensions for YOLO normalization
        img_w = img_info.get('width', 3840)
        img_h = img_info.get('height', 1504)

        with open(lbl_out_path, 'w') as f:
            if img_id in anno_dict:
                for anno in anno_dict[img_id]:
                    bbox = anno.get('bbox')
                    if not bbox or len(bbox) != 4:
                        continue

                    # COCO: [top_left_x, top_left_y, width, height]
                    bx, by, bw, bh = bbox

                    # YOLO: [center_x, center_y, width, height] (Normalized 0.0 to 1.0)
                    cx_norm = (bx + (bw / 2.0)) / img_w
                    cy_norm = (by + (bh / 2.0)) / img_h
                    w_norm = bw / img_w
                    h_norm = bh / img_h

                    # Class ID is 0 (Player)
                    # Clip values to ensure they stay between 0 and 1
                    cx_norm = max(0.0, min(1.0, cx_norm))
                    cy_norm = max(0.0, min(1.0, cy_norm))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    f.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        processed_count += 1
        if processed_count % 100 == 0:
            logger.info(f"Processed {processed_count} images for {split}...")

    logger.info(f"Finished {split}! Processed {processed_count} images.")

if __name__ == "__main__":
    convert_to_yolo("train")
    # convert_to_yolo("val") # Uncomment if you downloaded the validation split too!