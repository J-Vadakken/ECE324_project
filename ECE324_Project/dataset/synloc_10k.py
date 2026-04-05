import os
import random
import shutil
from pathlib import Path
from ECE324_Project.config import PROJ_ROOT, logger

def create_custom_split(total_size=10000, val_ratio=0.2):
    # 1. We ONLY pull from the TRAIN folder because we verified those labels are perfect
    src_img_dir = PROJ_ROOT / "data" / "processed" / "yolo-synloc" / "images" / "train"
    src_lbl_dir = PROJ_ROOT / "data" / "processed" / "yolo-synloc" / "labels" / "train"

    # 2. The new clean dataset directory
    target_dir = PROJ_ROOT / "data" / "processed" / "yolo-synloc-10k"

    # Clean out the target dir if it exists so we don't mix old corrupted val data
    if target_dir.exists():
        logger.info("🧹 Wiping old 10k directory...")
        shutil.rmtree(target_dir)

    # Create new train/val folder structures
    for split in ["train", "val"]:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 3. Gather perfectly matched pairs
    all_images = list(src_img_dir.glob("*.jpg"))
    valid_pairs = []
    for img_path in all_images:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))

    if len(valid_pairs) < total_size:
        logger.error(f"❌ Not enough valid pairs! Found {len(valid_pairs)}, need {total_size}.")
        return

    logger.info(f"✅ Found {len(valid_pairs)} verified pairs. Shuffling and splitting...")

    # 4. Shuffle to ensure random distribution and prevent data leakage
    random.seed(42)
    random.shuffle(valid_pairs)

    # 5. Calculate split sizes (8000 train, 2000 val)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:total_size]

    def copy_pairs(pairs, split_name):
        img_dest_dir = target_dir / "images" / split_name
        lbl_dest_dir = target_dir / "labels" / split_name
        for i, (img_src, lbl_src) in enumerate(pairs):
            shutil.copy2(img_src, img_dest_dir / img_src.name)
            shutil.copy2(lbl_src, lbl_dest_dir / lbl_src.name)
            if (i + 1) % 1000 == 0:
                logger.info(f"[{split_name.upper()}] Copied {i+1} files...")

    logger.info(f"📦 Creating Train Split ({train_size} images)...")
    copy_pairs(train_pairs, "train")

    logger.info(f"📦 Creating Val Split ({val_size} images)...")
    copy_pairs(val_pairs, "val")

    logger.info("🎉 Custom split complete! Your train and val labels are now mathematically identical.")

if __name__ == "__main__":
    create_custom_split(10000, 0.2)