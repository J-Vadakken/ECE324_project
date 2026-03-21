import cv2
import random
from pathlib import Path
from ECE324_Project.config import SYNLOC_IMG_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, logger

def visualize_random_yolo_labels(num_images=5, split="train"):
    # Paths to your newly generated YOLO labels
    labels_dir = PROCESSED_DATA_DIR / "yolo-calibration" / "labels" / split
    output_dir = PROJ_ROOT / "debug_labels" / "yolo_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The exact 11 keypoints in the order we generated them
    kp_names = [
        "Center", "L-Goal", "R-Goal",
        "TL-Corner", "BL-Corner", "TR-Corner", "BR-Corner",
        "TL-Penalty", "BL-Penalty", "TR-Penalty", "BR-Penalty"
    ]

    # Grab all .txt files
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        logger.error(f"❌ No label files found in {labels_dir}. Did the generation script finish?")
        return

    # Pick 5 random files
    sample_files = random.sample(label_files, min(num_images, len(label_files)))

    for txt_path in sample_files:
        img_name = txt_path.stem + ".jpg"
        img_path = SYNLOC_IMG_DIR / img_name

        if not img_path.exists():
            logger.warning(f"⚠️ Could not find image {img_name} for label {txt_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]

        # Read the YOLO format text file
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # YOLO format: class x_center y_center width height px1 py1 vis1 px2 py2 vis2 ...
            # We skip the first 5 elements (bbox info) and just grab the keypoints
            kp_data = parts[5:]

            for i in range(len(kp_names)):
                idx = i * 3
                if idx + 2 >= len(kp_data):
                    break

                x_norm = float(kp_data[idx])
                y_norm = float(kp_data[idx+1])
                vis = float(kp_data[idx+2])

                # If visibility is "2" (labeled and visible), draw it!
                if vis == 2.0:
                    # Un-normalize from percentages back to absolute pixels
                    pixel_x = int(x_norm * w)
                    pixel_y = int(y_norm * h)

                    # Draw a bright cyan dot and label
                    cv2.circle(img, (pixel_x, pixel_y), radius=8, color=(255, 255, 0), thickness=-1)
                    cv2.putText(img, kp_names[i], (pixel_x + 10, pixel_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out_path = output_dir / f"verified_{img_name}"
        cv2.imwrite(str(out_path), img)
        logger.info(f"✅ Saved verification image to {out_path.relative_to(PROJ_ROOT)}")

if __name__ == "__main__":
    visualize_random_yolo_labels(num_images=5, split="train")