import os
import json
import shutil
from pathlib import Path
from ECE324_Project.config import CALIB_DIR, YOLO_CALIB_DIR

def line_intersection_infinite(line1, line2, padding=0.15):
    """
    Extends polylines into infinite vectors to find their crossing point.
    Fixes the issue where annotators stop short of the actual corner.
    """
    if not line1 or not line2 or len(line1) < 2 or len(line2) < 2:
        return None

    # Use first and last points to define the vector direction
    x1, y1 = line1[0]['x'], line1[0]['y']
    x2, y2 = line1[-1]['x'], line1[-1]['y']
    x3, y3 = line2[0]['x'], line2[0]['y']
    x4, y4 = line2[-1]['x'], line2[-1]['y']

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-10: return None # Parallel lines

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    # Keep points slightly off-screen for homography stability
    if (-padding <= px <= 1.0 + padding) and (-padding <= py <= 1.0 + padding):
        return (px, py)
    return None

def process_calibration_dataset(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # THE RELIABLE 14
    INTERSECTION_MAP = [
        ("Big rect. right top", "Big rect. right main"),      # 0
        ("Big rect. right bottom", "Big rect. right main"),   # 1
        ("Small rect. right top", "Small rect. right main"),  # 2
        ("Small rect. right bottom", "Small rect. right main"),# 3
        ("Side line top", "Side line right"),                 # 4
        ("Side line bottom", "Side line right"),              # 5
        ("Side line top", "Middle line"),                     # 6
        ("Side line bottom", "Middle line"),                  # 7
        ("Big rect. left top", "Big rect. left main"),        # 8
        ("Big rect. left bottom", "Big rect. left main"),     # 9
        ("Small rect. left top", "Small rect. left main"),     # 10
        ("Small rect. left bottom", "Small rect. left main"), # 11
        ("Side line top", "Side line left"),                  # 12
        ("Side line bottom", "Side line left")                # 13
    ]

    for img_path in Path(json_dir).glob("*.jpg"):
        json_file = img_path.with_suffix('.json')
        if not json_file.exists(): continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        keypoints_output = []
        v_x, v_y = [], []

        for l1_key, l2_key in INTERSECTION_MAP:
            inter = line_intersection_infinite(data.get(l1_key), data.get(l2_key))
            if inter:
                keypoints_output.extend([inter[0], inter[1], 2])
                v_x.append(inter[0]); v_y.append(inter[1])
            else:
                keypoints_output.extend([0.0, 0.0, 0])

        if not v_x: continue # No points found in this frame

        # YOLO formatting
        min_x, max_x, min_y, max_y = min(v_x), max(v_x), min(v_y), max(v_y)
        bw, bh = max(max_x - min_x, 0.01), max(max_y - min_y, 0.01)
        cx, cy = min_x + (bw/2), min_y + (bh/2)

        yolo_str = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
        yolo_str += " ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in keypoints_output])

        with open(output_dir / f"{img_path.stem}.txt", 'w') as f:
            f.write(yolo_str + "\n")

if __name__ == "__main__":
    for split in ["train", "valid"]:
        process_calibration_dataset(CALIB_DIR / split, YOLO_CALIB_DIR / "labels" / split)
    print("✅ Successfully generated 14-point labels with vector extensions.")