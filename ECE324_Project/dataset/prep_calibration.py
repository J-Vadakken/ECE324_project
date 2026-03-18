import os
import json
import shutil
from pathlib import Path
from ECE324_Project.config import CALIB_DIR, YOLO_CALIB_DIR

def line_intersection(line1, line2):
    """
    Calculates the intersection of two lines defined by a pair of normalized (x,y) points.
    Returns (x, y) if it intersects within the image [0,1], else None.
    """
    # Check if lines exist
    if not line1 or not line2:
        return None
        
    # Ensure both lines have at least 2 points
    if len(line1) < 2 or len(line2) < 2:
        return None

    # Line 1 points
    x1, y1 = line1[0]['x'], line1[0]['y']
    x2, y2 = line1[1]['x'], line1[1]['y']
    # Line 2 points
    x3, y3 = line2[0]['x'], line2[0]['y']
    x4, y4 = line2[1]['x'], line2[1]['y']

    # Denominator
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if D == 0:
        return None # Lines are parallel

    # Calculate intersection
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D

    # Ensure the point is actually on the screen (normalized 0.0 to 1.0)
    if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
        return (px, py)
    
    return None

def process_calibration_dataset(json_dir, output_dir):
    """
    Reads SoccerNet JSONs and outputs YOLOv8-Pose .txt files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- THE COMPLETE 14-POINT KEYPOINT MAP ---
    KEYPOINT_PAIRS = [
        # --- RIGHT SIDE ---
        ("Big rect. right top", "Big rect. right main"),       # KP 0: Right Pen Box (Top Corner)
        ("Big rect. right bottom", "Big rect. right main"),    # KP 1: Right Pen Box (Bot Corner)
        ("Small rect. right top", "Small rect. right main"),   # KP 2: Right 6yd Box (Top Corner)
        ("Small rect. right bottom", "Small rect. right main"),# KP 3: Right 6yd Box (Bot Corner)
        ("Side line top", "Side line right"),                  # KP 4: Top Right Corner Flag
        ("Side line bottom", "Side line right"),               # KP 5: Bottom Right Corner Flag
        
        # --- MIDFIELD ---
        ("Side line top", "Middle line"),                      # KP 6: Top Halfway Line Intersection
        ("Side line bottom", "Middle line"),                   # KP 7: Bottom Halfway Line Intersection
        
        # --- LEFT SIDE ---
        ("Big rect. left top", "Big rect. left main"),         # KP 8: Left Pen Box (Top Corner)
        ("Big rect. left bottom", "Big rect. left main"),      # KP 9: Left Pen Box (Bot Corner)
        ("Small rect. left top", "Small rect. left main"),     # KP 10: Left 6yd Box (Top Corner)
        ("Small rect. left bottom", "Small rect. left main"),  # KP 11: Left 6yd Box (Bot Corner)
        ("Side line top", "Side line left"),                   # KP 12: Top Left Corner Flag
        ("Side line bottom", "Side line left")                 # KP 13: Bottom Left Corner Flag
    ]
    
    NUM_KEYPOINTS = len(KEYPOINT_PAIRS)
    jpg_files = list(Path(json_dir).glob("*.jpg"))
    json_files = [img.with_suffix('.json') for img in jpg_files if img.with_suffix('.json').exists()]
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        keypoints_output = []
        valid_points_x = []
        valid_points_y = []

        # Find intersections for our defined keypoints
        for line1_key, line2_key in KEYPOINT_PAIRS:
            line1 = data.get(line1_key)
            line2 = data.get(line2_key)
            
            intersection = line_intersection(line1, line2)
            
            if intersection:
                x, y = intersection
                vis = 2 # 2 = visible and labeled in YOLO format
                keypoints_output.extend([x, y, vis])
                valid_points_x.append(x)
                valid_points_y.append(y)
            else:
                # Point is off-screen or lines are not in this frame
                keypoints_output.extend([0.0, 0.0, 0]) # 0 = not visible
                
        # If we didn't find any keypoints in this frame, skip creating a text file
        if len(valid_points_x) == 0:
            continue
            
        # YOLO needs a bounding box for the "object". Our object is the pitch.
        # We calculate the tightest bounding box around all visible keypoints.
        min_x, max_x = min(valid_points_x), max(valid_points_x)
        min_y, max_y = min(valid_points_y), max(valid_points_y)
        
        # Calculate center, width, height for YOLO
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_cx = min_x + (bbox_w / 2)
        bbox_cy = min_y + (bbox_h / 2)
        
        # Prevent zero-area boxes if only 1 point is visible
        bbox_w = max(bbox_w, 0.01)
        bbox_h = max(bbox_h, 0.01)

        # Build the final YOLO string: <class> <cx> <cy> <w> <h> <k1_x> <k1_y> <k1_v> ...
        class_id = 0 # Only 1 class: "Pitch"
        yolo_string = f"{class_id} {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f} "
        yolo_string += " ".join([f"{val:.6f}" if isinstance(val, float) else str(val) for val in keypoints_output])
        
        # Save to .txt
        txt_path = Path(output_dir) / f"{json_file.stem}.txt"
        with open(txt_path, 'w') as f:
            f.write(yolo_string + "\n")

    print(f"Processed {len(json_files)} JSONs. Generated text files in {output_dir}")


if __name__ == "__main__":
    splits = ["train", "valid"]
    
    for split in splits:
        print(f"\n--- Starting {split.upper()} split ---")
        
        # 1. Process Labels
        json_dir = CALIB_DIR / split
        labels_output_dir = YOLO_CALIB_DIR / "labels" / split
        
        print(f"Extracting keypoints to {labels_output_dir}...")
        process_calibration_dataset(json_dir, labels_output_dir)
        
        # 2. Copy Images
        images_output_dir = YOLO_CALIB_DIR / "images" / split
        os.makedirs(images_output_dir, exist_ok=True)
        
        jpg_files = list(json_dir.glob("*.jpg"))
        print(f"Copying {len(jpg_files)} images to {images_output_dir}...")
        
        for img_path in jpg_files:
            dest_path = images_output_dir / img_path.name
            if not dest_path.exists():
                shutil.copy(img_path, dest_path)
        
    print("\nData preparation complete! YOLO sandbox is ready.")