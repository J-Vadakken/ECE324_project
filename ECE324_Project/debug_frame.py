import cv2
import json
from pathlib import Path

# Update these to your exact paths if needed
PROJ_ROOT = Path("/Users/lucaschoi/Documents/GitHub/ECE324_project")
GSR_DIR = PROJ_ROOT / "data" / "SoccerNetV2" / "gsr"
VIDEO_DIR = PROJ_ROOT / "data" / "SoccerNetV2" / "videos"

def verify_gt_pixels(match_id="117092", num_frames=10):
    json_path = GSR_DIR / match_id / f"{match_id}_1st.json"
    video_path = VIDEO_DIR / match_id / f"{match_id}_panorama_1st_half.mp4"
    
    out_dir = PROJ_ROOT / "output" / "gt_pixel_boxes"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Opening Video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    for f_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video.")
            break

        # Get the image ID for this specific frame
        target_img_id = str(data['images'][f_idx]['image_id'])
        
        # Find all player annotations for this frame
        players = [a for a in data.get('annotations', []) 
                   if str(a.get('image_id')) == target_img_id 
                   and a.get('supercategory') != 'pitch']

        # Draw the exact pixels straight onto the image
        count = 0
        for p in players:
            bbox = p.get('bbox_image')
            if not bbox: continue
            
            x = bbox.get('x')
            y = bbox.get('y')
            w = bbox.get('w')
            h = bbox.get('h')
            
            if x is not None and y is not None and w is not None and h is not None:
                # Team color based on 'team' attribute (left vs right)
                team = p.get('attributes', {}).get('team', 'unknown')
                color = (0, 255, 0) if team == 'left' else (0, 0, 255) # Green vs Red
                
                # Draw Box
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                
                # Draw Jersey Number/ID above the box
                jersey = p.get('attributes', {}).get('jersey', '??')
                cv2.putText(frame, f"#{jersey}", (int(x), int(y)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                count += 1

        out_path = out_dir / f"frame_{f_idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"Saved {out_path.name} with {count} bounding boxes.")

    cap.release()

if __name__ == "__main__":
    verify_gt_pixels("117092")