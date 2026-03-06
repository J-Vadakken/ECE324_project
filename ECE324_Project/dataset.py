import cv2
import json
from loguru import logger
from ECE324_Project.config import GSR_DIR, VIDEO_DIR

class SoccerTrackDataset:
    def __init__(self, match_id, half=1):
        self.match_id = match_id
        suffix = "1st" if half == 1 else "2nd"
        self.video_path = VIDEO_DIR / match_id / f"{match_id}_panorama_{suffix}_half.mp4"
        self.json_path = GSR_DIR / match_id / f"{match_id}_{suffix}.json"

        self.cap = cv2.VideoCapture(str(self.video_path))
        
        logger.info(f"Loading COCO-style Ground Truth for Match {match_id}...")
        with open(self.json_path, 'r') as f:
            full_data = json.load(f)

        self.img_id_to_frame = {}
        for img in full_data.get('images', []):
            frame_num = int(img['file_name'].split('.')[0])
            self.img_id_to_frame[img['image_id']] = frame_num

        # Group annotations by frame number
        self.frame_to_annotations = {}
        for ann in full_data.get('annotations', []):
            img_id = ann['image_id']
            frame_num = self.img_id_to_frame.get(img_id)
            if frame_num not in self.frame_to_annotations:
                self.frame_to_annotations[frame_num] = []
            self.frame_to_annotations[frame_num].append(ann)

        logger.success(f"Indexed {len(self.frame_to_annotations)} labeled frames.")

    def find_active_range(self):
        indices = sorted(self.frame_to_annotations.keys())
        return (indices[0], indices[-1]) if indices else (None, None)

    def get_frame_and_labels(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        labels = self.frame_to_annotations.get(frame_idx, [])
        return frame, labels

    def close(self):
        self.cap.release()

def run_test():
    ds = SoccerTrackDataset("117092")
    start, end = ds.find_active_range()
    
    first_player_frame = None
    player_sample = None
    
    logger.info("Searching for the first frame with active players...")
    
    for i in range(start, end, 100):
        _, annotations = ds.get_frame_and_labels(i)
        players = [a for a in annotations if a.get('category_id') == 1]
        if players:
            first_player_frame = i
            player_sample = players[0]
            break
            
    if first_player_frame:
        logger.success(f"🏃 First player found at Frame {first_player_frame}!")
        logger.info(f"📍 Sample Player Data: {player_sample}")
        
        # Verify the coordinate keys
        keys = player_sample.keys()
        logger.info(f"Available keys in player object: {list(keys)}")
    else:
        logger.warning("Could not find any players in the sampled range.")

    ds.close()
    
if __name__ == "__main__":
    run_test()