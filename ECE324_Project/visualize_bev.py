import torch
import cv2
import numpy as np
from models.football_model import FootballPositionEstimator
from ECE324_Project.dataset import SynLocDataset, custom_collate_fn
from ECE324_Project.config import logger

def draw_pitch():
    # 900x600 canvas (1.5:1 ratio roughly matching a pitch)
    pitch = np.zeros((600, 900, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34) # Forest Green
    
    # Draw White Lines
    cv2.rectangle(pitch, (50, 50), (850, 550), (255, 255, 255), 2) # Boundary
    cv2.line(pitch, (450, 50), (450, 550), (255, 255, 255), 2)     # Halfway
    cv2.circle(pitch, (450, 300), 60, (255, 255, 255), 2)         # Center Circle
    return pitch

def map_to_pixels(coords, pitch_w=105, pitch_h=68, img_w=800, img_h=500, offset=(50, 50)):
    # Spiideo/SoccerNet often use X as length (105) and Y as width (68)
    # Map [-52.5, 52.5] -> [0, 800] and [-34, 34] -> [0, 500]
    x, y = coords
    px = int(((x + pitch_w/2) / pitch_w) * img_w) + offset[0]
    py = int(((y + pitch_h/2) / pitch_h) * img_h) + offset[1]
    return px, py

def run_bev_val(model_path="model_epoch_1.pth"):
    device = torch.device("cpu")
    model = FootballPositionEstimator().to(device)
    
    # --- SMART LOADING ---
    logger.info(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if we are loading raw detector weights or the full wrapper
    first_key = list(state_dict.keys())[0]
    if not first_key.startswith("detector."):
        logger.info("Raw detector weights found. Mapping to 'model.detector'...")
        model.detector.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()

    dataset = SynLocDataset(split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for i, (images, targets) in enumerate(loader):
        if i >= 5: break
        
        with torch.no_grad():
            # This custom forward returns detections with 'pitch_coords' keys
            output = model(images) 
        
        pitch_img = draw_pitch()
        
        # 1. Plot Ground Truth (RED)
        gt_coords = targets[0]['pitch_coords']
        for coord in gt_coords:
            px, py = map_to_pixels(coord)
            cv2.circle(pitch_img, (px, py), 6, (0, 0, 255), -1)

        # 2. Plot Predictions (GREEN)
        det = output[0]
        for coord, score in zip(det['pitch_coords'], det['scores']):
            if score > 0.4:
                px, py = map_to_pixels(coord)
                cv2.circle(pitch_img, (px, py), 4, (0, 255, 0), -1)

        cv2.putText(pitch_img, f"Frame {i} - RED: GT, GREEN: Pred", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(f"bev_output_{i}.jpg", pitch_img)
        logger.info(f"Saved bev_output_{i}.jpg")

if __name__ == "__main__":
    run_bev_val()