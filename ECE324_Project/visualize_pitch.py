import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.football_model import FootballPositionEstimator
from ECE324_Project.dataset import SynLocDataset, custom_collate_fn
import os

def draw_pitch():
    # 900x600 canvas
    pitch = np.zeros((600, 900, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34) # Green
    cv2.rectangle(pitch, (50, 50), (850, 550), (255, 255, 255), 2) # Boundary
    cv2.line(pitch, (450, 50), (450, 550), (255, 255, 255), 2)     # Halfway
    cv2.circle(pitch, (450, 300), 60, (255, 255, 255), 2)         # Center
    return pitch

def map_to_pixels(coords, pitch_w=105, pitch_h=68, img_w=800, img_h=500, offset=(50, 50)):
    x, y = coords
    px = int(((x + pitch_w/2) / pitch_w) * img_w) + offset[0]
    py = int(((y + pitch_h/2) / pitch_h) * img_h) + offset[1]
    return px, py

def run_visualizer(model_path="pitch_model_checkpoint_epoch_1.pth"):
    device = torch.device("cpu")
    model = FootballPositionEstimator().to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using random weights for testing.")

    model.eval()
    dataset = SynLocDataset(split="train") 
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for i, (images, targets) in enumerate(loader):
        if i >= 5: break
        
        with torch.no_grad():
            output = model(images)
        
        # --- Prepare Camera View (Left Plot) ---
        img_bgr = (images[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        # --- Prepare Pitch View (Right Plot) ---
        pitch_img = draw_pitch()
        
        # 1. Plot Ground Truth (RED)
        gt_coords = targets[0]['pitch_coords']
        for coord in gt_coords:
            px, py = map_to_pixels(coord)
            cv2.circle(pitch_img, (px, py), 8, (0, 0, 255), -1)

        # 2. Plot Predictions (GREEN)
        det = output[0]
        pred_count = 0
        for coord, score in zip(det['pitch_coords'], det['scores']):
            if score > 0.3: # Match the lenient threshold from training
                px, py = map_to_pixels(coord)
                cv2.circle(pitch_img, (px, py), 5, (0, 255, 0), -1)
                pred_count += 1

        # --- COMBINE INTO ONE IMAGE ---
        # Resize camera view to match pitch height (600px)
        h, w, _ = img_bgr.shape
        scale = 600 / h
        img_resized = cv2.resize(img_bgr, (int(w * scale), 600))
        
        combined = np.hstack((img_resized, pitch_img))
        
        # Add Text Overlays
        cv2.putText(combined, f"GT Players: {len(gt_coords)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(combined, f"Pred Players: {pred_count}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out_path = f"results_frame_{i}.jpg"
        cv2.imwrite(out_path, combined)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    # Check if the file exists yet, if not, wait for it!
    run_visualizer()