import cv2
import json
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import torch

from sskit import unnormalize, world_to_image
from ECE324_Project.config import SYNLOC_ANNO_DIR, SYNLOC_IMG_DIR, REPORTS_DIR, logger

def visualize_verified_labels(split="train", frame_index=0, save=True):

    json_path = SYNLOC_ANNO_DIR / f"{split}.json"
    
    logger.info(f"Loading data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Load image and camera info
    img_info = data['images'][frame_index]
    img_path = SYNLOC_IMG_DIR / img_info['file_name']
    
    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.error(f"Could not load image at {img_path}")
        return
        
    players = [a for a in data['annotations'] if a['image_id'] == img_info['id']]

    # Calculate scaling factors for Bounding Boxes (to handle 4K vs 1080p mismatch)
    json_w = img_info.get('width', frame.shape[1])
    json_h = img_info.get('height', frame.shape[0])
    img_h, img_w = frame.shape[:2]
    
    scale_x = img_w / json_w
    scale_y = img_h / json_h

    fig = plt.figure(figsize=(20, 18))
    ax1 = fig.add_subplot(2, 1, 1) # Top: 4K Image
    ax2 = fig.add_subplot(2, 1, 2) # Bottom: 2D Pitch
    
    # Setup the mplsoccer Pitch to use Custom Dimensions
    pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68,
                  pitch_color='#2ecc71', line_color='white', stripe=True, linewidth=3)
    
    pitch.draw(ax=ax2)

    # Process and Plot Annotations
    players_on_pitch_count = 0
    
    for p in players:
        bbox = p.get('bbox')
        if bbox and len(bbox) == 4:
            # Scale the [top_left_x, top_left_y, width, height]
            bx = int(bbox[0] * scale_x)
            by = int(bbox[1] * scale_y)
            bw = int(bbox[2] * scale_x)
            bh = int(bbox[3] * scale_y)
            
            # Draw only the clean green bounding box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{p['id']}", (bx, by - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        pos = p.get('position_on_pitch')
        if not pos: 
            continue # This player exists but has no Ground Truth (x,y) location

        # Handle missing Z-coordinates (we still need sskit tensors to manage data types)
        pos_3d = torch.tensor([pos[0], pos[1], 0.0 if len(pos) == 2 else pos[2]], dtype=torch.float32)

        plot_x = pos_3d[1].item() + 52.5 
        plot_y = pos_3d[0].item() + 34.0 

        # Plot the player on the 2D Pitch map
        pitch.scatter(plot_x, plot_y, ax=ax2, 
                      c='red', s=180, edgecolors='white', linewidth=2, zorder=5)
        players_on_pitch_count += 1

    # Debug reporting
    logger.info(f"Image has {len(players)} bounding boxes.")
    logger.info(f"Found 2D coordinates for {players_on_pitch_count} of them (Mapping Issue should be fixed now).")
    
    if players_on_pitch_count < (len(players) * 0.8):
        logger.warning("Fewer than 80% of players have pitch coordinates in the JSON Ground Truth.")

    # Display Image
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    ax1.set_title(f"Camera View (Bboxes Only) - Frame {frame_index}", fontsize=18)
    
    # 2D title
    ax2.set_title(f"Tactical Radar (mplsoccer METER Mapping): {players_on_pitch_count} Players Found", fontsize=18)
    
    plt.tight_layout()
    
    # Save the image
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / f"verified_labels_{split}_{frame_index:04d}.png"
        plt.savefig(out_path, dpi=120)
        logger.info(f"Saved verified visualization to: {out_path}")
        
    plt.show()

if __name__ == "__main__":
    # Visualize and save the first frame of the train split
    visualize_verified_labels("train", 0, save=True)