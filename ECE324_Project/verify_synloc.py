import cv2
import json
import torch
import matplotlib.pyplot as plt
from sskit import unnormalize, world_to_image

# 1. Import your central config variables
from ECE324_Project.config import SYNLOC_ANNO_DIR, SYNLOC_DIR, SYNLOC_IMG_DIR, FIGURES_DIR, logger

def visualize_synloc(split="train", frame_index=0):
    json_path = SYNLOC_ANNO_DIR / f"{split}.json"
    
    logger.info(f"Loading annotations from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_info = data['images'][frame_index]
    
    # 2. Convert raw JSON lists into PyTorch Tensors
    cam_mat = torch.tensor(img_info['camera_matrix'], dtype=torch.float32)
    dist_poly = torch.tensor(img_info['dist_poly'], dtype=torch.float32)

    # The image path is relative to the base SpiideoSynLoc directory
    img_path = SYNLOC_IMG_DIR / img_info['file_name']
    
    logger.info(f"Reading image: {img_path}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.error(f"Could not read image at {img_path}")
        return

    players = [a for a in data['annotations'] if a['image_id'] == img_info['id']]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1]})
    ax2.set_facecolor('#2ecc71')
    ax2.add_patch(plt.Rectangle((-52.5, -34), 105, 68, fill=True, color='#27ae60', zorder=0))
    ax2.axvline(0, color='white', linewidth=2)
    ax2.add_patch(plt.Circle((0, 0), 9.15, fill=False, color='white', linewidth=2))

    for p in players:
        pos = p.get('position_on_pitch')
        if not pos: continue

        # 3. --- THE FIX ---
        # If the dataset only gave [x, y], append 0.0 for the Z-axis (ground level)
        if len(pos) == 2:
            pos_3d = torch.tensor([pos[0], pos[1], 0.0], dtype=torch.float32)
        else:
            pos_3d = torch.tensor(pos, dtype=torch.float32)

        # Plot on Radar (using .item() to pull the float out of the tensor)
        ax2.scatter(pos_3d[0].item(), pos_3d[1].item(), c='red', s=150, edgecolors='white', zorder=5)

        # 4. Project 3D meters to 2D pixels using sskit
        img_norm = world_to_image(cam_mat, dist_poly, pos_3d)
        img_pixel = unnormalize(img_norm, frame.shape)
        
        u, v = int(img_pixel[0].item()), int(img_pixel[1].item())

        cv2.circle(frame, (u, v), 15, (0, 0, 255), -1)

    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    ax2.set_xlim(-60, 60); ax2.set_ylim(-40, 40)
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"synloc_verify_{split}_{frame_index}.png"
    plt.savefig(out_path, dpi=150)
    logger.info(f"Saved visualization to: {out_path}")
    plt.show()

if __name__ == "__main__":
    visualize_synloc("train", 0)