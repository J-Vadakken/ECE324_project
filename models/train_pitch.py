import torch
import torch.nn as nn
from models.football_model import FootballPositionEstimator
from models.matcher import HungarianMatcher
from ECE324_Project.dataset import SynLocDataset, custom_collate_fn
from ECE324_Project.config import logger
import os

# Local helper to avoid import errors
def prepare_targets_rcnn(targets, device, img_size=(736, 1920)):
    h_max, w_max = img_size
    new_targets = []
    for t in targets:
        boxes = t["boxes"]
        if boxes.shape[0] > 0:
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1, y1 = (cx - w/2) * w_max, (cy - h/2) * h_max
            x2, y2 = (cx + w/2) * w_max, (cy + h/2) * h_max
            rescaled = torch.stack((x1, y1, x2, y2), dim=1)
            rescaled[:, [0, 2]] = rescaled[:, [0, 2]].clamp(0, w_max - 1)
            rescaled[:, [1, 3]] = rescaled[:, [1, 3]].clamp(0, h_max - 1)
            keep = (rescaled[:, 2] > rescaled[:, 0]) & (rescaled[:, 3] > rescaled[:, 1])
            new_targets.append({"boxes": rescaled[keep].to(device), "labels": t["labels"][keep].to(device)})
        else:
            new_targets.append({"boxes": torch.zeros((0, 4), device=device), "labels": torch.zeros((0,), dtype=torch.int64, device=device)})
    return new_targets

def train_pitch_stable():
    device = torch.device("cpu")
    model = FootballPositionEstimator().to(device)

    # --- 1. OPEN THE FLOODGATES ---
    model.detector.roi_heads.score_thresh = 0.05  # Catch even the "maybe" players
    model.detector.roi_heads.nms_thresh = 0.7     # Allow more overlapping candidates
    model.detector.roi_heads.detections_per_img = 200 # Don't cap the search
    
    # 2. LOAD DETECTOR WEIGHTS
    weights_path = "model_epoch_1.pth"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        if not list(state_dict.keys())[0].startswith("detector."):
            model.detector.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        logger.info(f"Detector weights loaded. Ready for geometry training.")
    
    # 3. FREEZE BACKBONE
    for param in model.detector.parameters(): param.requires_grad = False
    for param in model.pitch_head.parameters(): param.requires_grad = True

    matcher = HungarianMatcher() 
    optimizer = torch.optim.AdamW(model.pitch_head.parameters(), lr=1e-3)
    mse_loss_fn = nn.MSELoss()

    dataset = SynLocDataset(split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    model.train()
    logger.info("Starting Multi-Camera Geometry Learning...")

    for epoch in range(1, 4):
        for i, (images, targets) in enumerate(loader):
            images = [img.to(device) for img in images]
            rcnn_targets = prepare_targets_rcnn(targets, device)

            # Detection Pass (eval mode to avoid 'targets=None' error)
            model.detector.eval()
            with torch.no_grad():
                images_list, _ = model.detector.transform(images)
                features = model.detector.backbone(images_list.tensors)
                proposals, _ = model.detector.rpn(images_list, features)
                detections, _ = model.detector.roi_heads(features, proposals, images_list.image_sizes)
            model.detector.train()

            # Match and Regress
            batch_loss = torch.tensor(0.0, device=device)
            matches_count = 0
            for b_idx in range(len(images)):
                matches = matcher(detections[b_idx]['boxes'], rcnn_targets[b_idx]['boxes'])
                
                if matches:
                    p_idx = [m[0] for m in matches]
                    g_idx = [m[1] for m in matches]
                    
                    # Gradient-enabled re-pooling for matched boxes
                    box_feats = model.detector.roi_heads.box_roi_pool(features, [detections[b_idx]['boxes'][p_idx]], [images[b_idx].shape[-2:]])
                    box_feats = model.detector.roi_heads.box_head(box_feats)
                    
                    preds = model.pitch_head(box_feats)
                    gt = targets[b_idx]['pitch_coords'][g_idx].to(device)
                    
                    batch_loss += mse_loss_fn(preds, gt)
                    matches_count += len(matches)

            # Update weights
            if matches_count > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if i % 20 == 0:
                loss_val = batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
                avg_l = loss_val / (matches_count + 1e-6)
                boxes_found = len(detections[0]['boxes'])
                logger.info(f"Batch {i} | MSE: {avg_l:.2f} | Matches: {matches_count} | Found: {boxes_found}")
        torch.save(model.state_dict(), f"pitch_model_checkpoint_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_pitch_stable()