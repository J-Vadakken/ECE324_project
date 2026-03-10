import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ECE324_Project.dataset import SynLocDataset, custom_collate_fn
from ECE324_Project.config import logger

def generate_samples(model_path="model_epoch_1.pth", num_samples=3):
    device = torch.device("cpu")
    
    # 1. Load Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    logger.info(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Init Dataset
    dataset = SynLocDataset(split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    logger.info("Generating stacked samples with player counts...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= num_samples: break
            
            predictions = model(images)
            base_img = (images[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            base_img = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
            h_img, w_img = base_img.shape[:2]

            img_gt = base_img.copy()
            img_preds = base_img.copy()
            
            # --- 1. PROCESS GROUND TRUTH ---
            gt_boxes = targets[0]['boxes']
            gt_count = len(gt_boxes)
            for box in gt_boxes:
                cx, cy, w, h = box
                x1, y1 = int((cx - w/2) * w_img), int((cy - h/2) * h_img)
                x2, y2 = int((cx + w/2) * w_img), int((cy + h/2) * h_img)
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # --- 2. PROCESS PREDICTIONS ---
            preds = predictions[0]
            # Filter by confidence threshold
            keep_idx = preds['scores'] > 0.4
            pred_boxes = preds['boxes'][keep_idx]
            pred_scores = preds['scores'][keep_idx]
            pred_count = len(pred_boxes)
            
            for box, score in zip(pred_boxes, pred_scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_preds, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img_preds, f"{score:.2f}", (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # --- 3. DRAW COUNT OVERLAYS ---
            # Scorecard box at the top right
            count_color = (0, 255, 0) if gt_count == pred_count else (0, 165, 255) # Green if match, Orange if not
            
            # GT Scorecard
            cv2.rectangle(img_gt, (w_img - 300, 10), (w_img - 10, 60), (0,0,0), -1)
            cv2.putText(img_gt, f"GT Count: {gt_count}", (w_img - 280, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Preds Scorecard
            cv2.rectangle(img_preds, (w_img - 350, 10), (w_img - 10, 60), (0,0,0), -1)
            cv2.putText(img_preds, f"Pred Count: {pred_count}", (w_img - 330, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, count_color, 2)

            # --- 4. STACK AND SAVE ---
            stacked_img = cv2.vconcat([img_gt, img_preds])
            out_path = f"counted_val_{i}.jpg"
            cv2.imwrite(out_path, stacked_img)
            logger.info(f"Saved {out_path} | GT: {gt_count} | Pred: {pred_count}")

if __name__ == "__main__":
    generate_samples()