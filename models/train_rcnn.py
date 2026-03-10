import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ECE324_Project.dataset import SynLocDataset, custom_collate_fn
from ECE324_Project.config import logger
import numpy as np

class LossTracker:
    def __init__(self):
        self.history = {"total": [], "loss_classifier": [], "loss_box_reg": [], "loss_objectness": [], "loss_rpn_box_reg": []}

    def update(self, loss_dict):
        total = 0
        for k, v in loss_dict.items():
            self.history[k].append(v.item())
            total += v.item()
        self.history["total"].append(total)

    def plot(self, save_path="training_loss_curve.png"):
        plt.figure(figsize=(10, 6))
        # Plot total loss with a thicker line
        plt.plot(self.history["total"], label="Total Loss", color='black', linewidth=2)
        # Plot individual heads to see which one is dominant
        plt.plot(self.history["loss_classifier"], label="Class Loss", alpha=0.6)
        plt.plot(self.history["loss_box_reg"], label="Box Reg Loss", alpha=0.6)
        
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Faster R-CNN Training Progress")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(save_path)
        plt.close()

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

def run_overnight_with_plots():
    device = torch.device("cpu") # Stable for overnight
    tracker = LossTracker()
    
    # 1. Init Data & Model
    dataset = SynLocDataset(split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    model.train()
    logger.info("Starting Overnight Training. You can check 'training_loss_curve.png' anytime to see progress.")

    for epoch in range(1, 11):
        for i, (images, targets) in enumerate(loader):
            images = [img.to(device) for img in images]
            rcnn_targets = prepare_targets_rcnn(targets, device)

            loss_dict = model(images, rcnn_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            tracker.update(loss_dict)
            
            if i % 50 == 0:
                logger.info(f"Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {losses.item():.4f}")
                # Save plot every 50 batches so you can check it while it runs
                tracker.plot()

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        logger.info(f"Epoch {epoch} complete. model_epoch_{epoch}.pth saved.")

if __name__ == "__main__":
    run_overnight_with_plots()