import os
import cv2
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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

# --- DATASET ---
class SynLocDataset(Dataset):
    def __init__(self, split="train", img_size=(736, 1920)):
        from ECE324_Project.config import PROCESSED_DATA_DIR, SYNLOC_ANNO_DIR
        self.img_dir = PROCESSED_DATA_DIR / "yolo-synloc" / "images" / split
        self.lbl_dir = PROCESSED_DATA_DIR / "yolo-synloc" / "labels" / split
        self.img_size = img_size

        with open(SYNLOC_ANNO_DIR / f"{split}.json", 'r') as f:
            data = json.load(f)
            
        # Group 3D positions by image_id to ensure 1:1 mapping with bboxes
        self.pitch_data = {}
        for anno in data['annotations']:
            img_id = anno['image_id']
            if img_id not in self.pitch_data:
                self.pitch_data[img_id] = []
            self.pitch_data[img_id].append(anno.get('position_on_pitch', [0.0, 0.0])[:2])

        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.transforms = T.Compose([T.ToPILImage(), T.Resize(self.img_size), T.ToTensor()])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img = cv2.imread(str(self.img_dir / img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load 2D Boxes (Normalized YOLO format: [cx, cy, w, h])
        boxes, labels = [], []
        lbl_path = self.lbl_dir / (Path(img_name).stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    labels.append(int(parts[0]) + 1) # Shift 0->1 (Background is 0)
                    boxes.append([float(x) for x in parts[1:]])

        # Load 3D Coordinates
        pitch_coords = torch.tensor(self.pitch_data.get(idx, []), dtype=torch.float32)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pitch_coords": pitch_coords 
        }
        return self.transforms(img), target

# --- CUSTOM COLLATE FN ---
def custom_collate_fn(batch):
    """Handles variable number of players per image."""
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)