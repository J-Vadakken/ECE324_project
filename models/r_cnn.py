import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class FootballPositionEstimator(nn.Module):
    def __init__(self, num_classes=2): # 1: Player, 0: Background
        super().__init__()
        # Load pre-trained Faster R-CNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        # 1. Replace the standard classification/box head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 2. Add the custom 3D Pitch Regression Head
        # This takes the 1024-dimensional feature vector from the ROI pooling
        self.pitch_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # Output: [x, y] meters on pitch
        )

    def forward(self, images, targets=None):
        """
        In training mode, this returns the standard detection losses.
        We will manually extract the ROI features for the pitch loss.
        """
        if self.training:
            # Standard detection forward pass
            return self.model(images, targets)
        else:
            # Inference: return detections + predicted pitch coords
            return self.model(images)

def get_model():
    return FootballPositionEstimator(num_classes=2)