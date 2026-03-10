import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FootballPositionEstimator(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 1. Base Faster R-CNN
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 2. 2D Pitch Head (Regression)
        self.pitch_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # [x, y] on the pitch
        )

    def forward(self, images, targets=None):
        """
        Custom forward pass that extracts ROI features for pitch regression.
        """
        if self.training:
            # During training, we use the standard detection loss
            # We'll calculate the pitch loss separately in the training loop
            return self.detector(images, targets)
        
        # --- INFERENCE MODE ---
        # 1. Standard detection to get boxes and features
        # We wrap the internal logic to get 'box_features'
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.detector.transform(images, targets)
        
        # Get features from Backbone + FPN
        features = self.detector.backbone(images.tensors)
        
        # Get Proposals from RPN
        proposals, _ = self.detector.rpn(images, features, targets)
        
        # Pass to ROI Heads (This is where the box features are generated)
        # detections: list of dicts with 'boxes', 'labels', 'scores'
        # box_features: [N, 1024] tensor where N is the number of detections
        detections, _ = self.detector.roi_heads(features, proposals, images.image_sizes, targets)
        
        # To get the pitch coordinates, we need to re-pool the features for the FINAL boxes
        final_boxes = [d['boxes'] for d in detections]
        box_features = self.detector.roi_heads.box_roi_pool(features, final_boxes, images.image_sizes)
        box_features = self.detector.roi_heads.box_head(box_features) # Flatten to 1024
        
        # 2. Pass features through our custom Pitch Head
        pitch_coords = self.pitch_head(box_features)
        
        # 3. Format output
        # We split the long [N_total, 2] pitch_coords back into per-image lists
        start_idx = 0
        for i, det in enumerate(detections):
            num_boxes = len(det['boxes'])
            det['pitch_coords'] = pitch_coords[start_idx : start_idx + num_boxes]
            start_idx += num_boxes

        # Rescale boxes back to original image size
        detections = self.detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        
        return detections