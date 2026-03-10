import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

class HungarianMatcher(torch.nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        # These weights can be tuned, but distance (bbox) is usually key
        self.cost_bbox = cost_bbox 
        self.iou_threshold = 0.05 # Almost zero, we rely on distance now

    @torch.no_grad()
    def forward(self, pred_boxes, gt_boxes):
        """
        pred_boxes: [N, 4] (x1, y1, x2, y2)
        gt_boxes: [M, 4] (x1, y1, x2, y2)
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return []

        # 1. Calculate IoU Cost (1 - IoU)
        ious = box_iou(pred_boxes, gt_boxes)
        cost_iou = 1.0 - ious

        # 2. Calculate Center Distance (L1 or L2)
        # Find centroids
        pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        
        # Calculate pairwise L1 distance (Manhattan) normalized by image size
        # This acts as a robust 'proximity' metric
        cost_bbox = torch.cdist(pred_centers, gt_centers, p=1) / 1000.0

        # 3. Final Cost Matrix: Weighted sum
        # We want a match if it's either high overlap OR very close spatially
        C = self.cost_bbox * cost_bbox + cost_iou
        C = C.cpu().numpy()

        # 4. Hungarian Assignment
        pred_indices, gt_indices = linear_sum_assignment(C)
        
        # 5. Final Filtering
        # Because we used distance, we don't need a hard IoU threshold anymore!
        # We just need to make sure the cost isn't 'infinite' (e.g. across the whole pitch)
        matches = []
        for p, g in zip(pred_indices, gt_indices):
            # If the cost is reasonable (e.g. they aren't on opposite sides of the screen)
            if C[p, g] < 5.0: 
                matches.append((p, g))
                
        return matches