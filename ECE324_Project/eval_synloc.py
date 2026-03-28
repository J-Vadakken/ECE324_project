import torch
import numpy as np
import os
import time
from pathlib import Path
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ECE324_Project.config import PROJ_ROOT, logger

def get_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

def calculate_ap(precisions, recalls):
    """Calculates Average Precision using all-points interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def evaluate_at_threshold(model, img_files, lbl_dir, conf_thresh, iou_thresh=0.5):
    all_scores, all_matches = [], []
    total_gt = 0
    total_preds = 0

    for img_path in tqdm(img_files, desc=f"Evaluating @ conf={conf_thresh}", leave=False):
        # 1. Load GT
        gt_boxes = []
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    _, cx, cy, bw, bh = parts[:5]
                    gt_boxes.append([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])
        total_gt += len(gt_boxes)

        # 2. Predict
        results = model.predict(img_path, conf=conf_thresh, imgsz=960, verbose=False)[0]
        h, w = results.orig_shape
        preds = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            preds.append({'box': [b[0]/w, b[1]/h, b[2]/w, b[3]/h], 'score': float(box.conf[0])})
        
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        total_preds += len(preds)

        # 3. Hungarian Match (Strict 1-to-1)
        if len(preds) > 0 and len(gt_boxes) > 0:
            iou_matrix = np.zeros((len(preds), len(gt_boxes)))
            for p_idx, p in enumerate(preds):
                for g_idx, g in enumerate(gt_boxes):
                    iou_matrix[p_idx, g_idx] = get_iou(p['box'], g)
            
            # Use linear_sum_assignment to find optimal global matching
            p_indices, g_indices = linear_sum_assignment(-iou_matrix) # Negative for maximization
            
            matched_preds = set()
            for p_idx, g_idx in zip(p_indices, g_indices):
                if iou_matrix[p_idx, g_idx] >= iou_thresh:
                    all_scores.append(preds[p_idx]['score'])
                    all_matches.append(1) # True Positive
                    matched_preds.add(p_idx)
            
            # Remaining preds are False Positives
            for p_idx in range(len(preds)):
                if p_idx not in matched_preds:
                    all_scores.append(preds[p_idx]['score'])
                    all_matches.append(0)
        else:
            # All preds are FPs if no GT exists
            for p in preds:
                all_scores.append(p['score'])
                all_matches.append(0)

    # Calculate PR Curve
    all_scores, all_matches = np.array(all_scores), np.array(all_matches)
    indices = np.argsort(-all_scores)
    tp_cumsum = np.cumsum(all_matches[indices])
    fp_cumsum = np.cumsum(1 - all_matches[indices])
    
    recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if len(tp_cumsum) > 0 else np.zeros_like(tp_cumsum)
    
    ap = calculate_ap(precisions, recalls)
    f1 = 2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]) if (precisions[-1] + recalls[-1]) > 0 else 0

    return {
        "ap": ap, "precision": precisions[-1] if len(precisions) > 0 else 0,
        "recall": recalls[-1] if len(recalls) > 0 else 0,
        "f1": f1, "count": total_preds, "gt_total": total_gt
    }

if __name__ == "__main__":
    MODEL_P = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"
    IMG_D = PROJ_ROOT / "data/processed/yolo-synloc-10k/images/val"
    LBL_D = PROJ_ROOT / "data/processed/yolo-synloc-10k/labels/val"
    
    model = YOLO(MODEL_P)
    img_files = sorted(list(IMG_D.glob("*.jpg")))

    # --- REPORT 1: RESEARCH MODE ---
    res_research = evaluate_at_threshold(model, img_files, LBL_D, conf_thresh=0.001)

    # --- REPORT 2: OPERATIONAL MODE ---
    res_op = evaluate_at_threshold(model, img_files, LBL_D, conf_thresh=0.40)

    print("\n" + "="*55)
    print(f"{'Metric':<20} | {'Research (0.001)':<18} | {'Operational (0.40)':<18}")
    print("-" * 55)
    print(f"{'mAP@50':<20} | {res_research['ap']:>18.4f} | {'N/A (Truncated)':>18}")
    print(f"{'Precision':<20} | {res_research['precision']:>18.4f} | {res_op['precision']:>18.4f}")
    print(f"{'Recall':<20} | {res_research['recall']:>18.4f} | {res_op['recall']:>18.4f}")
    print(f"{'F1-Score':<20} | {res_research['f1']:>18.4f} | {res_op['f1']:>18.4f}")
    print(f"{'Pred Count':<20} | {res_research['count']:>18} | {res_op['count']:>18}")
    print(f"{'Ground Truth':<20} | {res_research['gt_total']:>18} | {res_op['gt_total']:>18}")
    print("="*55)