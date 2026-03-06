import cv2
import torch
import numpy as np
from ultralytics import YOLO
from loguru import logger
from ECE324_Project.location_engine import LocationEngine
from ECE324_Project.dataset import SoccerTrackDataset

class SoccerDetector:
    def __init__(self, model_variant="yolov8n.pt", match_id="117092"):
        # Load the model (n = nano is fastest for testing)
        self.model = YOLO(model_variant)
        self.engine = LocationEngine(match_id)
        
        # We only care about the 'person' class (COCO class 0)
        self.target_class = 0 
        
    def process_frame(self, frame):
        # Get original frame dimensions
        h_orig, w_orig = frame.shape[:2]
        
        # Run inference - imgsz=1280 is for the internal scaling
        results = self.model(frame, imgsz=1280, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == self.target_class:
                # Force conversion to standard floats
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                # Calculate bottom-center (feet)
                u = (x1 + x2) / 2.0
                v = y2
                
                # Project directly using absolute pixels -> Decimeters -> Meters
                x_m, y_m = self.engine.pixel_to_meters(u, v)
                
                detections.append({
                    "pitch_coords": (x_m, y_m),
                    "conf": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2]
                })
        
        return detections
    
if __name__ == "__main__":
    detector = SoccerDetector()
    ds = SoccerTrackDataset("117092")
    engine = LocationEngine("117092")
    
    # FIND A FRAME WITH GUARANTEED ACTION
    logger.info("🔍 Searching for a frame with at least 10 players...")
    frame_idx = 1
    frame, gt_labels = None, []
    
    for i in range(1, 5000, 50): # Scan every 50 frames
        f, labels = ds.get_frame_and_labels(i)
        gt_players = [a for a in labels if a.get('category_id') == 1]
        if len(gt_players) >= 10:
            frame_idx = i
            frame = f
            gt_labels = labels
            logger.success(f"Found action at Frame {frame_idx}! ({len(gt_players)} GT players)")
            break
            
    if frame is None:
        logger.error("Could not find any players in the first 5000 frames.")
        exit()
    
    logger.info("🤖 Running YOLO inference...")
    yolo_results = detector.process_frame(frame)
    
    # ==========================================
    # RADAR MAP GENERATION
    # ==========================================
    SCALE = 10  
    W_M, H_M = 105.0, 68.0  
    
    img_w, img_h = int(W_M * SCALE), int(H_M * SCALE)
    radar = np.full((img_h, img_w, 3), (60, 150, 60), dtype=np.uint8)
    
    center_x, center_y = img_w // 2, img_h // 2
    cv2.rectangle(radar, (0, 0), (img_w-1, img_h-1), (255, 255, 255), 2)  
    cv2.line(radar, (center_x, 0), (center_x, img_h), (255, 255, 255), 2) 
    cv2.circle(radar, (center_x, center_y), int(9.15 * SCALE), (255, 255, 255), 2) 
    
    def m_to_px(x_meters, y_meters):
        px = int((x_meters * SCALE) + center_x)
        py = int((y_meters * SCALE) + center_y)
        return px, py

    # PLOT GROUND TRUTH (BLUE)
    gt_players = [a for a in gt_labels if a.get('category_id') == 1]
    plotted_gt = 0
    for p in gt_players:
        bbox = p.get('bbox_image')
        if not bbox: continue
        
        # Engine takes absolute pixels naturally now
        gt_x, gt_y = engine.pixel_to_meters(bbox['x_center'], bbox['y'] + bbox['h'])
        px, py = m_to_px(gt_x, gt_y)
        
        # Check if coordinates are inside the image bounds
        if 0 <= px < img_w and 0 <= py < img_h:
            cv2.circle(radar, (px, py), 12, (255, 0, 0), -1) # Large Blue dot
            plotted_gt += 1
        else:
            logger.warning(f"GT Player out of bounds: {gt_x:.1f}m, {gt_y:.1f}m -> px:{px}, py:{py}")
            
    # PLOT YOLO PREDICTIONS (RED)
    plotted_yolo = 0
    for det in yolo_results:
        yolo_x, yolo_y = det['pitch_coords']
        px, py = m_to_px(yolo_x, yolo_y)
        
        if 0 <= px < img_w and 0 <= py < img_h:
            cv2.circle(radar, (px, py), 10, (0, 0, 255), -1) # Red dot
            cv2.circle(radar, (px, py), 10, (255, 255, 255), 2) # White outline
            plotted_yolo += 1

    logger.success(f"Plotted {plotted_gt}/{len(gt_players)} GT and {plotted_yolo}/{len(yolo_results)} YOLO players.")
    
    out_name = "tactical_radar_SUCCESS.jpg"
    cv2.imwrite(out_name, radar)
    logger.info(f"Saved radar map to '{out_name}'")