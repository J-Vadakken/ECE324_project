import json
from pathlib import Path
from ultralytics import YOLO

# Central config imports
from ECE324_Project.config import SYNLOC_ANNO_DIR, SYNLOC_IMG_DIR, FIGURES_DIR, logger

def run_basic_yolo(split="train", frame_index=0):
    # Load the pre-trained YOLOv8 Nano model 
    logger.info("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt") 

    # Find an image from the dataset
    json_path = SYNLOC_ANNO_DIR / f"{split}.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_info = data['images'][frame_index]
    img_path = SYNLOC_IMG_DIR / img_info['file_name']
    
    if not img_path.exists():
        logger.error(f"Could not find image at {img_path}")
        return

    logger.info(f"Running inference on {img_path.name}...")

    # Run Inference
    results = model(img_path, imgsz=1920,)

    # Save the result
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"yolo_basic_inference_{frame_index}.jpg"
    
    # The results object has a built-in save method that plots the boxes
    results[0].save(filename=str(out_path))
    
    logger.info(f"Success! Check out the predictions here: {out_path}")

if __name__ == "__main__":
    run_basic_yolo("train", 0)