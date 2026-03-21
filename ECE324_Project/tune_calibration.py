import os
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger, CALIB_SYNLOC_CONFIG_PATH

def finetune_manual():
    # 1. PATHS
    # The model you already trained for 15+ epochs
    base_model_path = PROJ_ROOT / "models" / "runs" / "calibration" / "weights" / "best.pt"
    
    # Your new manual calibration YAML (ensure this points to /images/manual)
    yaml_path = CALIB_SYNLOC_CONFIG_PATH
    
    if not base_model_path.exists():
        logger.error(f"❌ Base model not found at {base_model_path}")
        return

    # 2. LOAD MODEL
    model = YOLO(str(base_model_path))

    # 3. FINE-TUNE SETTINGS
    # We use a smaller learning rate (lr0) so we don't 'break' the existing weights.
    # We also disable mosaic augmentation because 40 images is a small set and 
    # we want the model to see the full, clean frame every time.
    model.train(
    data=str(yaml_path),
    epochs=200,           # More epochs to really 'burn in' the coordinates
    imgsz=960,            # Match your inference size
    lr0=0.0001,           # VERY low learning rate (don't break the weights!)
    device='mps',          # Use GPU if available
    degrees=0.0,          
    translate=0.0,        
    scale=0.0,            
    shear=0.0,            
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,           # Keep this 0.0 if the camera never flips
    mosaic=0.0,           
    copy_paste=0.0,
    project=str(PROJ_ROOT / "models" / "runs"),
    name="static_arena_finetune"
)

    logger.info("🎉 Fine-tuning complete! Check 'calibration_finetuned' for results.")

if __name__ == "__main__":
    finetune_manual()