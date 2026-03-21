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
        epochs=100,           # 40 images train very fast (seconds per epoch)
        imgsz=640,
        device='mps',         # Use your M2 Max GPU
        batch=8,              # Small batch for a small dataset
        lr0=0.001,            # Start with a 10x smaller learning rate than default
        lrf=0.01,             # Final learning rate factor
        augment=True,
        mosaic=0.0,           # Disable mosaic for precise keypoint alignment
        close_mosaic=0,
        workers=0,            # macOS stability
        amp=False,            # macOS stability
        project=str(PROJ_ROOT / "models" / "runs"),
        name="calibration_finetuned",
        exist_ok=True
    )

    logger.info("🎉 Fine-tuning complete! Check 'calibration_finetuned' for results.")

if __name__ == "__main__":
    finetune_manual()