import os
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger, CALIB_SYNLOC_CONFIG_PATH

def finetune_synloc():
    # 1. PATHS
    # The Pose model trained on the SoccerNet/FIFA broadcast dataset
    base_model_path = PROJ_ROOT / "models" / "runs" / "calibration" / "weights" / "best.pt"
    yaml_path = CALIB_SYNLOC_CONFIG_PATH
    
    if not base_model_path.exists():
        logger.error(f"❌ Base model not found at {base_model_path}")
        return

    # 2. LOAD MODEL
    # model = YOLO(str(base_model_path))
    model = YOLO('yolov8n-pose.pt')

    # 3. DOMAIN ADAPTATION SETTINGS (The 150-Image Strategy)
    model.train(
        data=str(CALIB_SYNLOC_CONFIG_PATH),
        epochs=500,
        imgsz=1920,           # Native resolution for surgical precision
        rect=True,            # CRITICAL: Saves VRAM by not padding to a square
        batch=8,              # Lowered to prevent OOM on 32GB Mac
        device='mps',
        
        # --- THE "MEMORIZE" LOCK ---
        fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0,
        degrees=0.0, translate=0.0, scale=0.0, perspective=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        
        # --- THE "SNAP" WEIGHTS ---
        pose=35.0,            # Bumped slightly to prioritize the 1920px precision
        kobj=5.0,             
        box=2.0,              # Lowered since it already found the box
        
        lr0=0.001,
        warmup_epochs=0,
        patience=0,           
        project=str(PROJ_ROOT / "models/runs"),
        name="calibration_synloc"
    )

    logger.info("🎉 Fine-tuning complete! Check 'calibration_synloc' for results.")

if __name__ == "__main__":
    finetune_synloc()