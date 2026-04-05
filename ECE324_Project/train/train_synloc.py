import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger, SYNLOC_CONFIG_PATH

def plot_training_results(run_dir):
    """
    Reads the YOLO results.csv and generates a clean matplotlib figure.
    Adapted for Object Detection (Box & Class Loss).
    """
    csv_path = Path(run_dir) / 'results.csv'
    if not csv_path.exists():
        logger.error(f"Could not find results.csv in {run_dir}")
        return
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Box Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Bounding Box Loss (Spatial Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Class Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Classification Loss (Player Confidence)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path(run_dir) / 'detection_loss_curves.png'
    plt.savefig(plot_path, dpi=300) 
    logger.info(f"✅ Saved custom loss plot for your report to {plot_path}")

def start_training():
    save_dir = PROJ_ROOT / "models" / "runs"
    run_name = "synloc_50"
    
    # Define the path where the model would have failed/saved last.pt
    # (Adjusted to look in the run_name directory you are saving to)
    last_weights_pth = save_dir / run_name / "weights" / "last.pt"

    # Dynamic Resume Logic
    if last_weights_pth.exists():
        logger.info(f"🔄 Crash detected. Resuming from {last_weights_pth}...")
        model = YOLO(str(last_weights_pth))
        is_resume = True
    else:
        logger.info("🆕 No previous weights found. Starting fresh from yolov8n.pt...")
        model = YOLO("yolov8n.pt")  
        is_resume = False

    yaml_path = SYNLOC_CONFIG_PATH
    
    logger.info("🚀 Starting optimized M2 training run...")
    
    # The M2-Optimized Training Block
    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=960,
        device='mps',      
        batch=16,          
        workers=0,         # MUST BE 0 to prevent macOS multiprocessing locks
        amp=False,         # Disable mixed precision for stability on M1/M2
        cache='ram',       
        project=str(save_dir),
        name=run_name,
        exist_ok=True,     # Must be True so it can write back to the same folder
        resume=is_resume   # <-- Plugs in True or False automatically
    )
    
    logger.info("🎉 Training complete! Generating loss curves...")
    
    # Run the plotting function pointing to the folder YOLO just created
    plot_training_results(save_dir / run_name)

if __name__ == "__main__":
    start_training()