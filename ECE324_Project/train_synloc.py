import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger, SYNLOC_CONFIG_PATH

def train_synloc_model():
    # Define paths for checkpoint and save directory
    checkpoint_path = PROJ_ROOT / 'models' / 'runs' / 'synloc' / 'weights' / 'last.pt'
    save_dir = PROJ_ROOT / 'models' / 'runs'
    
    if checkpoint_path.exists():
        logger.info(f"Existing SynLoc checkpoint found at {checkpoint_path}. Resuming...")
        model = YOLO(checkpoint_path)
        resume_flag = True
    else:
        logger.info("No checkpoint found. Initializing new Model...")
        model = YOLO('yolov8n.pt')
        resume_flag = False

    logger.info("Starting player detection training loop...")
    
    model.train(
        data=str(SYNLOC_CONFIG_PATH),
        epochs=10,                
        imgsz=1920,               
        device='mps',             
        batch=16,                 
        workers=8,           
        cache=True,
        project=str(save_dir),
        name='synloc',
        exist_ok=True,
        resume=resume_flag,
        box=7.5,
        cls=0.5
    )
    
    logger.info("Training complete! Generating loss curves...")
    plot_training_results(str(save_dir / 'synloc'))

def plot_training_results(run_dir):
    """
    Reads the YOLO results.csv and generates a clean matplotlib figure.
    """
    csv_path = Path(run_dir) / 'results.csv'
    if not csv_path.exists():
        logger.error(f"Could not find results.csv in {run_dir}")
        return
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    plt.figure(figsize=(12, 5))
    
    # Bounding Box Loss
    plt.subplot(1, 2, 1)
    if 'train/box_loss' in df.columns:
        plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#2c3e50', linewidth=2)
        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Bounding Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Classification Loss
    plt.subplot(1, 2, 2)
    if 'train/cls_loss' in df.columns:
        plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='#2c3e50', linewidth=2)
        plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(run_dir) / 'custom_detection_plot.png'
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved custom loss plot to {plot_path}")
    plt.show()

if __name__ == '__main__':
    train_synloc_model()