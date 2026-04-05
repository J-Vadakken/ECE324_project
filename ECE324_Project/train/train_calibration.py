import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger, CALIB_CONFIG_PATH

def train_calibration_model():
    logger.info("Initializing YOLOv8 Nano Pose Model...")
    model = YOLO('yolov8n-pose.pt')

    # Define where the training logs and weights will be saved
    save_dir = PROJ_ROOT / 'models' / 'runs'
    checkpoint_path = save_dir / 'calibration' / 'weights' / 'last.pt'


    if checkpoint_path.exists():
        logger.info(f"Found existing checkpoint at {checkpoint_path}. Resuming training...")
        model = YOLO(str(checkpoint_path))
        resume_flag = True
    else:
        logger.info("No checkpoint found. Initializing new model...")       
        resume_flag = False
    
    logger.info("Starting training loop... ")
    
    # Run the training
    model.train(
        data=CALIB_CONFIG_PATH,
        epochs=50,
        imgsz=960,
        device='mps',
        batch=32,            
        workers=8,           
        cache=True,          
        project=str(save_dir),
        name='calibration',
        exist_ok=True,
        resume=resume_flag
    )
    
    logger.info("Training complete! Generating loss curves...")
    plot_training_results(str(save_dir / 'calibration'))

def plot_training_results(run_dir):
    """
    Reads the YOLO results.csv and generates a clean matplotlib figure.
    """
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        logger.error(f"Could not find results.csv in {run_dir}")
        return
        
    # YOLO formats column names with annoying leading spaces, so we strip them
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Create a nice wide figure for the paper
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Box Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Bounding Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Keypoint Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train/kp_loss'], label='Train Keypoint Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/kp_loss'], label='Val Keypoint Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Keypoint Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot inside the run directory
    plot_path = os.path.join(run_dir, 'calibration_loss.png')
    plt.savefig(plot_path, dpi=300) # High DPI for paper figures
    logger.info(f"Saved custom loss plot to {plot_path}")
    
    # plt.show()

if __name__ == '__main__':
    # Ensure you run this from the project root: python train_calibration.py
    train_calibration_model()