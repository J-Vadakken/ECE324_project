import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger, CALIB_CONFIG_PATH

def train_pose_model():
    logger.info("Initializing YOLOv8 Nano Pose Model...")
    # Load a pre-trained Nano model (lightweight and fast)
    model = YOLO('yolov8n-pose.pt')

    # Define where the training logs and weights will be saved
    save_dir = PROJ_ROOT / 'models' / 'runs'
    
    logger.info("Starting training loop... (YOLO will handle the tqdm bars)")
    
    # Run the training
    # The API natively handles the epoch loop and progress bars
    model.train(
        data=CALIB_CONFIG_PATH, 
        epochs=50,
        imgsz=960,
        device='mps',             # Hardware acceleration for Apple Silicon
        project=str(save_dir),
        name='calibration',
        exist_ok=True             # Overwrites the folder if you run it twice
    )
    
    logger.info("Training complete! Generating publication-ready loss curves...")
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
    
    # Plot 1: Box Loss (How well it guesses the bounding box around the visible pitch)
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Bounding Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Pose Loss (How accurately it places the 14 keypoints)
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train/pose_loss'], label='Train Pose Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/pose_loss'], label='Val Pose Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Keypoint (Pose) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot inside the run directory
    plot_path = os.path.join(run_dir, 'custom_loss_plot.png')
    plt.savefig(plot_path, dpi=300) # High DPI for paper figures
    logger.info(f"Saved custom loss plot to {plot_path}")
    
    # Display the plot on your screen
    plt.show()

if __name__ == '__main__':
    # Ensure you run this from the project root: python train_pose.py
    train_pose_model()