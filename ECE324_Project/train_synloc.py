import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger, SYNLOC_CONFIG_PATH

def train_synloc_model():
    logger.info("Initializing YOLOv8 Nano Detection Model...")
    # Load a pre-trained Nano detection model (not the pose model)
    model = YOLO('yolov8n.pt')

    # Define where the training logs and weights will be saved
    save_dir = PROJ_ROOT / 'models' / 'runs'
    
    logger.info("Starting player detection training loop...")
    
    # Run the training
    model.train(
        data=SYNLOC_CONFIG_PATH,
        epochs=50,
        imgsz=1920,               # Crucial: Keeps the high-res SynLoc players visible
        device='mps',             # Apple Silicon hardware acceleration
        project=str(save_dir),
        name='synloc_detection',
        exist_ok=True             # Overwrites the folder if you restart
    )
    
    logger.info("Training complete! Generating publication-ready loss curves...")
    plot_training_results(str(save_dir / 'synloc_detection'))

def plot_training_results(run_dir):
    """
    Reads the YOLO results.csv and generates a clean matplotlib figure.
    """
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        logger.error(f"Could not find results.csv in {run_dir}")
        return
        
    # YOLO formats column names with leading spaces, so we strip them
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Create a nice wide figure for the paper
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Box Loss (Accuracy of drawing the box around the player)
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Bounding Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Class Loss (Accuracy of identifying the object as 'Player')
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='#2c3e50', linewidth=2)
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot inside the run directory
    plot_path = os.path.join(run_dir, 'custom_detection_plot.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved custom loss plot to {plot_path}")
    
    # Display the plot on your screen
    plt.show()

if __name__ == '__main__':
    train_synloc_model()