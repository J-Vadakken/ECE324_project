import torch
import time
import os
import numpy as np
from ultralytics import YOLO
from ECE324_Project.config import PROJ_ROOT, logger

def get_model_size_mb(model_path):
    """Calculates the physical disk size of the model."""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)
def profile_model(model_path, img_size=640, iterations=100, device='None'):
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    logger.info(f"Profiling {model_path.name} on {device}...")
    model = YOLO(str(model_path))
    model.to(device)

    # --- FIX: Manual Parameter and FLOPs Calculation ---
    # Parameters in Millions
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    # GFLOPs (theoretical)
    # We access the 'task' info which YOLO stores after the first dummy pass
    dummy_input = torch.zeros((1, 3, img_size, img_size)).to(device)
    
    # Capture the internal FLOPs estimate
    try:
        # This is the internal YOLOv8/v11 profiler
        from ultralytics.utils.ops import get_flops
        flops = get_flops(model.model, imgsz=img_size)
    except:
        flops = 0.0 # Fallback if internal API shifts

    # --- LATENCY BENCHMARK ---
    # Warm-up
    for _ in range(10):
        _ = model(dummy_input, verbose=False)
    
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model(dummy_input, verbose=False)
        latencies.append(time.perf_counter() - start)
    
    avg_latency = np.mean(latencies) * 1000
    fps = 1000 / avg_latency
    
    return {
        "Model Name": model_path.name,
        "Device": device,
        "Params (M)": params,
        "GFLOPs": flops,
        "Disk Size (MB)": os.path.getsize(model_path) / (1024*1024),
        "Avg Latency (ms)": avg_latency,
        "Throughput (FPS)": fps
    }

def print_metrics_table(results):
    """Prints a clean, formatted table for easy copy-pasting into your report."""
    header = f"{'Metric':<25} | {'Value':<15}"
    print("\n" + "="*45)
    print(f"REPORT: {results['Model Name']}")
    print("-" * 45)
    for key, val in results.items():
        if key == "Model Name": continue
        if isinstance(val, float):
            print(f"{key:<25} | {val:>12.3f}")
        else:
            print(f"{key:<25} | {val:>12}")
    print("="*45 + "\n")

if __name__ == "__main__":
    # Define your model paths
    PITCH_MODEL = PROJ_ROOT / "models/runs/synloc_pixel_refinement_1920/weights/best.pt"
    PLAYER_MODEL = PROJ_ROOT / "models/runs/synloc_50/weights/best.pt"
    
    models_to_test = [PITCH_MODEL, PLAYER_MODEL]
    
    all_results = []
    for m_path in models_to_test:
        if m_path.exists():
            res = profile_model(m_path, img_size=640)
            print_metrics_table(res)
            all_results.append(res)
        else:
            logger.warning(f"Model not found at {m_path}")