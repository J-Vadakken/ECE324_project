import json
from ECE324_Project.config import PROJ_ROOT

def discover_pitch_dims(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    xs, ys = [], []
    for ann in data['annotations']:
        pos = ann.get('position_on_pitch')
        if pos:
            xs.append(pos[0])
            ys.append(pos[1])
    
    print(f"--- Coordinate Analysis for {json_path.name} ---")
    print(f"JSON X Range: {min(xs):.2f} to {max(xs):.2f} (Total: {max(xs)-min(xs):.2f}m)")
    print(f"JSON Y Range: {min(ys):.2f} to {max(ys):.2f} (Total: {max(ys)-min(ys):.2f}m)")
    
    # Heuristic: The larger span is the length, smaller is the width
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    
    if span_x < span_y:
        print("\nConclusion: JSON X is WIDTH, JSON Y is LENGTH")
    else:
        print("\nConclusion: JSON X is LENGTH, JSON Y is WIDTH")

if __name__ == "__main__":
    JSON_P = PROJ_ROOT / "data/SoccerNet/SpiideoSynLoc/annotations/test.json"
    discover_pitch_dims(JSON_P)