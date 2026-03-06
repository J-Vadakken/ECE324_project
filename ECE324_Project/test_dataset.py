from ECE324_Project.dataset import SoccerTrackDataset
import cv2

def debug_data_sync(match_id="117092"):
    ds = SoccerTrackDataset(match_id)
    
    # Check middle of the match
    frame, labels = ds.get_frame_and_labels(1000)
    
    if frame is not None:
        print(f"Success! Frame Shape: {frame.shape}")
        print(f"Labels found for frame 1000: {len(labels)} players")
        
        if labels:
            p = labels[0]
            print(f"Sample Player Location: ID {p['id']} at ({p['x']:.2f}m, {p['y']:.2f}m)")
    
    ds.close()

if __name__ == "__main__":
    debug_data_sync()