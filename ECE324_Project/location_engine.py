import numpy as np
import cv2
import xml.etree.ElementTree as ET
from loguru import logger

# Import your custom paths from config
from ECE324_Project.config import RAW_CALIB_DIR

class LocationEngine:
    def __init__(self, match_id):
        self.match_id = match_id
        self.root_path = RAW_CALIB_DIR / match_id
        
        if not self.root_path.exists():
            logger.error(f"Directory not found: {self.root_path}")
            raise FileNotFoundError(f"Check your data path at {self.root_path}")
        self._load_metadata()
        
        # Load Camera Calibration (Fisheye Params)
        intrinsics_path = self.root_path / f"{match_id}_camera_intrinsics.npz"
        intrinsics = np.load(intrinsics_path)
        self.K = intrinsics['K']
        self.D = intrinsics['D']
        
        # Load Homography (Pitch <-> Image)
        h_path = self.root_path / f"{match_id}_homography.npy"
        self.H = np.load(h_path)
        self.H_inv = np.linalg.inv(self.H)
        
        logger.info(f"LocationEngine successfully initialized for {match_id}")

    def _load_metadata(self):
        xml_path = self.root_path / f"{self.match_id}_tracker_box_metadata.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        pitch_el = root.find("pitch")
        
        # Pull dimensions directly from XML 
        self.pitch_width = float(pitch_el.get("width"))
        self.pitch_height = float(pitch_el.get("height"))
        
        # Origin Offset: Maps Top-Left (0,0) to Center-Circle (0,0)
        self.origin_x = self.pitch_width / 2.0
        self.origin_y = self.pitch_height / 2.0

    def pitch_to_pixel(self, x_m, y_m):
        """Maps 2D Pitch Meters (GSR) to Raw Image Pixels."""
        # Convert centered meters back to Raw Matrix Units (Decimeters)
        x_raw = (x_m + self.origin_x) * 10.0
        y_raw = (y_m + self.origin_y) * 10.0
        
        # Project using Homography
        pitch_pt = np.array([x_raw, y_raw, 1.0])
        img_pt = self.H @ pitch_pt
        
        u = img_pt[0] / img_pt[2]
        v = img_pt[1] / img_pt[2]
        
        # Apply Fisheye Distortions to match raw panorama
        points = np.array([[[u, v]]], dtype=np.float32)
        distorted_pt = cv2.fisheye.distortPoints(points, self.K, self.D)
        
        return distorted_pt[0][0].astype(int)

    def pixel_to_pitch(self, u, v):
        """Maps Raw Image Pixels (YOLO) to 2D Pitch Meters."""
        # Undistort Pixel first
        points = np.array([[[u, v]]], dtype=np.float32)
        undistorted_pt = cv2.fisheye.undistortPoints(points, self.K, self.D, P=self.K)
        u_flat, v_flat = undistorted_pt[0][0]
        
        # Map to Pitch Plane using Inverse Homography
        pixel_pt = np.array([u_flat, v_flat, 1.0])
        projected = self.H_inv @ pixel_pt
        
        w = projected[2]
        x_raw = projected[0] / w
        y_raw = projected[1] / w
        
        # Convert back to centered meters
        x_m = (x_raw / 10.0) - self.origin_x
        y_m = (y_raw / 10.0) - self.origin_y
        
        return round(x_m, 2), round(y_m, 2)