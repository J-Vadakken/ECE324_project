# ECE324_Project_Code

A collaboration between Joel Vadakken, Lucas Choi, and Zakariyya Brewster!

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

To estimate the position of football players from a camera image.

## Proposed Pipeline
### 1. Run yolo-pose to find 14 keypoints on the field

**Right Side** 
- KP 0: Top Corner of the Penalty Box 
- KP 1: Bottom Corner of the Penalty Box
- KP 2: Top Corner of the Goal Area 
- KP 3: Bottom Corner of the Goal Area
- KP 4: Top-Right Corner Flag
- KP 5: Bottom-Right Corner Flag

**Midfield**
- KP 6: Top Midfield Point (where the halfway line meets the top sideline)
- KP 7: Bottom Midfield Point (where the halfway line meets the bottom sideline)

**Left Side**
- KP 8: Top Corner of the Penalty Box
- KP 9: Bottom Corner of the Penalty Box
- KP 10: Top Corner of the Goal Area
- KP 11: Bottom Corner of the Goal Area
- KP 12: Top-Left Corner Flag
- KP 13: Bottom-Left Corner Flag

### 2. Use keypoints to find the homography matrix (projective transformation between two planes)
### 3. Run YOLO object detection model to find players on the field
### 4. Project player locations onto a 2D pitch using homography matrix
Additional: Kalman filter for sequential (video) data, team matching by jersey colour 
### 5. Baseline comparison of finetuned vs. zero-shot, lightweight models
### 6. High-level tactical analysis from projected player locations in web app demo

Potential Applications of Position Tracking:
- AI Coach: Provide automated tactical feedback on positioning, spacing, and formation discipline during training or match review.
- Performance Analytics: Generate quantitative metrics such as heatmaps, distance covered, and player involvement.
- Tactical Visualization: Create visual overlays like formations, team shape, and pitch control regions to help analyze team structure.
- Scouting Tools: Evaluate player positioning tendencies and spatial awareness for recruitment or talent identification.

## Model Architecture
<img width="1920" height="1080" alt="model_arch_324" src="https://github.com/user-attachments/assets/6da6ff93-eb3a-4ac1-8a7b-c36f8bdccd51" />

## Project Organization

```
├── LICENSE
├── Makefile
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── SoccerNet
│   │   ├── calibration-2023    <- Raw keypoint dataset (JSON/JPG)
│   │   └── SpiideoSynLoc       <- Raw player localization dataset (JSON/JPG)
│   └── processed
│       ├── yolo-calibration    <- Formatted for YOLO Pose (Images + 14-KP labels)
│       └── yolo-synloc         <- Formatted for YOLO Detect (Symlinked images + labels)
│
├── models
│   └── runs                    <- YOLO training outputs (weights, plots, results.csv)
│       ├── calibration         <- Best pitch geometry weights (best.pt)
│       └── synloc_detection    <- Best player detection weights (best.pt)
│
├── notebooks           <- Experimental discovery and EDA
│
├── pyproject.toml
├── reports
│   └── figures         <- Training loss curves and 2D pitch projection graphics
│
├── requirements.txt
├── ECE324_Project      <- Source code
    │
    ├── __init__.py
    ├── config.py       <- Project paths and global constants (PROJ_ROOT, etc.)
    │
    ├── dataset         <- Data preparation logic
    │   ├── __init__.py
    │   ├── prep_calibration.py <- Converts SoccerNet to 14-keypoint YOLO format
    │   └── prep_synloc.py      <- Converts SynLoc COCO to YOLO detect format
    │
    ├── core            <- The mathematical "Brain" of the project
    │   ├── __init__.py
    │   └── geometry.py         <- Homography (H) calculation and RANSAC filtering
    │
    ├── training        <- Training execution scripts
    │   ├── __init__.py
    │   ├── train_calibration.py <- Kick off YOLOv8-Pose training
    │   └── train_synloc.py      <- Kick off YOLOv8-Detect training
    │
    └── visualization
        ├── __init__.py
        └── pitch_mapping.py    <- Generates the top-down 2D mini-map
```

--------

