# ECE324_Project_Code

A collaboration between Joel Vadakken, Lucas Choi, and Zakariyya Brewster!

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

To estimate the position of football players from a camera image.

Potential Applications of Position Tracking:
- AI Coach: Provide automated tactical feedback on positioning, spacing, and formation discipline during training or match review.
- Performance Analytics: Generate quantitative metrics such as heatmaps, distance covered, and player involvement.
- Tactical Visualization: Create visual overlays like formations, team shape, and pitch control regions to help analyze team structure.
- Scouting Tools: Evaluate player positioning tendencies and spatial awareness for recruitment or talent identification.

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

