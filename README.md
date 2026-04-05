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
### 3. Run YOLO object detection model to find players on the field, team matching by jersey colour.
### 4. Project player locations onto a 2D pitch using homography matrix
Future: Kalman filter for sequential (video) data
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
├── README.md               <- Top-level README
├── environment.yml         <- Conda environment
├── pyproject.toml
├── yolov8n.pt              <- Pretrained YOLOv8-Detect base weights
├── yolov8n-pose.pt         <- Pretrained YOLOv8-Pose base weights
│
├── data                    <- Not tracked by git; populate via dataset_download.py
│   ├── SoccerNet
│   │   ├── calibration-2023    <- Raw pitch keypoint dataset
│   │   └── SpiideoSynLoc       <- Raw player localization dataset
│   └── processed
│       ├── yolo-calibration        <- Manually annotated SpiideoSynLoc images (14 KP)
│       ├── yolo-calibration-2023   <- Alternative calibration dataset
│       ├── yolo-synloc             <- YOLO Detect format (full dataset)
│       └── yolo-synloc-10k         <- YOLO Detect format (10k-image subset)
│
├── models
│   └── runs                    <- YOLO training weights
│       ├── calibration                     <- Pitch keypoint model (best.pt), not used in pipeline
│       ├── synloc_50                       <- Player detection (50-epoch run, best.pt)
│       └── calibration_synloc              <- Pitch keypoint model from manual annotations
│
├── references              <- Papers and external references
│
├── docs                    <- MkDocs project documentation
│
└── ECE324_Project          <- Source code
    ├── __init__.py
    ├── config.py           <- Project paths and global constants (PROJ_ROOT, etc.)
    ├── pipeline.py         <- End-to-end inference: keypoints → homography → player map
    ├── viz_team.py         <- Team classification by jersey colour visualization 
    │
    ├── configs             <- YOLO dataset YAML configs
    │   ├── calibration.yaml
    │   ├── calibration_synloc.yaml
    │   └── synloc.yaml
    │
    ├── dataset             <- Data preparation scripts
    │   ├── dataset_download.py     <- Downloads SoccerNet datasets via API
    │   ├── prep_calibration.py     <- Converts calibration-2023 to YOLO Pose format with 14 KP
    │   ├── prep_synloc.py          <- Converts SpiideoSynLoc COCO to YOLO Detect
    │   ├── synloc_10k.py           <- Samples a 10k-image subset from SynLoc
    │   ├── synloc_to_calib.py      <- Script to manually annotate Synloc images for calibration (14 KP)
    │   ├── edit_anno.py            <- Manual annotation editing helpers
    │   ├── sync_manual.py          <- Syncs manually annotated samples into dataset
    │   └── verify_synloc_calib.py  <- Visual verification of manually annotated image alignment
    │
    ├── train               <- Training and hyperparameter tuning scripts
    │   ├── train_calibration.py    <- Trains YOLOv8-Pose for pitch keypoints
    │   ├── train_synloc.py         <- Trains YOLOv8-Detect for player detection
    │   └── tune_calibration.py     <- Hyperparameter search for calibration model
    │
    └── eval                <- Evaluation scripts
        ├── eval_calibration.py         <- Evaluates keypoint model on calibration set
        ├── eval_synloc.py              <- Evaluates detection model on SynLoc set
        ├── eval_pipeline.py            <- End-to-end pipeline evaluation
        ├── eval_calibration_on_synloc.py <- Cross-dataset: calib model on SynLoc
        ├── eval_synloc_on_calib.py     <- Cross-dataset: detection model on calib set
        └── model_sizes.py              <- Reports model parameter counts and latency
```

--------

