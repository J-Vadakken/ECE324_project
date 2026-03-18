# ECE324_Project_Code

A collaboration between Joel Vadakken, Lucas Choi, and Zakariyya Brewster!

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

To estimate the position of football players from a camera image.

## Proposed Pipeline
1. Run yolo-pose to find 14 keypoints on the field
    - Corners of penalty box
    - Corners of goal area
    - Corner flags
    - Midfield points (halfway line meets sideline)
2. Use keypoints to find the homography matrix (projective transformation between two planes)
3. Run YOLO object detection model to find players on the field
4. Project player locations onto a 2D pitch using homography matrix
   - Additional: Kalman filter for sequential (video) data, team matching by jersey colour 
5. Baseline comparison of finetuned vs. zero-shot, lightweight models
6. High-level tactical analysis from projected player locations in web app demo

Potential Applications of Position Tracking:
- AI Coach: Provide automated tactical feedback on positioning, spacing, and formation discipline during training or match review.
- Performance Analytics: Generate quantitative metrics such as heatmaps, distance covered, and player involvement.
- Tactical Visualization: Create visual overlays like formations, team shape, and pitch control regions to help analyze team structure.
- Scouting Tools: Evaluate player positioning tendencies and spatial awareness for recruitment or talent identification.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ECE324_Project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ECE324_Project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ECE324_Project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

