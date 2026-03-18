from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ==========================================
# SOCCERNET SYNLOC DATASET PATHS
# ==========================================
# Points to: /Users/lucaschoi/Documents/GitHub/ECE324_project/data/SoccerNet/SpiideoSynLoc
SYNLOC_DIR = DATA_DIR / "SoccerNet" / "SpiideoSynLoc"
SYNLOC_ANNO_DIR = SYNLOC_DIR / "annotations"
SYNLOC_IMG_DIR = SYNLOC_DIR / "train"

# YOLO Training Data Output (Where we will put the .txt labels later)
YOLO_DATA_DIR = PROCESSED_DATA_DIR / "yolo-synloc"
SYNLOC_CONFIG_PATH = PROJ_ROOT / "ECE324_Project" / "configs" / "synloc.yaml"
# ==========================================

# ==========================================
# SOCCERNET CALIBRATION DATASET PATHS
# ==========================================
CALIB_DIR = DATA_DIR / "SoccerNet" / "calibration-2023"

# YOLO Pose Training Data Output (Where we put the 14-keypoint labels and symlinked images)
YOLO_CALIB_DIR = PROCESSED_DATA_DIR / "yolo-calibration-2023"
CALIB_CONFIG_PATH = PROJ_ROOT / "ECE324_Project" / "configs" / "calibration.yaml"
# ==========================================

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass