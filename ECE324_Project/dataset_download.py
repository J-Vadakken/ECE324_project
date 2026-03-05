import gdown
from loguru import logger
from ECE324_Project.config import RAW_DATA_DIR

def download_soccertrack_v2():
    # Folder ID on Google Drive
    folder_id = '1N2Qx2qkFgRtpbHitl2Vh6sLVYGgqkWwn'
    
    # Ensure raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting download of SoccerTrack-v2 folder to {RAW_DATA_DIR}...")
    
    try:
        gdown.download_folder(
            id=folder_id, 
            output=str(RAW_DATA_DIR), 
            quiet=False, 
            remaining_ok=True
        )
        logger.success("Download complete. Check data/raw for match folders.")
        
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        logger.info("Note: Google Drive may rate-limit large folder downloads. "
                    "If this fails, try downloading individual match ZIPs manually.")

if __name__ == "__main__":
    download_soccertrack_v2()