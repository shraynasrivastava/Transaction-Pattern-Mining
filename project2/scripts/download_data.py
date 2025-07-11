import os
import sys
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def download_dataset():
    """Download the credit card fraud dataset from Kaggle."""
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Set download path
        download_path = project_root / "data" / "raw"
        download_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading dataset from Kaggle...")
        
        # Download the dataset
        api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(download_path),
            unzip=True
        )
        
        logger.info(f"Dataset downloaded successfully to {download_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        logger.error("\nPlease ensure you have:")
        logger.error("1. Installed kaggle package: pip install kaggle")
        logger.error("2. Created a Kaggle account")
        logger.error("3. Generated an API token from https://www.kaggle.com/account")
        logger.error("4. Placed kaggle.json in ~/.kaggle/ directory")
        return False

if __name__ == "__main__":
    download_dataset() 