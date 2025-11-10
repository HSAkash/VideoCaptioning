from src import logger
from src.pipeline.stage_01_download_dataset import DownloadDatasetPipeline
from src.pipeline.stage_02_unzip_dataset import UnzipDatasetPipeline

if __name__ == "__main__":
    
    # Download dataset
    STAGE_NAME = "Download Dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    Pipeline = DownloadDatasetPipeline()
    Pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    # Unzip dataset
    STAGE_NAME = "Unzip dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    Pipeline = UnzipDatasetPipeline()
    Pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")