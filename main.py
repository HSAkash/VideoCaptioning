from src import logger
from src.pipeline.stage_01_download_dataset import DownloadDatasetPipeline
from src.pipeline.stage_02_unzip_dataset import UnzipDatasetPipeline
from src.pipeline.stage_03_ImageExtraction import ImageExtractionPipeline

if __name__ == "__main__":
    
    # Download dataset
    STAGE_NAME = "Download Dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = DownloadDatasetPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    # Unzip dataset
    STAGE_NAME = "Unzip dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = UnzipDatasetPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    # Image Extraction
    STAGE_NAME = "Image Extraction"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = ImageExtractionPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")