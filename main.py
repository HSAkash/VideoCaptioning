from src import logger
from src.pipeline.stage_01_download_dataset import DownloadDatasetPipeline
from src.pipeline.stage_02_unzip_dataset import UnzipDatasetPipeline
from src.pipeline.stage_03_ImageExtraction import ImageExtractionPipeline
from src.pipeline.stage_04_augmentation import AugmentationPipeline
from src.pipeline.stage_05_videoEncoding import VideoEncodingPipeline
from src.pipeline.stage_06_generateDatasetLabel import GenerateDatasetLabelPipeline
from src.pipeline.stage_07_training import TrainingPipeline
from src.pipeline.stage_08_generateCaption import GenerateCaptionPipeline
from src.pipeline.stage_09_refineGeneratedCaption import RefineGeneratedCaptionPipeline
from src.pipeline.stage_10_evaluation import EvaluationPipeline
from src.pipeline.stage_11_plotHistory import PlotHistoryPipeline

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

    # Augmentation
    STAGE_NAME = "Augmentation"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = AugmentationPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    STAGE_NAME = "Video Encoding"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = VideoEncodingPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    STAGE_NAME = "Generate Dataset Label"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = GenerateDatasetLabelPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    STAGE_NAME = "Training"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = TrainingPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    STAGE_NAME = "Generating Caption"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = GenerateCaptionPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    STAGE_NAME = "Refine the sentecne Caption"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = RefineGeneratedCaptionPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    STAGE_NAME = "Evaluation"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = EvaluationPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")

    STAGE_NAME = "Plot history"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = PlotHistoryPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")