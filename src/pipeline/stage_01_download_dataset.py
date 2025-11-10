from src import logger
from src.config.configuration import ConfigurationManager
from src.utils.helperFunctions import downloadDataset


class DownloadDatasetPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_download_dataset_config()
        downloadDataset(config)
        logger.info(f">>> Save it to: {config.video_output_path}")


if __name__ == "__main__":

    downloadDatasetPipeline = DownloadDatasetPipeline()

    # Download dataset
    STAGE_NAME = "Download Dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    downloadDatasetPipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")