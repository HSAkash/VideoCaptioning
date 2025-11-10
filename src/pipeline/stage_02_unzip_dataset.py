from src import logger
from src.config.configuration import ConfigurationManager
from src.utils.helperFunctions import unzipDataset


class UnzipDatasetPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_unzip_dataset_config()
        unzipDataset(config)
        logger.info(f">>> Save it to: {config.extract_dir}")


if __name__ == "__main__":

    unzipDatasetPipeline = UnzipDatasetPipeline()

    # Unzip dataset
    STAGE_NAME = "Unzip dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    unzipDatasetPipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")