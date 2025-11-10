from src.config.configuration import ConfigurationManager
from src.components.imageExtractionSplit import ImageExtractionSplit


class ImageExtractionPipeline:
    def __init__(self):
        pass

    def run(self, resume: bool=True):
        config = ConfigurationManager().get_image_extraction_split_config()
        ImageExtractionSplit(config, resume=resume).run()


if __name__ == "__main__":
    from src import logger
    
    STAGE_NAME = "Image Extraction"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = ImageExtractionPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")