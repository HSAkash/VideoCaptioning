from src.config.configuration import ConfigurationManager
from src.components.augmentation import Augmentation


class AugmentationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_augmentation_config()
        Augmentation(config).run()


if __name__ == "__main__":
    from src import logger
    
    STAGE_NAME = "Augmentation"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = AugmentationPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")