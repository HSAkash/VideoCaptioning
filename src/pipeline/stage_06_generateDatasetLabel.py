from src.config.configuration import ConfigurationManager
from src.components.generateDatasetLabel import GenerateDatasetLabel


class GenerateDatasetLabelPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_generate_dataset_label_config()
        GenerateDatasetLabel(config).run()


if __name__ == "__main__":
    from src import logger
    
    STAGE_NAME = "Generate Dataset Label"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = GenerateDatasetLabelPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")