from pyprojroot import here
from pathlib import Path
from src.entity.config_entity import RefineGeneratedCaptionConfig
from src.components.cloudRefiner import CloudRefiner

class RefineGeneratedCaption:
    def __init__(self, config: RefineGeneratedCaptionConfig):
        self.config = config
        self.cloudRefiner = CloudRefiner()


    def run(self, source_file: Path=None, destination_file: Path=None, process_type: str=None, caption: str=None):
        if source_file and destination_file:
            self.config.files_details = [(here(source_file), here(destination_file))]
        if process_type:
            self.config.process_type = process_type

        if caption and self.config.process_type == 'cloud':
            return self.cloudRefiner.single_sentence_refine(caption)
        if caption == None:
            for source_file, destination_file in self.config.files_details:
                if self.config.process_type == 'cloud':
                    self.cloudRefiner.run(source_path=source_file, destination_path=destination_file)


if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager

    # Video Encoding
    STAGE_NAME = "Refine Captions Using DeepSeek"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_refine_generated_caption_config()
    refiner = RefineGeneratedCaption(config)
    refiner.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")
