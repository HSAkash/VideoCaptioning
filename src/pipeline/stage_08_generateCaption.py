from src.config.configuration import ConfigurationManager
from src.components.generateCaption import GenerateCaption
from pathlib import Path
from src import logger


class GenerateCaptionPipeline:
    def __init__(self):
        pass

    def run(
            self,
            folder_path: Path=None,
            model_path: Path=None,
            generated_text_save_dir_path: Path=None,
            is_signle_path: bool = False
        ):
        config = ConfigurationManager().get_generate_caption_config()
        if is_signle_path:
            caption =  GenerateCaption(config).generate(folder_path, model_path)
            logger.info(f">>> Caption: {caption}")
        else:
            GenerateCaption(config).run(folder_path, model_path, generated_text_save_dir_path)


if __name__ == "__main__":
    STAGE_NAME = "Generating Caption"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = GenerateCaptionPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")