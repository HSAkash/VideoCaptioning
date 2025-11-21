from src.config.configuration import ConfigurationManager
from src.components.refineGeneratedCaption import RefineGeneratedCaption
from pathlib import Path
from src import logger


class RefineGeneratedCaptionPipeline:
    def __init__(self):
        pass

    def run(
            self,
            source: Path=None,
            destination: Path=None,
            process_type: str=None,
            caption: str = False
        ):
        config = ConfigurationManager().get_refine_generated_caption_config()
        caption = RefineGeneratedCaption(config).run(
            source_file = source,
            destination_file = destination,
            process_type = process_type,
            caption=caption
        )
        if caption:
            logger.info(f">>> Refine caption: {caption}")


if __name__ == "__main__":
    STAGE_NAME = "Refine the sentecne Caption"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = RefineGeneratedCaptionPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")