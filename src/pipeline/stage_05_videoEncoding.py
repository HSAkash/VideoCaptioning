from src.config.configuration import ConfigurationManager
from src.components.videoEncoder import VideoEncoder


class VideoEncodingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_video_encoding_config()
        VideoEncoder(config).run()


if __name__ == "__main__":
    from src import logger
    
    STAGE_NAME = "Video Encoding"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = VideoEncodingPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")