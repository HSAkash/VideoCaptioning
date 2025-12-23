from src.config.configuration import ConfigurationManager
from src.components.plotHistory import PlotHistory
from pathlib import Path

class PlotHistoryPipeline:
    def __init__(self):
        pass

    def run(self, history_path:Path=None):
        config = ConfigurationManager().get_plot_history_config()
        caption = PlotHistory(config).run(history_path=history_path)


if __name__ == "__main__":
    from src import logger
    STAGE_NAME = "Plot history"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = PlotHistoryPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")