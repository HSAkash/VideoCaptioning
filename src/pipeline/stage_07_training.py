from src.config.configuration import ConfigurationManager
from src.components.training import Training
from pathlib import Path


class TrainingPipeline:
    def __init__(self):
        pass

    def run(self, train_csv: Path = None, val_csv: Path = None, epochs: int = None):
        config = ConfigurationManager().get_training_config()
        Training(config).run(train_csv=train_csv, val_csv=val_csv, epochs=epochs)


if __name__ == "__main__":
    from src import logger
    
    STAGE_NAME = "Training"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = TrainingPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")