from src.config.configuration import ConfigurationManager
from src.components.evaluation import Evaluation
from pathlib import Path
from src import logger

class EvaluationPipeline:
    def __init__(self):
        pass

    def run(
            self,
            model_path: Path=None,
            reference_json_path: Path=None,
            reference_column: str=None,
            generated_json_path: str = False,
            generated_column: str=None,
            save_path: Path=None,
            verbose: bool=True

        ):
        config = ConfigurationManager().get_evaluation_config()
        caption = Evaluation(config).run(
            model_path = model_path,
            reference_json_path = reference_json_path,
            reference_column =reference_column,
            generated_json_path =generated_json_path,
            generated_column =generated_column,
            save_path =save_path,
            verbose = verbose
        )
        if caption:
            logger.info(f">>> Evaluation score saved in : {save_path or config.save_path}")


if __name__ == "__main__":
    STAGE_NAME = "Evaluation"
    logger.info(f">>> stage {STAGE_NAME} started")
    pipeline = EvaluationPipeline()
    pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")