import json
import pandas as pd
from pathlib import Path
from src.utils.commons import load_json_data
from src.utils.interrupt_check import DelayedInterruptMainProcess
from src.entity.config_entity import GenerateDatasetLabelConfig


class GenerateDatasetLabel:
    def __init__(self, config: GenerateDatasetLabelConfig):
        self.config = config

    def generate(
            self,
            label: str,
            source_json_path: Path,
            video_data_dir: Path
        ):
        destination_csv_path = (self.config.destination_root_dir / label).with_suffix('.csv')
        source_json_data = load_json_data(source_json_path)

        json_data_dict = {}
        for item in source_json_data:
            video_id, captions = item['video_id'], item['caption']
            json_data_dict[video_id] = captions
        
        filnal_data_dict = {
            'video_id': [],
            'caption': [],
            'frame_dir': []
        }

        for video_path in sorted(video_data_dir.glob("*")):
            caption_id = video_path.name
            caption_id = caption_id.split("___")[0].replace("_gray","")
            captions = json_data_dict[caption_id]
            for caption in captions:
                filnal_data_dict['frame_dir'].append(video_path)
                filnal_data_dict['video_id'].append(caption_id)
                filnal_data_dict['caption'].append(caption)
        
        with DelayedInterruptMainProcess():
            pd.DataFrame(filnal_data_dict).to_csv(destination_csv_path, index=False)

    def run(self):
        self.config.destination_root_dir.mkdir(parents=True, exist_ok= True)
        for (label,
            source_json_path,
            video_data_dir) in self.config.dataset_details:
            self.generate(
                label,
                source_json_path,
                video_data_dir
            )


if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager

    # Generate Dataset Labels
    STAGE_NAME = "Generate Dataset Labels"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_generate_dataset_label_config()
    generateDatasetLabel = GenerateDatasetLabel(config)
    generateDatasetLabel.run()
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {config.destination_root_dir}")
