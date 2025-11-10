from pyprojroot import here
from src.utils.commons import read_yaml
from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import (
    DownloadDatasetConfig,
    UnzipDatasetConfig,
)


class ConfigurationManager:
    def __init__(self, config_file_path: str=CONFIG_FILE_PATH):
        self.config = read_yaml(config_file_path)

    def get_download_dataset_config(self) -> DownloadDatasetConfig:
        config = self.config.downloadDataset
        
        return DownloadDatasetConfig(
            video_url = config.video_url,
            video_output_path = here(config.video_output_path),
            caption_urls = config.caption_urls,
            caption_output_dir = here(config.caption_output_dir)
        )
    
    def get_unzip_dataset_config(self) -> UnzipDatasetConfig:
        config = self.config.unzipDataset

        return UnzipDatasetConfig(
            zip_path = here(config.zip_path),
            extract_dir = here(config.extract_dir)
        )