import os
from pyprojroot import here
from src.utils.commons import read_yaml
from src.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH
from src.entity.config_entity import (
    DownloadDatasetConfig,
    UnzipDatasetConfig,
    ImageExtractionSplitConfig,
    AugmentationConfig,
)


class ConfigurationManager:
    def __init__(self, config_file_path: str=CONFIG_FILE_PATH, param_file_path: str=PARAM_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(param_file_path)

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
    
    def get_image_extraction_split_config(self) -> ImageExtractionSplitConfig:
        config = self.config.imageExtractionSplit

        caption_details = [(here(item.file_path), item.label) for item in config.caption_details]
        
        # set MAX_WORKERS
        cpu_cores = os.cpu_count() or 1
        MAX_WORKERS = min(self.params.MAX_WORKERS, cpu_cores) or 1 # take which one minimum
        
        return ImageExtractionSplitConfig(
            video_source_dir = here(config.video_source_dir),
            caption_details = caption_details,
            image_destination_dir = here(config.image_destination_dir),
            image_format = config.image_format,
            FRAMES_PER_VIDEO = self.params.FRAMES_PER_VIDEO,
            IMAGE_SIZE = self.params.IMAGE_SIZE,
            MAX_WORKERS = MAX_WORKERS
        )
    
    def get_augmentation_config(self) -> AugmentationConfig:
        config = self.config.augmentation
        params = self.params.augmentation

        source_destination_dirs = [(here(item[0]), here(item[1])) for item in config.source_destination_dirs] 
        
        # set MAX_WORKERS
        cpu_cores = os.cpu_count() or 1
        MAX_WORKERS = min(self.params.MAX_WORKERS, cpu_cores) or 1 # take which one minimum

        return AugmentationConfig(
            source_destination_dirs = source_destination_dirs,
            image_format = config.image_format,
            resize_crop_scale = tuple(params.resize_crop_scale),
            horizontal_flip_p = params.horizontal_flip_p,
            color_jitter = tuple(params.color_jitter),
            gaussian_blur_sigma = tuple(params.gaussian_blur_sigma),
            N = params.N,
            FRAMES_PER_VIDEO = self.params.FRAMES_PER_VIDEO,
            IMAGE_SIZE = self.params.IMAGE_SIZE,
            MAX_WORKERS = MAX_WORKERS
        )