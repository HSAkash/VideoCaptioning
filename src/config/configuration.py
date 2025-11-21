import os
import torch
from pyprojroot import here
from src.utils.commons import read_yaml
from src.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH
from src.entity.config_entity import (
    DownloadDatasetConfig,
    UnzipDatasetConfig,
    ImageExtractionSplitConfig,
    AugmentationConfig,
    VideoEncodingConfig,
    GenerateDatasetLabelConfig,
    TrainingConfig,
    GenerateCaptionConfig,
    RefineGeneratedCaptionConfig,
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
    
    def get_video_encoding_config(self) -> VideoEncodingConfig:
        config = self.config.videoEncoding
        
        DEVICE = self.params.DEVICE if torch.cuda.is_available() else 'cpu'
        # set MAX_WORKERS
        cpu_cores = os.cpu_count() or 1
        MAX_WORKERS = min(self.params.MAX_WORKERS, cpu_cores) or 1 # take which one minimum

        return VideoEncodingConfig(
            vit_name = config.vit_name,
            source_root_dir = here(config.source_root_dir),
            sub_folders = config.sub_folders,
            destination_root_dir = here(config.destination_root_dir),
            image_format = config.image_format,
            FRAMES_PER_VIDEO = self.params.FRAMES_PER_VIDEO,
            FRAMES_BATCH = self.params.videoEncoding.FRAMES_BATCH,
            MAX_WORKERS = MAX_WORKERS,
            DEVICE = DEVICE,
            SEED = self.params.SEED
        )
    
    def get_generate_dataset_label_config(self) -> GenerateDatasetLabelConfig:
        config = self.config.generateDatasetLabel

        dataset_details = [
            (
                item.label, # label
                here(item.source_json_path), # source_json_path
                here(item.video_data_dir), # video_data_dir
            )for item in config.dataset_details
        ]

        return GenerateDatasetLabelConfig(
            dataset_details = dataset_details,
            destination_root_dir = here(config.destination_root_dir)
        )
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params.training

        DEVICE = self.params.DEVICE if torch.cuda.is_available() else 'cpu'
        # set MAX_WORKERS
        cpu_cores = os.cpu_count() or 1
        MAX_WORKERS = min(self.params.MAX_WORKERS, cpu_cores) or 1 # take which one minimum

        return TrainingConfig(
            vit_name = config.vit_name,
            gpt2_name = config.gpt2_name,
            train_csv = here(config.train_csv),
            val_csv = here(config.val_csv),
            checkpoint_dir = here(config.checkpoint_dir),
            checkpoint_best_dir = here(config.checkpoint_best_dir),
            checkpoint_training_dir = here(config.checkpoint_training_dir),
            FRAMES_PER_VIDEO = self.params.FRAMES_PER_VIDEO,
            EPOCHS = params.EPOCHS,
            LR = float(params.LR),
            BATCH_SIZE = self.params.BATCH_SIZE,
            MAX_WORKERS = MAX_WORKERS,
            SEED = self.params.SEED,
            DEVICE = DEVICE
        )
    
    def get_generate_caption_config(self) -> GenerateCaptionConfig:
        config = self.config.generateCaption

        DEVICE = self.params.DEVICE if torch.cuda.is_available() else 'cpu'
        data_dirs = [here(x) for x in config.data_dirs]
        
        return GenerateCaptionConfig(
            vit_name = config.vit_name,
            gpt2_name = config.gpt2_name,
            checkpoint_path = here(config.checkpoint_path),
            data_dirs = data_dirs,
            destination_dir = here(config.destination_dir),
            FRAMES_PER_VIDEO = self.params.FRAMES_PER_VIDEO,
            DEVICE = DEVICE
        )
    
    def get_refine_generated_caption_config(self) -> RefineGeneratedCaptionConfig:
        config = self.config.refineGeneratedCaption

        files_details = [(here(item['source']), here(item['destination'])) for item in config.files_details]
        
        return RefineGeneratedCaptionConfig(
            process_type = config.process_type,
            files_details = files_details
        )