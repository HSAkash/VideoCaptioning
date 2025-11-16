from pathlib import Path
from dataclasses import dataclass

@dataclass
class DownloadDatasetConfig:
    video_url:                          str
    video_output_path:                  Path
    caption_urls:                       list
    caption_output_dir:                 Path


@dataclass
class UnzipDatasetConfig:
    zip_path:                           Path
    extract_dir:                        Path


@dataclass
class ImageExtractionSplitConfig:
    video_source_dir:                   Path
    caption_details:                    list[tuple[Path, str]]
    image_destination_dir:              Path
    image_format:                       str
    FRAMES_PER_VIDEO:                   int
    IMAGE_SIZE:                         int
    MAX_WORKERS:                        int
    

@dataclass
class AugmentationConfig:
    source_destination_dirs:            list[tuple[Path, Path]]
    image_format:                       str
    N:                                  int # N time augmentation
    FRAMES_PER_VIDEO:                   int
    IMAGE_SIZE:                         int
    MAX_WORKERS:                        int
    resize_crop_scale:                  tuple[float, float] = (0.9, 1.0)
    horizontal_flip_p:                  float = 0.5
    color_jitter:                       tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.0) # (brightness, contrast, saturation, hue)
    gaussian_blur_sigma:                tuple[float, float] = (0.1, 1.0)
    

@dataclass
class VideoEncodingConfig:
    vit_name:                           str
    source_root_dir:                    Path
    sub_folders:                        list[str]
    destination_root_dir:               Path
    image_format:                       str
    FRAMES_PER_VIDEO:                   int
    FRAMES_BATCH:                       int
    MAX_WORKERS:                        int
    DEVICE:                             str
    SEED:                               int


@dataclass
class GenerateDatasetLabelConfig:
    dataset_details:                    list[tuple[str, Path, Path]] # label, source_json_path, video_data_dir
    destination_root_dir:               Path
