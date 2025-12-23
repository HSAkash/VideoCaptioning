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


@dataclass
class TrainingConfig:
    vit_name:                           str
    gpt2_name:                          str
    train_csv:                          Path
    val_csv:                            Path
    checkpoint_dir:                     Path
    checkpoint_best_dir:                Path
    checkpoint_training_dir:            Path
    FRAMES_PER_VIDEO:                   int
    EPOCHS:                             int
    LR:                                 float
    BATCH_SIZE:                         int
    MAX_WORKERS:                        int
    SEED:                               int
    DEVICE:                             str


@dataclass
class GenerateCaptionConfig:
    vit_name:                           str
    gpt2_name:                          str
    checkpoint_path:                    Path
    data_dirs:                          list[Path]
    destination_dir:                    Path
    FRAMES_PER_VIDEO:                   int
    DEVICE:                             str


@dataclass
class RefineGeneratedCaptionConfig:
    process_type:                       str
    files_details:                      list[tuple[Path, Path]] # source, destination 


@dataclass
class EvaluationConfig:
    save_path:                          str
    # [model_label, dataset_label,model_path, reference_json_path, reference_column, relatum_json_path, relatum_column]
    files_details:                      list[tuple[str, str, Path, Path, str, Path, str]] # source, destination
    DEVICE:                             str
    verbose:                            bool = True


@dataclass
class PlotHistoryConfig:
    destination_dir:                    Path
    model_path:                         Path
    history_path:                       Path
    vit_name:                           str
    gpt2_name:                          str
    FRAMES_PER_VIDEO:                   int
    DEVICE:                             str