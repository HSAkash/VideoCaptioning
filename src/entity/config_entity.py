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
    