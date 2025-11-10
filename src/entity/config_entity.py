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