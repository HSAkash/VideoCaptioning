from pathlib import Path
from dataclasses import dataclass

@dataclass
class DownloadDatasetConfig:
    url:                                str
    output_path:                        Path

@dataclass
class UnzipDatasetConfig:
    zip_path:                           Path
    extract_dir:                        Path