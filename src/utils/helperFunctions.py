import gdown
import zipfile
from tqdm import tqdm
from pathlib import Path
from src.entity.config_entity import (
    DownloadDatasetConfig,
    UnzipDatasetConfig
)

def downloadDataset(config: DownloadDatasetConfig):
    url = config.url
    output_path = config.output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdown.download(url=url, output=output_path.__str__(), resume=True)

def unzipDataset(config: UnzipDatasetConfig):
    extract_dir = config.extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(config.zip_path, 'r') as zip_ref:
        for file in tqdm(zip_ref.infolist(), desc="Extracting", unit="file"):
            target_path = extract_dir / file.filename

            # Create parent dirs if nested
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file already exists and is complete
            if target_path.exists() and target_path.stat().st_size >= file.file_size:
                continue  # Skip — already done or larger (rare)

            # (Re)extract this file
            zip_ref.extract(file, extract_dir)


if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager
    configurationManager = ConfigurationManager()

    # Download dataset
    STAGE_NAME = "Download Dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    download_dataset_config = configurationManager.get_download_dataset_config()
    downloadDataset(download_dataset_config)
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {download_dataset_config.output_path}")

    # Unzip dataset
    STAGE_NAME = "Unzip dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    unzip_dataset_config = configurationManager.get_unzip_dataset_config()
    unzipDataset(unzip_dataset_config)
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {unzip_dataset_config.extract_dir}")

