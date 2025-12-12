import os
import errno
import gdown
import torch
import zipfile
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
from src.config.training_config import TrainCfg
from src.utils.commons import load_json_data, save_json_data
from src.entity.config_entity import (
    DownloadDatasetConfig,
    UnzipDatasetConfig
)

def caption_str_to_list(json_path:Path, col:str='caption'):
    if not json_path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), json_path)

    json_data = load_json_data(json_path)
    for idx in range(len(json_data)):
        if isinstance(json_data[idx][col], str):
            json_data[idx][col] = [json_data[idx][col]]
    save_json_data(json_data, json_path)


def downloadDataset(config: DownloadDatasetConfig):
    # Video download
    video_output_path = config.video_output_path
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url=config.video_url, output=video_output_path.__str__(), resume=True)

    # Caption download
    caption_output_dir = config.caption_output_dir
    caption_output_dir.mkdir(parents=True, exist_ok=True)
    for item in config.caption_urls:
        file_name, url = item.file_name, item.url
        output_path = caption_output_dir / file_name
        gdown.download(url=url, output=output_path.__str__(), resume=True)
        caption_str_to_list(output_path)

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

def save_checkpoint(path_dir: str, model_enc, model_dec, tok, cfg: TrainCfg, epoch: int, history, optimizer, scheduler, scaler, best_val_loss):
    os.makedirs(path_dir, exist_ok=True)
    # tokenizer
    tok.save_pretrained(os.path.join(path_dir, "tokenizer"))
    ck = {
        "epoch": epoch,
        "cfg": asdict(cfg),
        "model_dec": model_dec.state_dict(),
        "model_enc_proj": model_enc.state_dict(),  # includes projector & PE, not ViT weights change (frozen)
        "history": history,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss
    }
    torch.save(ck, os.path.join(path_dir, "checkpoint.pt"))

def load_checkpoint(path_dir: str, model_enc, model_dec):
    ck = torch.load(os.path.join(path_dir, "checkpoint.pt"), map_location="cpu")
    model_dec.load_state_dict(ck["model_dec"], strict=False)
    model_enc.load_state_dict(ck["model_enc_proj"], strict=False)

    return ck, 

    

if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager
    configurationManager = ConfigurationManager()

    # Download dataset
    STAGE_NAME = "Download Dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    download_dataset_config = configurationManager.get_download_dataset_config()
    downloadDataset(download_dataset_config)
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {download_dataset_config.video_output_path}")

    # Unzip dataset
    STAGE_NAME = "Unzip dataset"
    logger.info(f">>> stage {STAGE_NAME} started")
    unzip_dataset_config = configurationManager.get_unzip_dataset_config()
    unzipDataset(unzip_dataset_config)
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {unzip_dataset_config.extract_dir}")