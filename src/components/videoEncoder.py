import os
import re
import glob
import math
import argparse
from typing import List
from pathlib import Path
from pyprojroot import here

import torch
from PIL import Image
from tqdm.auto import tqdm
from src.utils.commons import seed_everything
from transformers import AutoImageProcessor, ViTModel
from src.entity.config_entity import VideoEncodingConfig
from src.utils.interrupt_check import DelayedInterruptMainProcess


class VideoEncoder:
    def __init__(
        self,
        config: VideoEncodingConfig
    ):
        self.config = config
        seed_everything(seed=self.config.SEED)
        # ViT + processor
        self.iproc = AutoImageProcessor.from_pretrained(self.config.vit_name, use_fast=True)
        self.vit = ViTModel.from_pretrained(self.config.vit_name).to(self.config.DEVICE)
        self.vit.eval()
        
    def uniform_pick(self, paths: List[str], T: int) -> List[str]:
        n = len(paths)
        if n == T:
            return paths
        if n < T:
            return paths + [paths[-1]] * (T - n)  # pad with last frame
        step = n / T
        return [paths[int(i * step)] for i in range(T)]

    @torch.no_grad()
    def encode_video_dir( self, frame_dir: Path) -> torch.Tensor:
        """
        Returns CLS features of shape [T, D] (float32, on CPU).
        Processes frames in mini-batches to save memory.
        """
        # Select frames
        fpaths = sorted(frame_dir.glob(f"*.{self.config.image_format}"))
        fpaths = self.uniform_pick(fpaths, self.config.FRAMES_PER_VIDEO)
        # print(len(fpaths))

        # Process in chunks
        feats = []
        for i in range(0, len(fpaths), self.config.FRAMES_BATCH):
            chunk_paths = fpaths[i:i + self.config.FRAMES_BATCH]
            imgs = [Image.open(p).convert("RGB") for p in chunk_paths]
            batch = self.iproc(images=imgs, return_tensors="pt")
            pix = batch["pixel_values"].to(self.config.DEVICE)  # [B,C,H,W]

            out = self.vit(pix, output_hidden_states=False)
            cls = out.last_hidden_state[:, 0, :]  # [B, D]
            feats.append(cls.cpu().to(torch.float32))

            # close PIL images explicitly
            for im in imgs:
                try:
                    im.close()
                except Exception:
                    pass

        feats = torch.cat(feats, dim=0)  # [T, D]
        assert feats.shape[0] == self.config.FRAMES_PER_VIDEO, f"Got {feats.shape[0]} frames, expected {self.config.FRAMES_PER_VIDEO}"
        return feats.contiguous()

    def encode(self, source_dir:Path, destination_dir:Path):
        video_dirs = sorted(source_dir.glob('*'))
        for video_dir in tqdm(video_dirs, desc=f"{source_dir.stem}"):
            destination_path = destination_dir / video_dir.stem / "features.pt"
            if destination_path.exists():
                continue
            try:
                feats = self.encode_video_dir(frame_dir=video_dir)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                with DelayedInterruptMainProcess():
                    torch.save(feats, destination_path)
            except Exception as e:
                print(f"[ERROR] {video_dir}: {e}")

    def run(self):
        for sub_folder in self.config.sub_folders:
            source_dir = self.config.source_root_dir / sub_folder
            destination_dir = self.config.destination_root_dir / sub_folder
            self.encode(source_dir, destination_dir)


if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager

    # Video Encoding
    STAGE_NAME = "Video Encoding"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_video_encoding_config()
    videoEncoder = VideoEncoder(config)
    videoEncoder.run()
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {config.destination_root_dir}")





