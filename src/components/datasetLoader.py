import os, glob
import torch
from PIL import Image
import pandas as pd
from typing import List
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

def list_frames_sorted(frame_dir: str) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frame_dir, e)))
    # numeric sort (000001.jpg, 0002.jpg, 10.jpg handled)
    files = sorted(files, key=lambda p: (len(p), p))
    if not files:
        raise FileNotFoundError(f"No frames found in {frame_dir}")
    return files

def uniform_pick(seq: List[str], T: int) -> List[str]:
    n = len(seq)
    if n == T:
        return seq
    if n < T:
        # pad last frame
        return seq + [seq[-1]] * (T - n)
    # select T roughly uniform indices
    step = n / T
    return [seq[int(i*step)] for i in range(T)]

class VideoCaptionCSV(Dataset):
    def __init__(self, csv_path: str, frames_per_video: int, image_processor: AutoImageProcessor, max_txt_len: int):
        self.df = pd.read_csv(csv_path)
        assert "frame_dir" in self.df.columns and "caption" in self.df.columns, \
            "CSV must have columns: frame_dir, caption"
        self.frames_per_video = frames_per_video
        self.iproc = image_processor
        self.max_txt_len = max_txt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frame_dir = row["frame_dir"]
        cap = str(row["caption"])

        # optional: precomputed features at frame_dir/features.pt -> torch.Size([T,D]) normalized?
        feat_path = os.path.join(frame_dir, "features.pt")
        if os.path.exists(feat_path):
            feat = torch.load(feat_path)  # [T, D]
            # ensure T
            if feat.size(0) < self.frames_per_video:
                pad = feat[-1:].repeat(self.frames_per_video - feat.size(0), 1)
                feat = torch.cat([feat, pad], dim=0)
            elif feat.size(0) > self.frames_per_video:
                # uniform sample rows
                idxs = torch.linspace(0, feat.size(0)-1, steps=self.frames_per_video).round().long()
                feat = feat[idxs]
            pixel_values = None
        else:
            # load and preprocess frames
            frame_paths = uniform_pick(list_frames_sorted(frame_dir), self.frames_per_video)
            imgs = [Image.open(p).convert("RGB") for p in frame_paths]
            # image_processor returns dict with pixel_values [1,C,H,W] for each
            batch = self.iproc(images=imgs, return_tensors="pt")
            pixel_values = batch["pixel_values"]  # [T, C, H, W]
            feat = None

        return {
            "pixel_values": pixel_values,  # [T,C,H,W] or None
            "precomp_feat": feat,          # [T,D] or None
            "caption": cap
        }