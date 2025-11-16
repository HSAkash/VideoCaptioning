import torch
import torch.nn as nn
from transformers import ViTModel

from typing import Optional


class VideoEncoder(nn.Module):
    """ViT per-frame encoder -> CLS -> projection to d_model with temporal pos enc"""
    def __init__(self, vit_name: str, d_model: int, proj_hidden: int, dropout: float):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_name)
        self.vit.eval()  # keep in eval; we won't fine-tune by default (can unfreeze later)
        self.vit_hidden = self.vit.config.hidden_size  # e.g., 768
        self.proj = nn.Sequential(
            nn.LayerNorm(self.vit_hidden),
            nn.Linear(self.vit_hidden, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, d_model),
        )
        self.temporal_pe = nn.Parameter(torch.zeros(512, d_model))  # max T=512
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

    @torch.no_grad()
    def encode_frames(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, T, C, H, W] or [T,C,H,W]
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
        B, T = pixel_values.shape[:2]
        x = pixel_values.reshape(B*T, *pixel_values.shape[2:])
        out = self.vit(x, output_hidden_states=False)
        cls = out.last_hidden_state[:, 0, :]  # [B*T, Hv]
        cls = cls.reshape(B, T, self.vit_hidden)
        return cls  # [B,T,Hv]

    def forward(self, pixel_values: Optional[torch.Tensor], precomp_feat: Optional[torch.Tensor]) -> torch.Tensor:
        # returns video tokens: [B, T, d_model]
        if precomp_feat is not None:
            if precomp_feat.ndim == 2:  # [T,D] -> [1,T,D]
                precomp_feat = precomp_feat.unsqueeze(0)
            vid = precomp_feat  # assume features in ViT hidden space or generic D; we'll still project
        else:
            assert pixel_values is not None
            with torch.no_grad():
                vid_vit = self.encode_frames(pixel_values)  # [B,T,Hv]
            vid = vid_vit
        B, T, D = vid.shape
        vid_proj = self.proj(vid)  # [B,T,d_model]
        pe = self.temporal_pe[:T].unsqueeze(0)  # [1,T,d_model]
        return vid_proj + pe  # [B,T,d_model]