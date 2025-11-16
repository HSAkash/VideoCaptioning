import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config, GPT2LMHeadModel


class CrossModalCaptioner(nn.Module):
    """GPT2 decoder with cross-attention over video tokens + contrastive head"""
    def __init__(self, gpt2_name: str, d_model: int, dropout: float):
        super().__init__()
        # Re-init GPT2 with cross-attention
        base_cfg = GPT2Config.from_pretrained(gpt2_name)
        cfg = GPT2Config(
            vocab_size=base_cfg.vocab_size,
            n_embd=base_cfg.n_embd,
            n_layer=base_cfg.n_layer,
            n_head=base_cfg.n_head,
            n_positions=base_cfg.n_positions,
            n_ctx=base_cfg.n_ctx,
            add_cross_attention=True,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout
        )
        self.lm = GPT2LMHeadModel(cfg)
        self.lm.resize_token_embeddings(base_cfg.vocab_size)
        assert d_model == cfg.n_embd, f"d_model={d_model} must match GPT2 hidden size {cfg.n_embd}"

        # Projection heads for alignment (video/text -> common space)
        self.proj_vid = nn.Linear(d_model, d_model)
        self.proj_txt = nn.Linear(d_model, d_model)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))  # like CLIP

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=video_tokens,
            encoder_attention_mask=video_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True
        )
        return out

    def pooled_video(self, video_tokens: torch.Tensor, video_mask: torch.Tensor) -> torch.Tensor:
        # masked mean pool over time
        mask = video_mask.unsqueeze(-1).float()  # [B,T,1]
        s = (video_tokens * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        return F.normalize(self.proj_vid(s), dim=-1)  # [B,d]

    def pooled_text(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # take last layer hidden, masked mean pool (exclude padding)
        mask = attention_mask.unsqueeze(-1).float()
        s = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        return F.normalize(self.proj_txt(s), dim=-1)