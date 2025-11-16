import torch
from src.components.batchTokenizer import BatchTokenizer

def collate_fn(samples, tokenizer: BatchTokenizer):
    # videos
    has_pix = samples[0]["pixel_values"] is not None
    has_feat = samples[0]["precomp_feat"] is not None
    if has_pix:
        # stack to [B,T,C,H,W]
        T = samples[0]["pixel_values"].shape[0]
        pix = torch.stack([s["pixel_values"] for s in samples], dim=0)
    else:
        pix = None
        T = samples[0]["precomp_feat"].shape[0]

    if has_feat:
        feat = torch.stack([s["precomp_feat"] for s in samples], dim=0)  # [B,T,D]
    else:
        feat = None

    caps = [s["caption"] for s in samples]
    input_ids, attn_mask, labels, _ = tokenizer(caps)

    # video mask (all real frames; if you use variable T, set accordingly)
    B = len(samples)
    vid_mask = torch.ones(B, T, dtype=torch.long)

    return {
        "pixel_values": pix,       # [B,T,C,H,W] or None
        "precomp_feat": feat,      # [B,T,D] or None
        "vid_mask": vid_mask,      # [B,T]
        "input_ids": input_ids,    # [B,L]
        "attn_mask": attn_mask,    # [B,L]
        "labels": labels           # [B,L]
    }