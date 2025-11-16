import  torch
import torch.nn.functional as F

def align_loss_from_embeddings(vid_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    # vid_emb / txt_emb normalized already
    logits = logit_scale.exp().clamp(max=100) * vid_emb @ txt_emb.t()  # [B,B]
    B = vid_emb.size(0)
    target = torch.arange(B, device=vid_emb.device)
    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i2t + loss_t2i)