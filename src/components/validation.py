import  torch
from tqdm import tqdm
from src.config.training_config import TrainCfg
from src.utils.loss_fn import align_loss_from_embeddings

@torch.no_grad()
def validate(model_enc, model_dec, dl, tok, cfg: TrainCfg):
    model_enc.eval()
    model_dec.eval()
    device = cfg.device
    losses, losses_txt, losses_align = [], [], []
    # A tiny retrieval metric: in-batch R@1
    r1_hits = 0
    samples = 0

    for batch in tqdm(dl, desc="Valid"):
        pixel_values = batch["pixel_values"]
        precomp_feat = batch["precomp_feat"]
        vid_mask = batch["vid_mask"].to(device)

        if pixel_values is not None: pixel_values = pixel_values.to(device)
        if precomp_feat is not None: precomp_feat = precomp_feat.to(device)

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        labels = batch["labels"].to(device)

        video_tokens = model_enc(pixel_values, precomp_feat)
        out = model_dec(
            input_ids=input_ids,
            attention_mask=attn_mask,
            video_tokens=video_tokens,
            video_mask=vid_mask,
            labels=labels,
            output_hidden_states=True
        )
        loss_txt = out.loss
        txt_hidden = out.hidden_states[-1]
        vid_emb = model_dec.pooled_video(video_tokens, vid_mask)
        txt_emb = model_dec.pooled_text(txt_hidden, attn_mask)
        loss_align = align_loss_from_embeddings(vid_emb, txt_emb, model_dec.logit_scale)
        loss = cfg.lambda_txt * loss_txt + cfg.lambda_align * loss_align

        losses.append(loss.item())
        losses_txt.append(loss_txt.item())
        losses_align.append(loss_align.item())

        # Retrieval R@1
        sims = (vid_emb @ txt_emb.t()).softmax(dim=-1)  # [B,B]
        pred = sims.argmax(dim=-1)
        target = torch.arange(vid_emb.size(0), device=vid_emb.device)
        r1_hits += (pred == target).sum().item()
        samples += vid_emb.size(0)

    return {
        "loss": sum(losses)/len(losses),
        "loss_txt": sum(losses_txt)/len(losses_txt),
        "loss_align": sum(losses_align)/len(losses_align),
        "R1": r1_hits / max(1, samples)
    }
