import torch
from pathlib import Path
from tqdm import tqdm
from src.config.training_config import TrainCfg
from src.entity.config_entity import TrainingConfig
from src.utils.commons import seed_everything
from src.utils.collate import collate_fn
from src.utils.loss_fn import align_loss_from_embeddings
from src.utils.helperFunctions import save_checkpoint, load_checkpoint
from src.components.batchTokenizer import BatchTokenizer
from src.components.datasetLoader import VideoCaptionCSV

from src.components.trainingVideoEncoder import VideoEncoder
from src.components.validation import validate
from src.components.crossModalCaptioner import CrossModalCaptioner

from transformers import AutoImageProcessor
from torch.utils.data import DataLoader


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        seed_everything(seed=self.config.SEED)

    def training_configuration(self):
        self.cfg = TrainCfg(
            train_csv = str(self.config.train_csv),
            val_csv = str(self.config.val_csv),
            frames_per_video = self.config.FRAMES_PER_VIDEO,
            epochs = self.config.EPOCHS,
            batch_size = self.config.BATCH_SIZE,
            lr = self.config.LR,
            out_dir = str(self.config.checkpoint_dir),
            seed = self.config.SEED,
            num_workers = self.config.MAX_WORKERS,
            device = self.config.DEVICE,
            vit_name = self.config.vit_name,
            gpt2_name = self.config.gpt2_name
        )

        # --- single tokenizer (build first) ---
        self.btok = BatchTokenizer(
            self.cfg.gpt2_name,
            self.cfg.max_txt_len,
            self.cfg.caption_prefix,
        )
        self.tok = self.btok.tok  # use this everywhere
        # image processor matches ViT pretraining
        self.iproc = AutoImageProcessor.from_pretrained(self.cfg.vit_name)
        # tok = BatchTokenizer(cfg.gpt2_name, cfg.max_txt_len).tok

        # datasets
        ds_tr = VideoCaptionCSV(self.cfg.train_csv, self.cfg.frames_per_video, self.iproc, self.cfg.max_txt_len)
        ds_va = VideoCaptionCSV(self.cfg.val_csv, self.cfg.frames_per_video, self.iproc, self.cfg.max_txt_len)

        self.dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, shuffle=True,
                    num_workers=self.cfg.num_workers, collate_fn=lambda s: collate_fn(s, self.btok))
        self.dl_va = DataLoader(ds_va, batch_size=self.cfg.batch_size, shuffle=False,
                        num_workers=self.cfg.num_workers, collate_fn=lambda s: collate_fn(s, self.btok))
        
        # models
        self.model_enc = VideoEncoder(
            self.cfg.vit_name,
            self.cfg.d_model,
            self.cfg.proj_hidden,
            self.cfg.dropout,
            self.cfg.temporal_layers,
            self.cfg.temporal_heads,
        ).to(self.config.DEVICE)
        self.model_dec = CrossModalCaptioner(self.cfg.gpt2_name, self.cfg.d_model, self.cfg.dropout).to(self.config.DEVICE)

        # >>> CRITICAL: resize embeddings to match tokenizer <<<
        vocab_size = len(self.tok)
        self.model_dec.lm.resize_token_embeddings(vocab_size)
        self.model_dec.lm.config.pad_token_id = self.tok.pad_token_id
        self.model_dec.lm.config.eos_token_id = self.tok.eos_token_id

        self.params = list(self.model_dec.parameters()) + [p for n,p in self.model_enc.named_parameters() if "vit" not in n]
        self.optimizer = torch.optim.AdamW(self.params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.dl_tr)*self.cfg.epochs)
        # scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))
        self.scaler = torch.amp.GradScaler(enabled=(self.cfg.amp and self.config.DEVICE.startswith("cuda")))

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_loss_txt": [],
            "train_loss_align": [],
            "valid_loss": [],
            "valid_loss_txt": [],
            "valid_loss_align": [],
            "valid_R1": []
        }

        self.best_val = float("inf")
        self.start_epoch = 0

    def load_previous_checkpoint(self):
        if self.config.checkpoint_training_dir.exists():
            ck = load_checkpoint(self.config.checkpoint_training_dir, self.model_enc, self.model_dec)
            self.history = ck["history"]
            self.best_val = ck["best_val_loss"]
            self.start_epoch = ck["epoch"] + 1
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            self.scaler.load_state_dict(ck["scaler"])

    def train_one_epoch(self, epoch: int):
        self.model_dec.train()
        self.model_enc.train()
        if epoch < self.cfg.freeze_lm_epochs:
            for p in self.model_dec.lm.parameters(): p.requires_grad = False
        else:
            for p in self.model_dec.lm.parameters(): p.requires_grad = True

        device = self.cfg.device
        total_txt, total_align = 0.0, 0.0
        total, nstep = 0.0, 0
        grad_params = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad]

        max_lambda = self.cfg.lambda_align
        cur_lambda = max_lambda * min(1.0, epoch / max(1, self.cfg.align_warm_epochs))

        pbar = tqdm(self.dl_tr, desc=f"Train E{epoch} λ_align={cur_lambda:.2f}")
        for batch in pbar:
            pixel_values = batch["pixel_values"]
            precomp_feat = batch["precomp_feat"]
            vid_mask = batch["vid_mask"].to(device)

            if pixel_values is not None: pixel_values = pixel_values.to(device)  # [B,T,C,H,W]
            if precomp_feat is not None: precomp_feat = precomp_feat.to(device)  # [B,T,D]

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward
            if self.cfg.amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    video_tokens = self.model_enc(pixel_values, precomp_feat)  # [B,T,d]
                    # print(f"video_tokens dtype: {video_tokens.dtype}, device: {video_tokens.device}") # Debugging line
                    # print(f"vid_mask dtype: {vid_mask.dtype}, device: {vid_mask.device}") # Debugging line
                    out = self.model_dec(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        video_tokens=video_tokens,
                        video_mask=vid_mask,
                        labels=labels,
                        output_hidden_states=True
                    )
                    loss_txt = out.loss

                    txt_hidden = out.hidden_states[-1]  # [B,L,d]
                    vid_emb = self.model_dec.pooled_video(video_tokens, vid_mask)
                    txt_emb = self.model_dec.pooled_text(txt_hidden, attn_mask)
                    loss_align = align_loss_from_embeddings(vid_emb, txt_emb, self.model_dec.logit_scale)

                    loss = self.cfg.lambda_txt * loss_txt + cur_lambda * loss_align
            else:
                video_tokens = self.model_enc(pixel_values, precomp_feat)
                # print(f"video_tokens dtype: {video_tokens.dtype}, device: {video_tokens.device}") # Debugging line
                # print(f"vid_mask dtype: {vid_mask.dtype}, device: {vid_mask.device}") # Debugging line
                out = self.model_dec(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    video_tokens=video_tokens,
                    video_mask=vid_mask,
                    labels=labels,
                    output_hidden_states=True
                )
                loss_txt = out.loss
                txt_hidden = out.hidden_states[-1]
                vid_emb = self.model_dec.pooled_video(video_tokens, vid_mask)
                txt_emb = self.model_dec.pooled_text(txt_hidden, attn_mask)
                loss_align = align_loss_from_embeddings(vid_emb, txt_emb, self.model_dec.logit_scale)
                loss = self.cfg.lambda_txt * loss_txt + cur_lambda * loss_align

            self.optimizer.zero_grad(set_to_none=True)
            if self.cfg.amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(grad_params, self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(grad_params, self.cfg.grad_clip)
                self.optimizer.step()
            if self.scheduler: self.scheduler.step()

            total += loss.item()
            total_txt += loss_txt.item()
            total_align += loss_align.item()
            nstep += 1
            pbar.set_postfix(txt=f"{total_txt/nstep:.3f}", align=f"{total_align/nstep:.3f}", loss=f"{total/nstep:.3f}")

        return {"loss": total/nstep, "loss_txt": total_txt/nstep, "loss_align": total_align/nstep}
    
    def training(self):
        for ep in range(self.start_epoch, self.cfg.epochs):
            tr = self.train_one_epoch(ep)
            va = validate(
                self.model_enc,
                self.model_dec,
                self.dl_va,
                self.tok,
                self.cfg
            )
            print(f"[E{ep}] train: {tr}  valid: {va}")

            self.history["epoch"].append(ep)
            self.history["train_loss"].append(tr["loss"])
            self.history["train_loss_txt"].append(tr["loss_txt"])
            self.history["train_loss_align"].append(tr["loss_align"])
            self.history["valid_loss"].append(va["loss"])
            self.history["valid_loss_txt"].append(va["loss_txt"])
            self.history["valid_loss_align"].append(va["loss_align"])
            self.history["valid_R1"].append(va["R1"])

            if va["loss"] < self.best_val:
                self.best_val = va["loss"]
                save_checkpoint(
                    self.config.checkpoint_best_dir,
                    self.model_enc,
                    self.model_dec,
                    self.tok,
                    self.cfg,
                    ep,
                    self.history,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.best_val
            )

            save_checkpoint(
                self.config.checkpoint_training_dir,
                self.model_enc,
                self.model_dec,
                self.tok,
                self.cfg,
                ep,
                self.history,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.best_val
            )

    def run(self, train_csv: Path = None, val_csv: Path = None, epochs: int = None):
        self.training_configuration()
        if train_csv:
            self.cfg.train_csv = str(train_csv)
        if val_csv:
            self.cfg.val_csv = str(val_csv)
        if epochs:
            self.cfg.epochs = epochs
        
        self.load_previous_checkpoint()

        self.training()


if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager

    # Training
    STAGE_NAME = "Training"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_training_config()
    training = Training(config)
    training.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")
