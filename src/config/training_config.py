from dataclasses import dataclass, asdict
import torch

@dataclass
class TrainCfg:
    # data
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    val_csv: str = "val.csv"
    frames_per_video: int = 128
    image_size: int = 224
    max_txt_len: int = 40

    # model
    vit_name: str = "google/vit-base-patch16-224-in21k"
    gpt2_name: str = "gpt2"  # we will re-init with add_cross_attention=True but keep tokenizer/vocab from gpt2
    d_model: int = 768  # must match GPT2 hidden size
    proj_hidden: int = 768  # video -> d_model
    dropout: float = 0.1
    temporal_layers: int = 1
    temporal_heads: int = 8
    caption_prefix: str = "caption: "

    # loss weights
    lambda_txt: float = 1.0
    lambda_align: float = 0.1
    align_warm_epochs: int = 1  # warm up alignment over first N epochs

    # optimization
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    # training tricks
    freeze_lm_epochs: int = 0  # keep pretrained LM trainable unless explicitly experimenting

    # decoding
    num_beams: int = 5
    max_gen_len: int = 30
    do_sample: bool = False

    # ckpt
    out_dir: str = "checkpoints"
    save_every: int = 1
    seed: int = 42
