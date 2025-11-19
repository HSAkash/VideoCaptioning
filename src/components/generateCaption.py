import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pyprojroot import here
from src.config.training_config import TrainCfg
from src.entity.config_entity import GenerateCaptionConfig, VideoEncodingConfig, ImageExtractionSplitConfig
from src.components.trainingVideoEncoder import VideoEncoder
from src.components.videoEncoder import VideoEncoder as VE
from src.components.crossModalCaptioner import CrossModalCaptioner
from src.utils.helperFunctions import load_checkpoint
from src.config.configuration import ConfigurationManager
from src.components.imageExtractionSplit import ImageExtractionSplit
from src.utils.interrupt_check import DelayedInterruptMainProcess
from transformers import AutoImageProcessor

from transformers import AutoTokenizer

class GenerateCaption:
    def __init__(self, config: GenerateCaptionConfig):
        self.config = config
        configurationManager = ConfigurationManager()

        self.VEConfig:VideoEncodingConfig = configurationManager.get_video_encoding_config()
        self.video_encoder = VE(self.VEConfig)

        self.imgExtConfig:ImageExtractionSplitConfig = configurationManager.get_image_extraction_split_config()
        self.imgExt = ImageExtractionSplit(self.imgExtConfig)

        self.iproc = AutoImageProcessor.from_pretrained(self.config.vit_name, use_fast=True)

        self.vid_mask = torch.ones(1, self.config.FRAMES_PER_VIDEO, dtype=torch.long).to(self.config.DEVICE)
        self.load_models()

    def load_models(self):
        self.cfg = TrainCfg(
            frames_per_video = self.config.FRAMES_PER_VIDEO,
            device = self.config.DEVICE,
            vit_name = self.config.vit_name,
            gpt2_name = self.config.gpt2_name
        )
        
        self.tok = AutoTokenizer.from_pretrained(self.config.checkpoint_path / "tokenizer")

        # models
        self.model_enc = VideoEncoder(self.cfg.vit_name, self.cfg.d_model, self.cfg.proj_hidden, self.cfg.dropout).to(self.config.DEVICE)
        self.model_dec = CrossModalCaptioner(self.cfg.gpt2_name, self.cfg.d_model, self.cfg.dropout).to(self.config.DEVICE)

        # >>> CRITICAL: resize embeddings to match tokenizer <<<
        vocab_size = len(self.tok)
        self.model_dec.lm.resize_token_embeddings(vocab_size)
        self.model_dec.lm.config.pad_token_id = self.tok.pad_token_id
        self.model_dec.lm.config.eos_token_id = self.tok.eos_token_id

        if self.config.checkpoint_path.exists():
            load_checkpoint(self.config.checkpoint_path.__str__(), self.model_enc, self.model_dec)
        else:
            raise FileNotFoundError(f"The model file '{self.config.checkpoint_path}' was not found.")


    @torch.no_grad()
    def generate_caption(self, pixel_values, precomp_feat, prompt: str = ""):
        if pixel_values is not None: pixel_values = pixel_values.to(self.config.DEVICE)
        if precomp_feat is not None: precomp_feat = precomp_feat.to(self.config.DEVICE)
        video_tokens = self.model_enc(pixel_values, precomp_feat)

        # Build prompt ids (can be empty)
        if prompt == "":
            prompt = ""
        input_ids = self.tok.encode(prompt, return_tensors="pt").to(self.config.DEVICE)

        gen_ids = self.model_dec.lm.generate(
            input_ids=input_ids,
            max_new_tokens=self.cfg.max_gen_len,
            num_beams=self.cfg.num_beams,
            do_sample=self.cfg.do_sample,
            encoder_hidden_states=video_tokens,
            encoder_attention_mask=self.vid_mask,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id
        )
        out = self.tok.decode(gen_ids[0], skip_special_tokens=True)
        # strip the prompt
        return out[len(prompt):].strip()
    
    def generate(self, file_path: Path, model_path: Path=None):
        """
            args:
                - file_path: Path;
                    full path of the video
                    path of the feature.pt 
                    folder path of the images
                - model_path: Path;
                    model checkpoint path
            return:
                generated text
        """
        if model_path:
            self.load_models()
            self.config.checkpoint_path = here(model_path)
        if isinstance(file_path, str):
            file_path = Path(file_path)


        feat = None
        pixel_value = None
        if file_path.is_dir():
            if (file_path / 'features.pt').exists():
                feat = torch.load(file_path / 'features.pt')
            else:
                feat = self.video_encoder.encode_video_dir(file_path)
        elif file_path.suffix == '.pt': # video
            feat = torch.load(file_path)
        else: # video
            imgs = self.imgExt.frames_extraction(file_path)
            imgs = torch.tensor(imgs, dtype=torch.float32)
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs[:, [2, 1, 0], :, :]
            batch = self.iproc(images=imgs, return_tensors="pt")
            pixel_value = batch["pixel_values"].to(self.config.DEVICE)  # [B,C,H,W]

        caption = self.generate_caption(pixel_value, feat, prompt="caption: ")

        return caption
            

    def run(self, folder_path: Path=None, model_path: Path=None, generated_text_save_dir_path: Path=None):
        """
            args:
                - folder_path: Path; Root path of all features or videos
                - model_path: Path; model saved path
                - generated_text_save_dir_path: the location of the dir where the csv file will save.
        """
        if folder_path:
            self.config.data_dirs = [here(folder_path)]
        if model_path:
            self.config.checkpoint_path = here(model_path)
        if generated_text_save_dir_path:
            self.config.destination_dir = here(generated_text_save_dir_path)

        self.config.destination_dir.mkdir(parents=True, exist_ok=True)

        for dirs_path in self.config.data_dirs:
            file_name = dirs_path.stem
            csv_file_path = self.config.destination_dir / f"{file_name}.csv"
            df_dict = {"frame_dir": [], "caption": [], 'video_id':[]}
            if csv_file_path.exists():
                df = pd.read_csv(csv_file_path)
                df_dict["frame_dir"] = df["frame_dir"].to_list()
                df_dict["caption"] = df["caption"].to_list()
                df_dict["video_id"] = df["video_id"].to_list()
            # print(df_dict)
            for file_path in tqdm(sorted(dirs_path.glob('*'))):
                video_id = file_path.stem
                if video_id in df_dict["video_id"]:
                    continue
                caption = self.generate(file_path)

                df_dict["frame_dir"].append(file_path.__str__())
                df_dict["caption"].append(caption)
                df_dict["video_id"].append(video_id)
                with DelayedInterruptMainProcess():
                    df = pd.DataFrame(df_dict)
                    df.to_csv(csv_file_path)
            


if __name__ == "__main__":
    from src import logger

    # Caption Generation
    STAGE_NAME = "Caption Generation"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_generate_caption_config()
    generateCaption = GenerateCaption(config)
    # generateCaption.run()
    caption = generateCaption.generate(here("dataset/feature_dataset/val/b_BuSVZwq6M_1_9"))
    logger.info(f">>> Caption: {caption}")
    logger.info(f">>> stage {STAGE_NAME} completed")