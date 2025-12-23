import pandas as pd
import matplotlib.pyplot as plt
from src.entity.config_entity import PlotHistoryConfig


class PlotHistory:
    def __init__(self, config:PlotHistoryConfig):
        self.config = config

    def extract_history(self):
        from src.components.trainingVideoEncoder import VideoEncoder
        from src.components.crossModalCaptioner import CrossModalCaptioner
        from src.utils.helperFunctions import load_checkpoint
        from src.config.training_config import TrainCfg
        from transformers import AutoTokenizer

        self.cfg = TrainCfg(
            frames_per_video = self.config.FRAMES_PER_VIDEO,
            device = self.config.DEVICE,
            vit_name = self.config.vit_name,
            gpt2_name = self.config.gpt2_name
        )
        
        self.tok = AutoTokenizer.from_pretrained(self.config.model_path / "tokenizer")

        # models
        self.model_enc = VideoEncoder(self.cfg.vit_name, self.cfg.d_model, self.cfg.proj_hidden, self.cfg.dropout).to(self.config.DEVICE)
        self.model_dec = CrossModalCaptioner(self.cfg.gpt2_name, self.cfg.d_model, self.cfg.dropout).to(self.config.DEVICE)

        # >>> CRITICAL: resize embeddings to match tokenizer <<<
        vocab_size = len(self.tok)
        self.model_dec.lm.resize_token_embeddings(vocab_size)
        self.model_dec.lm.config.pad_token_id = self.tok.pad_token_id
        self.model_dec.lm.config.eos_token_id = self.tok.eos_token_id

        if self.config.model_path.exists():
            ck = load_checkpoint(self.config.model_path.__str__(), self.model_enc, self.model_dec)
            history = ck['history']
            df = pd.DataFrame(history)
            self.config.history_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.config.history_path, index=False)

        else:
            raise FileNotFoundError(f"The model file '{self.config.model_path}' was not found.")
        
    def plot_single(self):
        for column in self.history.columns:
            save_path = (self.config.destination_dir / column).with_suffix(".png")
            y_label = "Loss"
            if column == 'epoch':
                continue
            if column == "valid_R1":
                y_label = "R1"
            loss = self.history[column].to_list()
            title = column.replace("_", " ")
            epochs = self.history['epoch'].to_list()

            # Plot loss
            plt.plot(epochs, loss)
            plt.title(title, fontsize=22)
            plt.xlabel('Epochs', fontsize=18)
            plt.ylabel(y_label, fontsize=18)
            plt.legend(fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
    
    def plot_multi(self, columns, title, save_path):
        save_path = self.config.destination_dir / save_path
        epochs = self.history['epoch'].to_list()
        for i, column in enumerate(columns):
            loss = self.history[column].to_list()
            plt.plot(epochs, loss, label=column, linewidth= 4 - i * 2)

        plt.title(title, fontsize=22)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_history(self):
        self.plot_single()
        # epoch,train_loss,train_loss_txt,train_loss_align,valid_loss,train_loss_txt,valid_loss_align,valid_R1
        plot_info = [
            {
                'columns': ['train_loss', 'valid_loss'],
                'title': "Loss",
                'save_path': "loss.png" 
            },
            {
                'columns': ['train_loss_txt', 'valid_loss_txt'],
                'title': "Text loss",
                'save_path': "text_loss.png" 
            },
            {
                'columns': ['train_loss_align', 'valid_loss_align'],
                'title': "Align loss",
                'save_path': "align_loss.png" 
            },
        ]
        for item in plot_info:
            self.plot_multi(
                columns = item['columns'],
                title = item['title'],
                save_path = item['save_path']
            )

        

    def run(self, history_path:Path=None):
        if history_path:
            self.config.history_path = history_path
            
        if not self.config.history_path.exists():
            self.extract_history()
        self.history = pd.read_csv(self.config.history_path)
        self.config.destination_dir.mkdir(parents=True, exist_ok=True)
        self.plot_history()


if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager
    from src import logger

    # Plot history
    STAGE_NAME = "Plot History"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_plot_history_config()
    eval = PlotHistory(config)
    eval.run()
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {config.destination_dir}")
        
    