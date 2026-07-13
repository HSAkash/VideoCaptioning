from src.entity.config_entity import EvaluationConfig
from src.utils.commons import load_json_data, save_json_data
from src.utils.interrupt_check import DelayedInterruptMainProcess

import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from pathlib import Path
from src import logger
from tqdm import tqdm
from bert_score import BERTScorer

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

try:
    from nltk.translate import meteor_score
except ImportError:
    from nltk.translate.meteor import meteor_score
    

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.cider_scorer = Cider()
        self.spice_scorer = Spice()
        self.bert_scorer = BERTScorer(
            lang="en",
            model_type="roberta-large",
            device=self.config.DEVICE
        )

    @staticmethod
    def is_valid_caption(caption) -> bool:
        if not isinstance(caption, str):
            return False
        return bool(caption.strip())

    def calculate_bleu(self, predictions, references_list):
        """Calculate BLEU scores"""
        if self.config.verbose:
            logger.info(f">>> Calculating Bleu Score")
        smooth_fn = SmoothingFunction().method1
        bleu_scores = []

        with tqdm(total=len(references_list), desc="Bleu Score") as pbar:
            for pred, refs in zip(predictions, references_list):
                if not self.is_valid_caption(pred):
                    pbar.update(1)
                    continue
                pred_tokens = pred.split()
                refs_tokens = [ref.split() for ref in refs]
                score = sentence_bleu(refs_tokens, pred_tokens, 
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=smooth_fn)
                bleu_scores.append(score)
                pbar.update(1)
        
        bl_score = np.mean(bleu_scores, dtype=np.float32)
        if self.config.verbose:
            logger.info(f">>> Bleu Score: {bl_score}")
        
        return bl_score

    def calculate_rouge(self, predictions, references_list):
        """Calculate ROUGE-L scores"""
        if self.config.verbose:
            logger.info(f">>> Calculating ROUGE-L Score")

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        
        with tqdm(total=len(references_list), desc="ROUGE-L Score") as pbar:
            for pred, refs in zip(predictions, references_list):
                if not self.is_valid_caption(pred):
                    pbar.update(1)
                    continue
                best_score = 0
                for ref in refs:
                    scores = scorer.score(ref, pred)
                    rouge_l_f1 = scores['rougeL'].fmeasure
                    best_score = max(best_score, rouge_l_f1)
                rouge_scores.append(best_score)

                pbar.update(1)

        ro_score = np.mean(rouge_scores, dtype=np.float32)
        if self.config.verbose:
            logger.info(f">>> ROUGE-L Score: {ro_score}")
        
        return ro_score

    def calculate_meteor(self, predictions, references_list):
        """Calculate METEOR scores"""
        if self.config.verbose:
            logger.info(f">>> Calculating METEOR Score")
        meteor_scores = []
        
        with tqdm(total=len(references_list), desc="METEOR Score") as pbar:
            for pred, refs in zip(predictions, references_list):
                if not self.is_valid_caption(pred):
                    pbar.update(1)
                    continue
                pred_tokens = pred.split()
                refs_tokens = [ref.split() for ref in refs]
                best_score = 0
                for ref_tokens in refs_tokens:
                    try:
                        score = meteor_score.meteor_score([ref_tokens], pred_tokens)
                        best_score = max(best_score, score)
                    except:
                        score = 0
                meteor_scores.append(best_score)

                pbar.update(1)

        me_score = np.mean(meteor_scores, dtype=np.float32)
        if self.config.verbose:
            logger.info(f">>> METEOR Score: {me_score}")
        
        return me_score
    
    def calculate_cider(self, predictions, references_list):
        """Calculate CIDEr scores - THIS IS WHERE CIDEr IS!"""
        if self.config.verbose:
            logger.info(f">>> Calculating CIDEr Score")

        gts = {}
        res = {}
        
        for i, (pred, refs) in enumerate(zip(predictions, references_list)):
            if not self.is_valid_caption(pred):
                continue
            gts[i] = refs
            res[i] = [pred]

        if not res:
            return 0.0

        score, _ = self.cider_scorer.compute_score(gts, res)

        if self.config.verbose:
            logger.info(f">>> CIDEr Score: {score}")

        return score

    def calculate_spice(self, predictions, references_list, batch_size=500):
        """Calculate SPICE scores."""
        if self.config.verbose:
            logger.info(f">>> Calculating SPICE Score")

        total_score = 0.0
        total_count = 0

        with tqdm(total=len(references_list), desc="SPICE Score") as pbar:
            for start_idx in range(0, len(predictions), batch_size):
                preds_batch = predictions[start_idx:start_idx + batch_size]
                refs_batch = references_list[start_idx:start_idx + batch_size]
                batch_pairs = [
                    (pred, refs)
                    for pred, refs in zip(preds_batch, refs_batch)
                    if self.is_valid_caption(pred)
                ]

                if not batch_pairs:
                    pbar.update(len(preds_batch))
                    continue

                gts = {str(i): refs for i, (_, refs) in enumerate(batch_pairs)}
                res = {str(i): [pred] for i, (pred, _) in enumerate(batch_pairs)}

                score, _ = self.spice_scorer.compute_score(gts, res)
                total_score += score * len(batch_pairs)
                total_count += len(batch_pairs)
                pbar.update(len(preds_batch))

        sp_score = total_score / total_count if total_count else 0.0
        if self.config.verbose:
            logger.info(f">>> SPICE Score: {sp_score}")

        return sp_score

    def calculate_bert_score(self, predictions, references_list, batch_size = 256):
        """Calculate Bert scores """
        all_P = []
        all_R = []
        all_F1 = []

        for i in tqdm(range(0, len(predictions), batch_size)):
            preds_batch = predictions[i:i+batch_size]
            refs_batch = references_list[i:i+batch_size]

            P, R, F1 = self.bert_scorer.score(preds_batch, refs_batch)

            all_P.append(P.cpu())
            all_R.append(R.cpu())
            all_F1.append(F1.cpu())

        all_P = torch.cat(all_P)
        all_R = torch.cat(all_R)
        all_F1 = torch.cat(all_F1)

        return all_P.mean().item(), all_R.mean().item(), all_F1.mean().item()
    
    def get_references_prediction_list(self, ref_path:Path, ref_col:str, pred_path:Path, pred_col:str):
        ref_json_data = load_json_data(ref_path)
        pred_json_data = pd.read_json(pred_path)
        pred_json_data = pred_json_data.to_dict(orient='records')

        ref_dict = {item['video_id']: item[ref_col] for item in ref_json_data}

        references_list = []
        predictions = []

        for item in pred_json_data:
            video_id = item['video_id']
            pred = item.get(pred_col, None)
            if not self.is_valid_caption(pred):
                continue
            if ref_dict.get(video_id, None):
                predictions.append(pred.strip())
                references_list.append(ref_dict[video_id])

        return references_list, predictions
    
    def run(
        self,
        model_path: Path = None,
        reference_json_path:Path=None,
        reference_column:str=None,
        relatum_json_path:Path=None,
        relatum_column:str=None,
        save_path: Path=None,
        verbose: bool = None
        ):
        if model_path and reference_json_path and reference_column and relatum_json_path and relatum_column:
            item = [
                model_path.stem,
                relatum_json_path.stem,
                model_path,
                reference_json_path,
                reference_column,
                relatum_json_path,
                relatum_column
            ]
            self.config.files_details.append(item)
        if not verbose == None:
            self.config.verbose = verbose
        if save_path:
            self.config.save_path = save_path

        save_json_path = self.config.save_path.with_suffix('.json')
        save_json_path.parent.mkdir(parents=True, exist_ok=True)

        results = []

        if save_json_path.exists():
            results = load_json_data(save_json_path)
        
        for item in self.config.files_details:
            (
                model_label,
                dataset_label,
                model_path,
                reference_json_path,
                reference_column,
                relatum_json_path,
                relatum_column
            ) = item

            if self.config.verbose:
                logger.info(f">>>\nmodel_label: {model_label}\ndataset_label: {dataset_label}\nrelatum_column: {relatum_column}")

            references_list, predictions = self.get_references_prediction_list(
                reference_json_path,
                reference_column,
                relatum_json_path,
                relatum_column
            )

            current_item = None

            for idx, res_item in enumerate(results):
                if (
                    res_item['model_label'] == model_label and
                    res_item['dataset_label'] == dataset_label and
                    res_item['relatum_column'] == relatum_column
                ):
                    current_item = res_item
                    break
            if not current_item:
                current_item = {
                    'model_label': model_label,
                    'dataset_label': dataset_label,
                    'relatum_column': relatum_column
                }
                results.append(current_item)
                idx = -1

            if not current_item.get("bleu_score", None):
                bl_score = self.calculate_bleu(predictions, references_list)
                current_item['bleu_score'] = float(bl_score)
                results[idx] = current_item

                with DelayedInterruptMainProcess():
                    save_json_data(results, save_json_path)

            if not current_item.get("rouge_score", None):
                ro_score = self.calculate_rouge(predictions, references_list)
                current_item['rouge_score'] = float(ro_score)
                results[idx] = current_item

                with DelayedInterruptMainProcess():
                    save_json_data(results, save_json_path)

            if not current_item.get("meteor_score", None):
                me_score = self.calculate_meteor(predictions, references_list)
                current_item['meteor_score'] = float(me_score)
                results[idx] = current_item

                with DelayedInterruptMainProcess():
                    save_json_data(results, save_json_path)

            if not current_item.get("cider_score", None):
                ci_score = self.calculate_cider(predictions, references_list)
                current_item['cider_score'] = float(ci_score)
                results[idx] = current_item

                with DelayedInterruptMainProcess():
                    save_json_data(results, save_json_path)

            if "spice_score" not in current_item:
                sp_score = self.calculate_spice(predictions, references_list)
                current_item['spice_score'] = float(sp_score)
                results[idx] = current_item

                with DelayedInterruptMainProcess():
                    save_json_data(results, save_json_path)
            
            if not current_item.get("bert_precision", None):
                P, R, F1 = self.calculate_bert_score(predictions, references_list)
                current_item['bert_precision'] = float(P)
                current_item['bert_recall'] = float(R)
                current_item['bert_F1'] = float(F1)

                with DelayedInterruptMainProcess():
                    save_json_data(results, save_json_path)

        
        df = pd.read_json(save_json_path)
        if save_path:
            df.to_csv(self.config.save_path, index=False)
        else:
            logger.info(f"""Evaluation score:\n{results}""")


if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager

    # Evaluation
    STAGE_NAME = "Evaluation"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_evaluation_config()
    eval = Evaluation(config)
    eval.run()
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {config.save_path}")
