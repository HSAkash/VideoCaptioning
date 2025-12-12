import os
import json
import math
import requests
import pandas as pd
from tqdm import tqdm
from pyprojroot import here
from pathlib import Path
from dotenv import load_dotenv
from src.utils.interrupt_check import DelayedInterruptMainProcess
load_dotenv()

DEEPSEEK_API_KEY= os.getenv("DEEPSEEK_API_KEY")

class CloudRefiner:
    def __init__(self):
        pass


    def refine_sentences(self, sentences):
        """
        Refine multiple sentences to be more concise using DeepSeek API
        """
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        prompt = f"""
        Refine these sentences to be clear, grammatically correct, and concise. 
        Keep the original meaning but make them flow better.
        
        Sentences to refine:
        {sentences}
        
        Return ONLY the refined sentences in the same order, one per line.
        No numbering, no explanations, just the cleaned sentences.
        """
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            refined_text = result['choices'][0]['message']['content']
            
            # Split into individual sentences
            refined_sentences = [s.strip() for s in refined_text.split('\n') if s.strip()]
            
            return refined_sentences
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def single_sentence_refine(self, sentence: str) -> str:
        caption = self.refine_sentences(f"{sentence}\n{sentence}")
        # return ' '.join(caption)
        return caption[0]
        
    
    def run(self, source_path: Path, destination_path: Path):
        if isinstance(source_path, str):
            source_path = here(source_path)
        if isinstance(destination_path, str):
            destination_path

        destination_path.parent.mkdir(parents=True, exist_ok=True)

        source_df = pd.read_csv(source_path)
        source_json_data = source_df.to_dict(orient='records')
        destination_json_file_path = destination_path.with_suffix('.json')

        if destination_path.exists():
            return
        
        refine_data_dict = {
            'video_id': [],
            'frame_dir': [],
            'caption': [],
            'refine_caption': []
        }

        if destination_json_file_path.exists():
            with open(destination_json_file_path, 'r') as file:
                refine_data_dict = json.load(file)
        batch_size = 20
        total_iterations = math.ceil(len(source_json_data)/ batch_size)
        already_done_batch_count = math.ceil(len(refine_data_dict['frame_dir']) / batch_size)
        with tqdm(total=total_iterations, desc="Processing data") as pbar: 
            temp_source_json_data = source_json_data.copy()
            if already_done_batch_count:
                pbar.update(already_done_batch_count)
            while True:
                with DelayedInterruptMainProcess():
                    new_tem_source = []
                    for item in temp_source_json_data:
                        if item['frame_dir'] in refine_data_dict['frame_dir']:
                            continue
                        new_tem_source.append(item)
                    temp_source_json_data = new_tem_source.copy()
                    if len(temp_source_json_data) == 0:
                        final_df = pd.read_json(destination_json_file_path)
                        final_df.to_csv(destination_path, index=False)
                        break
                    items = temp_source_json_data[:batch_size]
                    input_sentences = [item['caption'] for item in items]
                    # Join sentences with newlines
                    sentences_text = "\n".join(input_sentences)
                    # Get refined sentences
                    refined = self.refine_sentences(sentences_text)
                    if refined:
                        for item, refine_caption in zip(items, refined):
                            refine_data_dict['frame_dir'].append(item['frame_dir'])
                            refine_data_dict['caption'].append(item['caption'])
                            refine_data_dict['video_id'].append(item['video_id'])
                            refine_data_dict['refine_caption'].append(refine_caption)
                        with open(destination_json_file_path, "w") as f:
                            json.dump(refine_data_dict, f, indent=4)
                pbar.update(1)


if __name__ == "__main__":
    from src import logger
    refiner = CloudRefiner()
    caption = "is a plastic bag is making a plastic bag and is being cut to make a plastic bag is taking out of how to make a plastic bag for a"
    refine_caption = refiner.single_sentence_refine(caption)
    logger.info(f">>> Caption: {caption}")
    logger.info(f">>> Refine Caption: {refine_caption}")