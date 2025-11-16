from typing import List
from transformers import GPT2TokenizerFast

class BatchTokenizer:
    def __init__(self, gpt2_name: str, max_len: int):
        self.tok = GPT2TokenizerFast.from_pretrained(gpt2_name)
        # ensure special tokens
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({'pad_token': '<|pad|>'})
        if self.tok.eos_token is None:
            self.tok.add_special_tokens({'eos_token': ''})
        self.max_len = max_len

    def __call__(self, caps: List[str]):
        enc = self.tok(
            caps,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=True
        )
        # labels for teacher forcing (ignore pad token id)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"], enc["attention_mask"], labels, self.tok