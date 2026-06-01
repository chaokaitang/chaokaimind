from torch.utils.data import Dataset
from datasets import load_dataset
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.samples = load_dataset("json", data_files=data_path, split="train")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone() 
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids,  "attention_mask": attention_mask, "labels": labels}

        
