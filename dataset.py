# import des librairies
import torch
from torch.utils.data import Dataset

# Custom Dataset
class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_length=128):
        self.premises = list(premises)
        self.hypotheses = list(hypotheses)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        premise = str(self.premises[idx])
        hypothesis = str(self.hypotheses[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
