import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from settings import SEQUENCE_MAX_LENGTH, DEVICE, PRETRAINED_MODEL_NAME


class TransformerDataset(Dataset):

    def __init__(self, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        self.documents = []
        self.labels = []
        for x in dataset[:10]:
            self.documents.append(x['document'])
            self.labels.append(x['label'])

        self.docs_tensor = self.tokenizer(self.documents,
                                          padding='max_length', max_length=SEQUENCE_MAX_LENGTH, truncation=True,
                                          return_tensors="pt", is_split_into_words=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        sample = {'input_ids': self.docs_tensor["input_ids"][idx],
                  'attention_mask': self.docs_tensor["attention_mask"][idx]}
        if "token_type_ids" in self.docs_tensor.keys():
            sample["token_type_ids"] = self.docs_tensor["token_type_ids"][idx]
        return sample, label
