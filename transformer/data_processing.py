import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset

from settings import BERT_SEQUENCE_MAX_LENGTH, DEVICE

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class BertDataset(Dataset):

    def __init__(self, dataset):
        self.documents = []
        self.labels = []
        for x in dataset[:10]:
            self.documents.append(x['document'])
            self.labels.append(x['label'])

        self.docs_tensor = [tokenizer(doc,
                                      padding='max_length', max_length=BERT_SEQUENCE_MAX_LENGTH, truncation=True,
                                      return_tensors="pt", is_split_into_words=True) for doc in self.documents]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        sample = {**self.docs_tensor[idx], 'label': label}
        return sample

    def collate_fn(self, data):
        new_item = {}

        for key in data[0].keys():
            new_item[key] = [d[key] for d in data]

        input_ids = torch.cat(new_item["input_ids"]).to(DEVICE)
        token_type_ids = torch.cat(new_item["token_type_ids"]).to(DEVICE)
        attention_mask = torch.cat(new_item["attention_mask"]).to(DEVICE)
        label = torch.LongTensor(new_item["label"]).to(DEVICE)

        return ((input_ids, token_type_ids, attention_mask), label)
