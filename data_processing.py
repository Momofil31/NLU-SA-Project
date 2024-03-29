from settings import DEVICE, PAD_TOKEN
from collections import Counter
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize


class Lang():
    def __init__(self, words, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'<pad>': PAD_TOKEN}
        if unk:
            vocab['<unk>'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab


class CustomDataset (Dataset):
    def __init__(self, dataset, lang, unk='<unk>', max_len=None):
        self.documents = []
        self.labels = []
        self.unk = unk
        self.max_len = max_len

        for x in dataset:
            self.documents.append(x['document'])
            self.labels.append(x['label'])

        self.docs_ids = self.mapping_seq(self.documents, lang.word2id)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = torch.tensor(self.docs_ids[idx])
        label = torch.tensor(self.labels[idx])
        sample = {'document': doc, 'label': label}
        return sample

    # Auxiliary methods
    def mapping_seq(self, data, mapper):  # Map sequences to number
        res = []
        if self.max_len:
            for doc in data:
                tmp_doc = []
                for i, x in enumerate(doc):
                    if (i >= self.max_len):
                        break
                    if x in mapper:
                        tmp_doc.append(mapper[x])
                    else:
                        tmp_doc.append(mapper[self.unk])
                if len(tmp_doc) < self.max_len:
                    tmp_doc += [PAD_TOKEN]*(self.max_len - len(tmp_doc))
                res.append(tmp_doc)
        else:
            for doc in data:
                tmp_doc = []
                for i, x in enumerate(doc):
                    if x in mapper:
                        tmp_doc.append(mapper[x])
                    else:
                        tmp_doc.append(mapper[self.unk])
                res.append(tmp_doc)
        return res

    def collate_fn(self, data):
        def merge(sequences):
            '''
            merge from batch * sent_len to batch * max_len 
            '''
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            # Pad token is zero in our case
            # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
            # batch_size X maximum length of a sequence
            padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
            # print(padded_seqs)
            padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
            return padded_seqs, lengths
        # Sort data by seq lengths
        data.sort(key=lambda x: len(x['document']), reverse=True)
        new_item = {}
        for key in data[0].keys():
            new_item[key] = [d[key] for d in data]
        # We just need one length for packed pad seq, since len(utt) == len(slots)
        src_docs, lenghts = merge(new_item['document'])
        label = torch.LongTensor(new_item["label"])
        text_lens = torch.LongTensor(lenghts)
        return ({"document": src_docs, "text_lens": text_lens}, label)


class TransformerDataset(Dataset):

    def __init__(self, documents, labels, config, task):
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
        self.documents = documents
        self.labels = labels
        max_len = config["sequence_max_len"][task]

        if config.get("truncation_strategy", "head") == "head-tail":
            # Truncation heuristic as in 5.3.1 Truncation Methods
            # https://arxiv.org/pdf/1905.05583.pdf
            # Getting first quarted + last three quarters of the document minus 2 token for [CLS] and [SEP] tokens

            docs = [word_tokenize(doc) for doc in self.documents]
            docs = [doc[:max_len//4]+doc[len(doc)-max_len//4*3+2:] if len(doc) > max_len else doc for doc in docs ]
            self.documents = [" ".join(doc) for doc in docs]
        else:
            if config.get("truncation_strategy", "head") == "tail":
                self.tokenizer.truncation_side = 'left'

        self.docs_tensor = self.tokenizer(self.documents,
                                              padding='max_length',
                                              max_length=max_len,
                                              truncation=True,
                                              return_tensors="pt")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        sample = {'input_ids': self.docs_tensor["input_ids"][idx],
                  'attention_mask': self.docs_tensor["attention_mask"][idx]}
        if "token_type_ids" in self.docs_tensor.keys():
            sample["token_type_ids"] = self.docs_tensor["token_type_ids"][idx]
        return sample, label
