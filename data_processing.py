from settings import DEVICE, PAD_TOKEN
from collections import Counter
from torch.utils.data import Dataset
import torch


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
    def __init__(self, dataset, lang, unk='<unk>'):
        self.documents = []
        self.labels = []
        self.unk = unk

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
        for doc in data:
            tmp_doc = []
            for x in doc:
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
        src_docs = src_docs.to(DEVICE)  # We load the Tensor on our seleceted device
        label = label.to(DEVICE)
        text_lens.to(DEVICE)

        return ((src_docs, text_lens), label)
