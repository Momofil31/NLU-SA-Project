import torch.nn as nn


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def removeObjectiveSents(docs_sents, mask, tokenized=False):
        i = 0
        remaining_sents = 0
        clean_docs = []
        for doc in docs_sents:
            clean_docs.append([])
            for sent in doc:
                if mask[i] == 1:
                    clean_docs[-1] += sent
                    remaining_sents += 1
                i += 1
        print(f"Remaining {remaining_sents} sentences from original {i} sentences count.")
        if not tokenized:
            clean_docs = [" ".join(sents) for sents in clean_docs]
        return clean_docs