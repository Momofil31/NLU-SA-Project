import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from settings import PAD_TOKEN


class SentimentGRU(nn.Module):
    '''
        Architecture based on the one seen during lab
    '''

    def __init__(self, vocab_size, config, pad_index=0):
        super(SentimentGRU, self).__init__()
        self.emb_size = config["emb_size"]
        self.hidden_size = config["hidden_size"]
        self.out_size = config["out_size"]
        self.num_layers = config["num_layers"]
        self.dropout_ratio = config["dropout_ratio"]
        self.bidirectional = config["bidirectional"]

        self.num_dir = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, self.emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.GRU(self.emb_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout_ratio)
        self.fc1 = nn.Linear(self.hidden_size*self.num_dir, self.out_size)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        batch_size = utterance.shape[0]
        utt_emb = self.embedding(utterance)  # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = utt_emb.permute(1, 0, 2)  # we need seq len first -> seq_len X batch_size X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        # Process the batch
        packed_output, (hidden) = self.utt_encoder(packed_input)
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)

        # Using last hidden state will give worse results since the RNN forgets
        # older important words and the hidden state representation won't account for them.
        # The RNN in fact encodes the entire sequence in the final hidden state
        # which can cause information loss as all information needs to be compressed into c.
        # "A potential issue with this encoder–decoder approach is that a neural network
        # needs to be able to compress all the necessary information of a source sentence into a fixed-length vector.
        # This may make it difficult for the neural network to cope with long sentences,
        # especially those that are longer than the sentences in the training corpus."

        # hidden_view = hidden.view(self.num_layers, self.num_dir, batch_size, self.hidden_size) # 2 for bidirectional
        # last_hidden = hidden_view[-1] # get last layer forward and backward last hidden state

        # if (self.utt_encoder.bidirectional):
        #     last_hidden_fwd = last_hidden[0]
        #     last_hidden_bwd = last_hidden[1]
        #     last_hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), dim = 1)

        # To solve the problem just sum over the outputs to get a representation a better representation of the entire sequence
        # It is possible also to employ an learnable attention mechanism to weight the summation similarly to:
        # Neural Machine Translation by jointly learning to align and translate, Bengio et. al. ICLR 2015.
        # https://arxiv.org/pdf/1409.0473.pdf
        utt_encoded = utt_encoded.sum(dim=0)
        out = self.fc1(utt_encoded)
        return out


class SentimentCNN(nn.Module):
    '''
        Architecture based on: 
        Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification. 
        In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 
        pages 1746–1751, Doha, Qatar. Association for Computational Linguistics.

        and 

        Ye Zhang, Byron Wallace. 2015. A Sensitivity Analysis of (and Practitioners' Guide to) 
        Convolutional Neural Networks for Sentence Classification.
        https://arxiv.org/abs/1510.03820

    '''

    def __init__(self, vocab_size, config):
        super(SentimentCNN, self).__init__()
        self.emb_size = config["emb_size"]
        self.hidden_size = config["hidden_size"]
        self.num_filters = config["num_filters"]
        self.filter_sizes = config["filter_sizes"]
        self.dropout_ratio = config["dropout_ratio"]

        self.embedding = nn.Embedding(vocab_size, self.emb_size, padding_idx=PAD_TOKEN, max_norm=5.0)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.emb_dim,
                      out_channels=self.num_filters[i],
                      kernel_size=self.filter_sizes[i])
            for i in range(len(self.filter_sizes))
        ])
        self.fc = nn.Linear(sum(self.num_filters), self.out_size)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x, _):
        x_emb = self.embedding(x).float()
        x_reshaped = x_emb.permute(0, 2, 1)
        x_conv_list = [torch.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [torch.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        logits = self.fc(self.dropout(x_fc))
        return logits