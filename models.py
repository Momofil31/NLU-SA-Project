import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from settings import PAD_TOKEN
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SoftAttention(nn.Module):
    '''
        Multilayer perception to learn attention coefficients. 
        As described in https://arxiv.org/pdf/1409.0473.pdf, Bengio et al. ICLR 2015
    '''

    def __init__(self, dim, dropout_ratio=0.1):
        super(SoftAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(dim, 1),
            nn.Softmax(dim=0)
        )

    def forward(self, context_vector):
        return self.attention(context_vector)


class BiGRU(nn.Module):
    '''
        Architecture based on the one seen during lab.
    '''

    def __init__(self, vocab_size, config, pad_index=0):
        super(BiGRU, self).__init__()
        self.emb_size = config["emb_size"]
        self.hidden_size = config["hidden_size"]
        self.out_size = config["out_size"]
        self.num_layers = config["num_layers"]
        self.dropout_ratio = config["dropout_ratio"]
        self.bidirectional = config["bidirectional"]
        self.attention = config["attention"]
        self.num_dir = 2 if self.bidirectional else 1

        if self.attention:
            self.att_hidden_size = config["att_hidden_size"]
            self.attention_module = SoftAttention(self.hidden_size*self.num_dir, dropout_ratio=self.dropout_ratio)
        if self.num_layers == 1:
            self.dropout_ratio = 0

        self.embedding = nn.Embedding(vocab_size, self.emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.GRU(self.emb_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout_ratio)
        self.fc1 = nn.Linear(self.hidden_size*self.num_dir, self.out_size)

    def forward(self, inputs):
        utterance, seq_lengths = inputs.values()
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
        # https://arxiv.org/pdf/1409.0473.pdf, Bengio et al. ICLR 2015
        if not self.attention:
            hidden_view = hidden.view(self.num_layers, self.num_dir, batch_size, self.hidden_size)  # 2 for bidirectional
            last_hidden = hidden_view[-1]  # get last layer forward and backward last hidden state

            if (self.utt_encoder.bidirectional):
                last_hidden_fwd = last_hidden[0]
                last_hidden_bwd = last_hidden[1]
                last_hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)
            out = self.fc1(last_hidden)
        else:
            # To get a better representation of the sequence it is possible to use a learnable soft attention mechanism
            # to perform a weighted summation of the encoded words similarly to:
            # Neural Machine Translation by jointly learning to align and translate, Bengio et. al. ICLR 2015.
            # https://arxiv.org/pdf/1409.0473.pdf
            alpha = self.attention_module(utt_encoded)
            context_vector = utt_encoded * alpha
            context_vector = context_vector.sum(dim=0)
            out = self.fc1(context_vector)
        return out


class TextCNN(nn.Module):
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
        super(TextCNN, self).__init__()
        self.emb_size = config["emb_size"]
        self.num_filters = config["num_filters"]
        self.filter_sizes = config["filter_sizes"]
        self.dropout_ratio = config["dropout_ratio"]
        self.out_size = config["out_size"]
        self.embedding = nn.Embedding(vocab_size, self.emb_size, padding_idx=PAD_TOKEN, max_norm=5.0)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.emb_size,
                      out_channels=self.num_filters[i],
                      kernel_size=self.filter_sizes[i])
            for i in range(len(self.filter_sizes))
        ])
        self.fc = nn.Linear(sum(self.num_filters), self.out_size)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, inputs):
        x, _ = inputs.values()
        x_emb = self.embedding(x).float()
        x_reshaped = x_emb.permute(0, 2, 1)
        x_conv_list = [torch.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [torch.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        logits = self.fc(self.dropout(x_fc))
        return logits


class TransformerClassifier(nn.Module):

    def __init__(self, config):
        super(TransformerClassifier, self).__init__()
        self.out_size = config["out_size"]
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=self.out_size,
            ignore_mismatched_sizes=True)

    def forward(self, input):
        return self.transformer(**input, return_dict=True).logits


class AMCNNAttention(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_channels: int = 1, mask_prob=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.in_features = in_features
        self.out_features = out_features
        self.mask_prob = mask_prob
        self.mask = nn.Dropout(self.mask_prob)

        self.w_l_lst = nn.ParameterList()
        self.b_l_lst = nn.ParameterList()
        self.w_v_l1_lst = nn.ParameterList()
        self.w_v_l2_lst = nn.ParameterList()
        self.b_v_l_lst = nn.ParameterList()

        for l in range(self.num_channels):
            self.w_l_lst.append(nn.parameter.Parameter(torch.empty((self.in_features, self.out_features))))
            self.b_l_lst.append(nn.parameter.Parameter(torch.empty((1, 1))))
            self.w_v_l1_lst.append(nn.parameter.Parameter(torch.empty((self.out_features, 1))))
            self.w_v_l2_lst.append(nn.parameter.Parameter(torch.empty((self.in_features, self.out_features))))
            self.b_v_l_lst.append(nn.parameter.Parameter(torch.empty((1, 1, self.out_features))))

    def forward(self, inputs):
        scalar_att_lst = []
        vector_att_lst = []
        C_l_lst = []
        maxlen = inputs.shape[1]
        for l in range(self.num_channels):  # l index for the channel
            # Scalar attention
            mat_l = torch.matmul(inputs, self.w_l_lst[l])
            mat_lj = torch.tile(mat_l.unsqueeze(1), (1, maxlen, 1, 1))
            mat_li = torch.tile(inputs.unsqueeze(1), (1, maxlen, 1, 1))
            mat_li = torch.permute(mat_li, (0, 2, 1, 3))
            M_l = torch.sum(mat_li * mat_lj, dim=3) + self.b_l_lst[l]  # Formula 4

            A_l = torch.tanh(self.mask(M_l))  # Formula 6

            s_lk = torch.sum(A_l, dim=2).unsqueeze(2)  # Formula 7
            score_lk = s_lk  # TODO add pad score
            a_l = torch.softmax(score_lk, dim=1)  # Formula 10
            scalar_att_lst.append(a_l)

            # Vectorial attention
            score_l_arrow = torch.sigmoid(torch.matmul(inputs, self.w_v_l2_lst[l])+self.b_v_l_lst[l])
            score_l_arrow = torch.matmul(score_l_arrow, self.w_v_l1_lst[l])
            a_l_arrow = torch.softmax(score_l_arrow, dim=1)
            vector_att_lst.append(a_l_arrow)

            new_inputs = torch.tile(torch.sum(inputs * a_l_arrow, dim=1).unsqueeze(1), (1, maxlen, 1))
            C_l = a_l * inputs + new_inputs

            C_l = C_l.unsqueeze(3)
            C_l_lst.append(C_l)
        C_features = torch.cat(C_l_lst, dim=3)
        return C_features


class AMCNN(nn.Module):
    def __init__(self, vocab_size, config, pad_index=0) -> None:
        super().__init__()
        self.emb_size = config["emb_size"]
        self.hidden_size = config["hidden_size"]
        self.out_size = config["out_size"]
        self.num_layers = config["num_layers"]
        self.dropout_ratio = config["dropout_ratio"]
        self.bidirectional = config["bidirectional"]

        self.num_dir = 2 if self.bidirectional else 1

        self.num_filters = config["num_filters"]
        self.filter_sizes = config["filter_sizes"]
        self.num_channels = config["num_channels"]
        self.maxlen = config["sequence_max_len"]
        self.features_size = self.hidden_size*self.num_dir

        self.embedding = nn.Embedding(vocab_size, self.emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.GRU(self.emb_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)
        self.attention = AMCNNAttention(self.features_size, self.features_size, self.num_channels)

        self.conv_block_lst = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.num_channels, out_channels=self.filter_sizes[i], kernel_size=(self.filter_sizes[i], self.maxlen)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.maxlen-self.filter_sizes[i]+1, 1)),
                nn.Flatten(),
            ) for i in range(len(self.filter_sizes))
        ])

        self.fc1 = nn.LazyLinear(self.out_size)

    def forward(self, inputs):
        utterance, seq_lengths = inputs.values()

        # utterance.size() = batch_size X seq_len
        batch_size = utterance.shape[0]
        utt_emb = self.embedding(utterance)  # utt_emb.size() = batch_size X seq_len X emb_size
        #print("Utt_emb", utt_emb.shape)
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (hidden) = self.utt_encoder(packed_input)
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)  # utt_encoded.shape = batch_size * seq_len * emb_size
        # print(utt_encoded.shape)

        C_features = self.attention(utt_encoded)  # batch * seq_len * feats * channels need to permute to feed it to conv2d
        C_features = torch.permute(C_features, dims=(0, 3, 1, 2))
        # print(C_features.shape)
        pools = []
        for conv_block in self.conv_block_lst:
            pools.append(conv_block(C_features))
            # print(pools[-1].shape)

        features = torch.cat(pools, -1)  # should be filter_size * num_filters
        # print(features.shape)
        out = self.fc1(features)
        return out
