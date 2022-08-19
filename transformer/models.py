from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    '''https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f'''

    def __init__(self, config):

        super(BertClassifier, self).__init__()
        self.dropout_ratio = config["dropout_ratio"]
        self.out_size = config["out_size"]
        self
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.out_size)

    def forward(self, input):
        input_id, _, mask = input
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        return linear_output
