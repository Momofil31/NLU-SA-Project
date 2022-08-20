from torch import nn
from transformers import AutoModelForSequenceClassification
from settings import PRETRAINED_MODEL_NAME


class TransformerClassifier(nn.Module):
    '''https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f'''

    def __init__(self, config):
        super(TransformerClassifier, self).__init__()
        self.out_size = config["out_size"]
        self.transformer = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=self.out_size)

    def forward(self, input):
        logits = self.transformer(**input, return_dict=True).logits
        return logits
