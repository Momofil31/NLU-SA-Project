import torch
import nltk
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# nltk.download("punkt")
# nltk.download("movie_reviews")
# nltk.download("subjectivity")

RANDOM_SEED = 42
N_FOLDS = 5
N_FOLDS_BASELINE = 5
TRAIN_TEST_SPLIT = 0.2

PAD_TOKEN = 0

EPOCHS = 50
EPOCHS_PRETRAINED = 5

LR = 0.001
LR_PRETRAINED = 5e-5

SEQUENCE_MAX_LENGTHS = {
    "polarity": 512, 
    "subjectivity": 128,
    "polarity-filter": 512
    }

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
CLIP_GRADIENTS = 5

WEIGHTS_SAVE_PATH = "./weights"
STATS_SAVE_PATH = "./stats"

# models config
BiGRUAttention_config = {
    "model_name": "BiGRUAttention",
    "epochs": EPOCHS,
    "batch_size": 256,
    "lr": LR,
    "emb_size": 300,
    "hidden_size": 128,
    "out_size": 1,
    "num_layers": 2,
    "dropout_ratio": 0.5,
    "bidirectional": True,
    "attention": True,
    "att_hidden_size": 64,
    "clip_gradients": CLIP_GRADIENTS
}

BiGRU_config = {
    "model_name": "BiGRU",
    "epochs": EPOCHS,
    "batch_size": 256,
    "lr": LR,
    "emb_size": 300,
    "hidden_size": 128,
    "out_size": 1,
    "num_layers": 2,
    "dropout_ratio": 0.5,
    "bidirectional": True,
    "attention": False,
    "clip_gradients": CLIP_GRADIENTS
}

TextCNN_config = {
    "model_name": "TextCNN",
    "epochs": EPOCHS,
    "batch_size": 256,
    "lr": LR,
    "emb_size": 300,
    "out_size": 1,
    "filter_sizes": [3, 5, 7],
    "num_filters": [100, 100, 100],
    "dropout_ratio": 0.5,
    "clip_gradients": 0
}

Transformer_config = {
    "model_name": PRETRAINED_MODEL_NAME,
    "epochs": EPOCHS_PRETRAINED,
    "batch_size": 32,
    "lr": LR_PRETRAINED,
    "sequence_max_len": SEQUENCE_MAX_LENGTHS,
    "out_size": 1,
    "pretrained": True,
    "clip_gradients": 0
}
