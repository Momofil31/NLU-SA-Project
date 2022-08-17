import torch
import nltk

nltk.download("punkt")
nltk.download("movie_reviews")
nltk.download("subjectivity")

N_FOLDS = 5
RANDOM_SEED = 42
BATCH_SIZE = 256
PAD_TOKEN = 0
TRAIN_TEST_SPLIT = 0.2
EPOCHS = 2
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# models config
SentimentGRU_config = {
    "emb_size": 300,
    "hidden_size": 128,
    "out_size": 1,
    "num_layers": 2,
    "dropout_ratio": 0.1,
    "bidirectional": True,
}

SentimentCNN_config = {
    "emb_size": 300,
    "out_size": 1,
    "filter_sizes": [3, 5, 7],
    "num_filters": [100, 100, 100],
}
