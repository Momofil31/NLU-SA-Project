from experiment import *
from baseline import BaselineExperiment
import argparse

nameToExperiment = {
    "BiGRU": BiGRUExperiment,
    "BiGRUAttention": BiGRUAttentionExperiment,
    "TextCNN": TextCNNExperiment,
    "AMCNN": AMCNNExperiment,
    "Transformer": TransformerExperiment,
    "Longformer": LongformerExperiment
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=nameToExperiment.keys(), help="Specify which experiment to run")
    parser.add_argument("task", choices=["polarity", "polarity-filter", "subjectivity"], help="Specify which task to perform")
    parser.add_argument("-pe", "--pretrained_embeddings", action="store_true", help="Specify if use pretrained embeddings")
    parser.add_argument("--truncation", choices=["head", "tail", "head-tail"], help="Specify document truncation strategy, to be used only with transformers.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging.")
    args = parser.parse_args()

    if args.experiment != "Transformer" and args.truncation:
        parser.error('Truncation strategy can be specified only with Transformer experiment.')

    sjv_classifier = None
    sjv_vectorizer = None
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    if args.task == "polarity-filter":
        # Run subjectivity
        exp_subjectivity = BaselineExperiment(task="subjectivity")
        sjv_classifier, sjv_vectorizer = exp_subjectivity.run()

    exp = nameToExperiment[args.experiment](args.task, sjv_classifier, sjv_vectorizer, pretrained_embeddings=args.pretrained_embeddings, truncation_strategy=args.truncation)
    best_model = exp.run()
