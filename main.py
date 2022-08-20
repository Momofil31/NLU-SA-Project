from experiment import Experiment, TransformerExperiment
from baseline import BaselineExperiment

if __name__ == "__main__":
    filter = False
    if filter:
        # Run subjectivity
        exp_subjectivity = BaselineExperiment(task="subjectivity")
        sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
        exp = TransformerExperiment("Transformer", "polarity-no-obj-sents", sjv_classifier, sjv_vectorizer)
    else:
        exp = TransformerExperiment("Transformer", "polarity")
    best_model = exp.run()