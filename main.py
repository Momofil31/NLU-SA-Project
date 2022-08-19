from experiment import Experiment, BertExperiment
from baseline import BaselineExperiment

if __name__ == "__main__":

    if filter:
        # Run subjectivity
        exp_subjectivity = BaselineExperiment(task="subjectivity")
        sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
        exp = BertExperiment("BertBase", "polarity-no-obj-sents", sjv_classifier, sjv_vectorizer)
    else:
        exp = BertExperiment("BertBase", "polarity")
    best_model = exp.run()