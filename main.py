from experiment import Experiment, TransformerExperiment
from baseline import BaselineExperiment

if __name__ == "__main__":
    filter = False
    # if filter:
    #     # Run subjectivity
    #     exp_subjectivity = BaselineExperiment(task="subjectivity")
    #     sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
    #     exp = TransformerExperiment("Transformer", "polarity-filter", sjv_classifier, sjv_vectorizer)
    # else:
    #     exp = TransformerExperiment("Transformer", "subjectivity")
    # best_model = exp.run()

    # if filter:
    #     # Run subjectivity
    #     exp_subjectivity = BaselineExperiment(task="subjectivity")
    #     sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
    #     exp = Experiment("BiGRU", "polarity-filter", sjv_classifier, sjv_vectorizer)
    # else:
    #     exp = Experiment("BiGRU", "polarity")
    # best_model = exp.run()

    # if filter:
    #     # Run subjectivity
    #     exp_subjectivity = BaselineExperiment(task="subjectivity")
    #     sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
    #     exp = Experiment("BiGRUAttention", "polarity-filter", sjv_classifier, sjv_vectorizer)
    # else:
    #     exp = Experiment("BiGRUAttention", "subjectivity")
    # best_model = exp.run()

    if filter:
        # Run subjectivity
        exp_subjectivity = BaselineExperiment(task="subjectivity")
        sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
        exp = Experiment("TextCNN", "polarity-filter", sjv_classifier, sjv_vectorizer)
    else:
        exp = Experiment("TextCNN", "subjectivity")
    best_model = exp.run()