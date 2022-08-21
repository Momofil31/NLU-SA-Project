from experiment import *
from baseline import BaselineExperiment

if __name__ == "__main__":
    filter = True
    if filter:
        # Run subjectivity
        exp_subjectivity = BaselineExperiment(task="subjectivity")
        sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
        exp = TransformerExperiment("polarity-filter", sjv_classifier, sjv_vectorizer)
    else:
        exp = TransformerExperiment("subjectivity")
    best_model = exp.run()

    # if filter:
    #     # Run subjectivity
    #     exp_subjectivity = BaselineExperiment(task="subjectivity")
    #     sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
    #     exp = BiGRUExperiment("polarity-filter", sjv_classifier, sjv_vectorizer)
    # else:
    #     exp = BiGRUExperiment("polarity")
    # best_model = exp.run()

    # if filter:
    #     # Run subjectivity
    #     exp_subjectivity = BaselineExperiment(task="subjectivity")
    #     sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
    #     exp = BiGRUAttentionExperiment("polarity-filter", sjv_classifier, sjv_vectorizer)
    # else:
    #     exp = BiGRUAttentionExperiment("subjectivity")
    # best_model = exp.run()

    # if filter:
    #     # Run subjectivity
    #     exp_subjectivity = BaselineExperiment(task="subjectivity")
    #     sjv_classifier, sjv_vectorizer = exp_subjectivity.run()  
    #     exp = TextCNNExperiment("polarity-filter", sjv_classifier, sjv_vectorizer)
    # else:
    #     exp = TextCNNExperiment("subjectivity")
    # best_model = exp.run()