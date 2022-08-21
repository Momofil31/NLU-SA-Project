from settings import N_FOLDS_BASELINE

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from nltk.corpus import movie_reviews, subjectivity
import numpy as np
import pandas as pd
import itertools


class BaselineExperiment:

    def __init__(self, task="polarity", sjv_classifier=None, sjv_vectorizer=None):
        self.task = task
        self.data_raw = None
        self.data_Y = None
        self.sjv_classifier = sjv_classifier
        self.sjv_vectorizer = sjv_vectorizer

    @staticmethod
    def removeObjectiveSents(docs_sents, mask):
        i = 0
        remaining_sents = 0
        clean_docs = []
        for doc in docs_sents:
            clean_docs.append([])
            for sent in doc:
                if mask[i] == 1:
                    clean_docs[-1] += sent
                    remaining_sents += 1
                i += 1
        clean_docs = [" ".join(sents) for sents in clean_docs]
        print(f"Remaining {remaining_sents} sentences from original {i} sentences count.")
        return clean_docs

    def prepare_data(self):
        print("Loading data")
        if self.task == "polarity":
            negative_fileids = movie_reviews.fileids('neg')
            positive_fileids = movie_reviews.fileids('pos')
            neg_raw = [movie_reviews.raw(fileids=fileid) for fileid in negative_fileids]
            pos_raw = [movie_reviews.raw(fileids=fileid) for fileid in positive_fileids]
            self.data_raw = neg_raw + pos_raw
            self.data_Y = [0] * len(neg_raw) + [1] * len(pos_raw)

        elif self.task == "subjectivity":
            obj_fileid = subjectivity.fileids()[0]  # plot.tok.gt9.5000
            subj_fileid = subjectivity.fileids()[1]  # quote.tok.gt9.5000

            # this to avoid splitting words into lists
            obj_raw = subjectivity.raw(fileids=obj_fileid).split('\n')[:5000]
            subj_raw = subjectivity.raw(fileids=subj_fileid).split('\n')[:5000]
            self.data_raw = obj_raw + subj_raw
            self.data_Y = [0] * len(obj_raw) + [1] * len(subj_raw)
        elif (self.task == "polarity-filter"
              and self.sjv_classifier is not None
              and self.sjv_vectorizer is not None
              ):

            # get docs divided in sentences
            negative_fileids = movie_reviews.fileids('neg')
            positive_fileids = movie_reviews.fileids('pos')
            mr_neg_sents = [movie_reviews.sents(fileids=fileid) for fileid in negative_fileids]
            mr_pos_sents = [movie_reviews.sents(fileids=fileid) for fileid in positive_fileids]
            mr_corpus = mr_neg_sents + mr_pos_sents
            mr_sents = [" ".join(sent) for doc in mr_corpus for sent in doc]

            mr_sjv_vectors = self.sjv_vectorizer.transform(mr_sents)
            pred = self.sjv_classifier.predict(mr_sjv_vectors)

            self.data_raw = BaselineExperiment.removeObjectiveSents(mr_corpus, pred)
            self.data_Y = [0] * len(negative_fileids) + [1] * len(positive_fileids)
        else:
            print("Cannot prepare data. Wrong parameters.")
        print("Total samples: ", len(self.data_raw))

    def run(self):
        print(f"Running experiment {self.task} classification.")
        self.prepare_data()
        vectorizer = CountVectorizer()
        classifier = MultinomialNB()
        vectors = vectorizer.fit_transform(self.data_raw)
        scores = cross_validate(classifier, vectors, self.data_Y, cv=StratifiedKFold(n_splits=N_FOLDS_BASELINE), scoring=['accuracy', 'f1'], return_estimator=True)
        best_model = scores["estimator"][np.argmax(scores["test_accuracy"])]

        metrics_df = pd.DataFrame.from_dict(scores)
        metrics_df.drop("estimator", axis='columns', inplace=True)
        metrics_df.loc["mean"] = metrics_df[:N_FOLDS_BASELINE].mean()
        metrics_df.loc["std"] = metrics_df[:N_FOLDS_BASELINE].std()
        print(metrics_df)
        metrics_df.to_csv(f"stats/baseline_{self.task}_stats.csv")

        return best_model, vectorizer


if __name__ == "__main__":
    # Run polarity on whole movie review dataset
    exp_polarity = BaselineExperiment(task="polarity")
    exp_polarity.run()

    # Run subjectivity
    exp_subjectivity = BaselineExperiment(task="subjectivity")
    sjv_classifier, sjv_vectorizer = exp_subjectivity.run()

    # Run polarity on movie review dataset removing objective sentences
    exp = BaselineExperiment(task="polarity-filter", sjv_classifier=sjv_classifier, sjv_vectorizer=sjv_vectorizer)
    exp.run()
