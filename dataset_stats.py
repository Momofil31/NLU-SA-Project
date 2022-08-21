import nltk
from nltk.corpus import movie_reviews, subjectivity, stopwords
from experiment import Experiment
from settings import STATS_SAVE_PATH
from baseline import BaselineExperiment

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

def compute_stats(data, name):
    stats = {}
    seq_lens = [len(sents) for sents in data]
    stats["num_sequences"] = len(data)
    stats["num_words"] = sum([len(sent) for sents in data for sent in sents])
    stats["avg_seq_len"] = np.average(seq_lens)
    stats["max_seq_len"] = np.max(seq_lens)
    stats["min_seq_len"] = np.min(seq_lens)

    NLTK_STOP_WORDS = set(stopwords.words('english')+list(string.punctuation))
    lexicon = set([w for doc in data for w in doc])

    filtered_mr_words = [word for word in lexicon if not word in NLTK_STOP_WORDS]
    lexicon_filtered = set(filtered_mr_words)

    stats["lexicon_size"] = len(lexicon)
    stats["lexicon_size_no_stopwords"] = len(lexicon_filtered)
    return stats
    

if __name__ == "__main__":
    stats = {}

    # Movie review dataset
    negative_fileids = movie_reviews.fileids('neg')
    positive_fileids = movie_reviews.fileids('pos')

    # each is a list of documents
    mr_neg_words = [movie_reviews.words(fileids=fileid) for fileid in negative_fileids]
    mr_pos_words = [movie_reviews.words(fileids=fileid) for fileid in positive_fileids]
    mr_neg_sents = [movie_reviews.sents(fileids=fileid) for fileid in negative_fileids]
    mr_pos_sents = [movie_reviews.sents(fileids=fileid) for fileid in positive_fileids]

    mr_sents =  mr_neg_sents + mr_pos_sents
    mr_words = mr_neg_words + mr_pos_words

    stats["MR"] = compute_stats(mr_words, "MR")

    # Treating MR as subjectivity dataset (list of sentences)
    mr_sjv = [sent for doc in mr_sents for sent in doc]
    stats["MR_sjv"] = compute_stats(mr_sjv, "MR_SJV")

    # Subjectivity dataset
    obj_fileid = subjectivity.fileids()[0]  # plot.tok.gt9.5000 
    subj_fileid = subjectivity.fileids()[1] # quote.tok.gt9.5000
    obj_words = subjectivity.sents(fileids=obj_fileid)
    subj_words = subjectivity.sents(fileids=subj_fileid)
    sjv_words = obj_words + subj_words
    stats["SJV"] = compute_stats(sjv_words, "SJV")
    
    # Clean MR
    # Train baseline subjectivity classifier
    exp_subjectivity = BaselineExperiment(task="subjectivity")
    sjv_classifier, sjv_vectorizer = exp_subjectivity.run()
    mr_vectors = sjv_vectorizer.transform([" ".join(sent) for sent in mr_sjv])
    preds = sjv_classifier.predict(mr_vectors)

    # Remove objective sentences
    mr_sents_filtered = Experiment.removeObjectiveSents(mr_sents, preds)
    stats["MR_clean_baseline"] = compute_stats(mr_sents_filtered, "MR_clean_baseline")

    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.to_csv(f"{STATS_SAVE_PATH}/datasets.csv")
    print(stats_df)