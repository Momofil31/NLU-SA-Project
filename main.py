from experiment import Experiment

if __name__ == "__main__":
    exp = Experiment("SentimentGRU", "polarity")
    best_model = exp.run()