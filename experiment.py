import wandb
import torch
import copy
from tqdm.auto import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
from baseline import BaselineExperiment

from models import BiGRU, TextCNN, TransformerClassifier
from utils import init_weights, removeObjectiveSents, load_pretrained_vectors
from settings import *
from data_processing import Lang, CustomDataset, TransformerDataset

from nltk.corpus import movie_reviews, subjectivity
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torchtext.vocab import FastText

class Experiment:
    def __init__(self, task="polarity", sjv_classifier=None, sjv_vectorizer=None):
        self.model_config = None
        self.ModelType = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.cost_fn = None
        self.task = task

        self.data_raw = None
        self.data_Y = None
        self.lang = None
        self.sjv_classifier = sjv_classifier
        self.sjv_vectorizer = sjv_vectorizer

    def prepare_data(self):
        if self.task == "polarity":
            # Load movie review dataset
            negative_fileids = movie_reviews.fileids('neg')
            positive_fileids = movie_reviews.fileids('pos')

            mr_neg = [{"document": list(movie_reviews.words(fileids=fileid)), "label": 0} for fileid in negative_fileids]
            mr_Y_neg = [0]*len(mr_neg)

            mr_pos = [{"document": list(movie_reviews.words(fileids=fileid)), "label": 1} for fileid in positive_fileids]
            mr_Y_pos = [1]*len(mr_pos)

            self.data_raw = mr_neg+mr_pos
            self.data_Y = mr_Y_neg + mr_Y_pos
            print("Total samples: ", len(self.data_raw))

        elif self.task == "subjectivity":
            obj_fileid = subjectivity.fileids()[0]  # plot.tok.gt9.5000
            subj_fileid = subjectivity.fileids()[1]  # quote.tok.gt9.5000

            obj_sents = subjectivity.sents(fileids=obj_fileid)
            subj_sents = subjectivity.sents(fileids=subj_fileid)

            self.data_raw = [{"document": sent, "label": 0} for sent in obj_sents]
            self.data_Y = [0]*len(obj_sents)

            self.data_raw += [{"document": sent, "label": 1} for sent in subj_sents]
            self.data_Y += [1]*len(subj_sents)
            print("Total samples: ", len(self.data_raw))

        elif (self.task == "polarity-filter"
              and self.sjv_classifier is not None
              and self.sjv_vectorizer is not None
              ):

            # get docs divided in sentences
            negative_fileids = movie_reviews.fileids('neg')
            positive_fileids = movie_reviews.fileids('pos')
            neg_docs_sents = [movie_reviews.sents(fileids=fileid) for fileid in negative_fileids]
            pos_docs_sents = [movie_reviews.sents(fileids=fileid) for fileid in positive_fileids]
            mr_docs_sents = neg_docs_sents + pos_docs_sents
            mr_sents = [" ".join(sent) for doc in mr_docs_sents for sent in doc]

            # shallow subjectivity classifier is used to allow comparisons
            movie_sjv_vectors = self.sjv_vectorizer.transform(mr_sents)
            pred = self.sjv_classifier.predict(movie_sjv_vectors)
            clean_mr = removeObjectiveSents(mr_docs_sents, pred, tokenized=True)

            mr_neg = [{"document": doc, "label": 0} for doc in clean_mr[:1000]]
            mr_Y_neg = [0]*len(mr_neg)

            mr_pos = [{"document": doc, "label": 1} for doc in clean_mr[1000:]]
            mr_Y_pos = [1]*len(mr_pos)

            self.data_raw = mr_neg+mr_pos
            self.data_Y = mr_Y_neg + mr_Y_pos
            print("Total samples: ", len(self.data_raw))

        else:
            print("Cannot prepare data. Wrong parameters.")
            exit()

    def create_fold(self):
        train, test, _, _ = train_test_split(self.data_raw, self.data_Y, test_size=TRAIN_TEST_SPLIT,
                                             random_state=RANDOM_SEED,
                                             shuffle=True,
                                             stratify=self.data_Y)

        words = [word for sample in train for word in sample["document"]]
        self.lang = Lang(words)
        train_dataset = CustomDataset(train, self.lang)
        test_dataset = CustomDataset(test, self.lang)

        self.train_loader = DataLoader(train_dataset, batch_size=self.model_config["batch_size"], collate_fn=train_dataset.collate_fn,  shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.model_config["batch_size"], collate_fn=test_dataset.collate_fn)

    def run(self):
        self.prepare_data()
        models = []
        metrics_list = []
        for fold_idx in range(N_FOLDS):
            self.create_fold()
            if self.lang:
                vocab_size = len(self.lang.word2id)
                model = self.ModelType(vocab_size, self.model_config)
                if self.model_config["pretrained_embeddings"]:
                    print("Loading pretrained word embeddings")
                    fast_text_embds = FastText('simple')
                    embeddings = torch.tensor(load_pretrained_vectors(self.lang.word2id, fast_text_embds), dtype=torch.float)
                    model.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=PAD_TOKEN)
            else:
                model = self.ModelType(self.model_config)

            model.to(DEVICE)

            run = wandb.init(
                project="NLU_SA",
                entity="filippomomesso",
                group=f"{self.model_config['model_name']}",
                name=f"{self.task}_{self.model_config['model_name']}_fold_{fold_idx:02d}",
                config={
                    "task": self.task,
                    **self.model_config,
                    "loss": "BCELoss",
                    "optimizer": "Adam"
                }
            )
            print(model)
            wandb.watch(model, "gradients", log_freq=5)
            self.optimizer = optim.Adam(model.parameters(), lr=run.config['lr'])
            self.cost_fn = torch.nn.BCEWithLogitsLoss()

            best_model, metrics = self.training_loop(model, self.train_loader, self.test_loader, run)
            models.append(best_model)
            metrics_list.append(metrics)

        # print average and std for metrics
        metrics_df = pd.DataFrame.from_dict(metrics_list)
        metrics_df.loc["mean"] = metrics_df[:N_FOLDS].mean()
        metrics_df.loc["std"] = metrics_df[:N_FOLDS].std()
        print(metrics_df)
        metrics_df.to_csv(f"{STATS_SAVE_PATH}/{self.model_config['model_name']}_{self.task}.csv")

        best_model_overall_idx = metrics_df["acc"].idxmax()
        return models[best_model_overall_idx]

    def training_loop(self, model, tr_dl, ts_dl, wandb_run, save=True):
        print(f"Runnig: {wandb_run.name}")

        # Check if model is pretrained to avoid initializing weights
        if not wandb_run.config.get("pretrained"):
            print("Model is not pretrained: initializing weigths.")
            model.apply(init_weights)

        optimizer = self.optimizer
        cost_fn = self.cost_fn

        best_loss = 0.
        best_acc = 0.

        print("Start training")
        for e in tqdm(range(wandb_run.config['epochs']), desc="Training Loop"):
            train_metrics = self.training_step(model, tr_dl, optimizer, cost_fn, clip=wandb_run.config["clip_gradients"], epoch=e)
            test_metrics = self.test_step(model, ts_dl, cost_fn, epoch=e)

            metrics = {**train_metrics, **test_metrics}
            wandb.log(metrics)

            train_loss = train_metrics['train/train_loss']
            train_acc = train_metrics['train/train_acc']

            test_loss = test_metrics['test/test_loss']
            test_acc = test_metrics['test/test_acc']
            test_f1 = test_metrics['test/test_f1']

            if best_acc < test_acc or e == 0:
                best_acc = test_acc
                best_loss = test_loss
                best_f1 = test_f1
                best_model = copy.deepcopy(model)
                # Save new best weights
                if save:
                    self.save_weights(e, model, optimizer, test_loss, f"{WEIGHTS_SAVE_PATH}/{wandb_run.name}.pth")
                    artifact = wandb.Artifact(f'{wandb_run.name}', type='model', metadata={**wandb_run.config, **metrics})
                    artifact.add_file(f"{WEIGHTS_SAVE_PATH}/{wandb_run.name}.pth")
                    wandb_run.log_artifact(artifact)

            print('\n Epoch: {:d}'.format(e + 1))
            print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_acc))
            print('\t Test loss {:.5f}, Test accuracy {:.2f}, Test F1 {:.2f}'.format(test_loss, test_acc, test_f1))
            print('-----------------------------------------------------')

        #visualize(best_model, ts_dl, wandb_run)
        print('\t BEST Test loss {:.5f}, Test accuracy {:.2f}, Test F1 {:.2f}'.format(best_loss, best_acc, best_f1))
        wandb.summary["test_best_loss"] = best_loss
        wandb.summary["test_best_accuracy"] = best_acc
        wandb.summary["test_best_f1"] = best_f1
        wandb.finish()
        best_metrics = {"loss": best_loss, "acc": best_acc, "f1": best_f1}
        return best_model, best_metrics

    def training_step(self, model, data_loader, optimizer, cost_function, clip=0, epoch=0):
        n_samples = 0
        cumulative_loss = 0.
        cumulative_accuracy = 0.

        model.train()

        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Training Step", leave=False)):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)

            loss = cost_function(outputs, targets.unsqueeze(-1).float())
            loss.backward()
            if clip != 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

            # add batch size
            n_samples += outputs.shape[0]
            # cumulative loss
            cumulative_loss += loss.item()

            # return predicted labels
            predicted = torch.sigmoid(outputs).round()

            # cumulative accuracy
            cumulative_accuracy += predicted.eq(targets.unsqueeze(-1)).sum().item()

        # avg loss and accuracy
        loss = cumulative_loss / n_samples
        acc = cumulative_accuracy / n_samples

        metrics = {
            "train/train_loss": loss,
            "train/train_acc": acc
        }

        return metrics

    def test_step(self, model, data_loader, cost_function, epoch=0):
        n_samples = 0
        cumulative_loss = 0.
        cumulative_accuracy = 0.

        model.eval()
        y_gt = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Test Step", leave=False)):
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(DEVICE)
                    targets = targets.to(DEVICE)
                outputs = model(inputs)
                loss = cost_function(outputs, targets.unsqueeze(-1).float())

                # add batch size
                n_samples += outputs.shape[0]
                # cumulative loss
                cumulative_loss += loss.item()
                # return predicted labels
                predicted = torch.sigmoid(outputs).round()

                y_pred += predicted.tolist()
                y_gt += targets.unsqueeze(-1).float().tolist()

                # cumulative accuracy
                cumulative_accuracy += predicted.eq(targets.unsqueeze(-1)).sum().item()

        # avg loss and accuracy
        loss = cumulative_loss / n_samples
        acc = cumulative_accuracy / n_samples
        f1 = f1_score(y_gt, y_pred)

        metrics = {
            "test/test_loss": loss,
            "test/test_acc": acc,
            "test/test_f1": f1
        }

        return metrics

    def save_weights(self,  epoch, model, optimizer, loss, path, scheduler=None):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None
        }
        torch.save(save_dict, path)

    def load_weights(self, model, optimizer, weights_path, DEVICE, scheduler=None):
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler'])

        return epoch, model, optimizer, scheduler


class TransformerExperiment(Experiment):
    def __init__(self, task="polarity", sjv_classifier=None, sjv_vectorizer=None, *args):
        super().__init__(task, sjv_classifier, sjv_vectorizer)
        self.model_config = Transformer_config
        self.ModelType = TransformerClassifier

    def create_fold(self):
        train, test, train_y, test_y = train_test_split(self.data_raw, self.data_Y, test_size=TRAIN_TEST_SPLIT,
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True,
                                                        stratify=self.data_Y)
        train_dataset = TransformerDataset(train, train_y, self.model_config, self.task)
        test_dataset = TransformerDataset(test, test_y, self.model_config, self.task)

        self.train_loader = DataLoader(train_dataset, batch_size=self.model_config["batch_size"],  shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.model_config["batch_size"])

    def prepare_data(self):
        BaselineExperiment.prepare_data(self)


class BiGRUExperiment(Experiment):
    def __init__(self, task="polarity", sjv_classifier=None, sjv_vectorizer=None, pretrained_embeddings=False):
        super().__init__(task, sjv_classifier, sjv_vectorizer)
        self.model_config = BiGRU_config
        self.ModelType = BiGRU
        self.model_config["pretrained_embeddings"] = pretrained_embeddings


class BiGRUAttentionExperiment(Experiment):
    def __init__(self, task="polarity", sjv_classifier=None, sjv_vectorizer=None, pretrained_embeddings=False):
        super().__init__(task, sjv_classifier, sjv_vectorizer)
        self.model_config = BiGRUAttention_config
        self.ModelType = BiGRU
        self.model_config["pretrained_embeddings"] = pretrained_embeddings


class TextCNNExperiment(Experiment):
    def __init__(self, task="polarity", sjv_classifier=None, sjv_vectorizer=None, pretrained_embeddings=False):
        super().__init__(task, sjv_classifier, sjv_vectorizer)
        self.model_config = TextCNN_config
        self.ModelType = TextCNN
        self.model_config["pretrained_embeddings"] = pretrained_embeddings
