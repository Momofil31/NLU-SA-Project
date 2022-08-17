import wandb
import torch
import copy
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd

from models.models import SentimentGRU, SentimentCNN
from utils import init_weights
from settings import N_FOLDS, EPOCHS, BATCH_SIZE, LR, RANDOM_SEED, TRAIN_TEST_SPLIT, SentimentGRU_config, SentimentCNN_config
from data_processing import Lang, CustomDataset, collate_fn

from nltk.corpus import movie_reviews, subjectivity
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class Experiment:
    def __init__(self, model_name, task="polarity", sjv_classifier=None, sjv_vectorizer=None):
        self.model_name = model_name
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

        if model_name == "SentimentGRU":
            self.model_config = SentimentGRU_config
        if model_name == "SentimentCNN":
            self.model_config = SentimentCNN_config

    def prepare_data(self):
        if self.task == "polarity":
            # Load movie review dataset
            negative_fileids = movie_reviews.fileids('neg')
            positive_fileids = movie_reviews.fileids('pos')

            mr_neg = [{"document": movie_reviews.words(fileids=fileid), "label": 0} for fileid in negative_fileids]
            mr_Y_neg = [0]*len(mr_neg)

            mr_pos = [{"document": movie_reviews.words(fileids=fileid), "label": 1} for fileid in positive_fileids]
            mr_Y_pos = [1]*len(mr_pos)

            self.data_raw = mr_neg+mr_pos
            self.data_Y = mr_Y_neg + mr_Y_pos
            print("Total samples: ", len(self.data_raw))

        elif self.task == "subjectivity":
            obj_fileid = subjectivity.fileids()[0]  # plot.tok.gt9.5000
            subj_fileid = subjectivity.fileids()[1]  # quote.tok.gt9.5000

            obj_sents = subjectivity.sents(fileids=obj_fileid)
            subj_sents = subjectivity.sents(fileids=subj_fileid)

            print(obj_sents[0])
            print(subj_sents[0])

            self.data_raw = [{"document": sent, "label": 0} for sent in obj_sents]
            self.data_Y = [0]*len(obj_sents)

            self.data_raw += [{"document": sent, "label": 1} for sent in subj_sents]
            self.data_Y += [1]*len(subj_sents)
            print("Total samples: ", len(self.data_raw))

        elif (self.task == "polarity-no-obj-sents"
            and self.sjv_classifier is not None
            and self.sjv_vectorizer is not None
            ):
            def removeObjectiveSents(docs_sents, mask):
                i = 0
                clean_docs = []
                for doc in docs_sents:
                    clean_docs.append([])
                    for sent in doc:
                        if mask[i] == 1:
                            clean_docs[-1] += sent
                        i += 1
                return clean_docs

            # get docs divided in sentences
            neg_docs_sents = [movie_reviews.sents(fileids=negative_fileids[id]) for id in range(len(negative_fileids))]
            pos_docs_sents = [movie_reviews.sents(fileids=positive_fileids[id]) for id in range(len(positive_fileids))]
            mr_docs_sents = neg_docs_sents + pos_docs_sents
            mr_sents = [" ".join(sent) for doc in mr_docs_sents for sent in doc]

            # shallow subjectivity classifier is used to allow comparisons
            movie_sjv_vectors = self.sjv_vectorizer.transform(mr_sents)
            pred = self.sjv_classifier.predict(movie_sjv_vectors)
            clean_mr = removeObjectiveSents(mr_docs_sents, pred)

            mr_neg = [{"document": doc, "label": 0} for doc in clean_mr[:1000]]
            mr_Y_neg = [0]*len(mr_neg)

            mr_pos = [{"document": doc, "label": 1} for doc in clean_mr[1000:]]
            mr_Y_pos = [1]*len(mr_pos)

            self.data_raw = mr_neg+mr_pos
            self.data_Y = mr_Y_neg + mr_Y_pos
            print("Total samples: ", len(self.data_raw))

        else:
            print("Cannot prepare data. Wrong parameters.")

    def create_fold(self):
        train, test, _, _ = train_test_split(self.data_raw, self.data_Y, test_size=TRAIN_TEST_SPLIT,
                                             random_state=RANDOM_SEED,
                                             shuffle=True,
                                             stratify=self.data_Y)

        words = [word for sample in train for word in sample["document"]]
        self.lang = Lang(words)
        train_dataset = CustomDataset(train, self.lang)
        test_dataset = CustomDataset(test, self.lang)

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,  shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)

    def run(self):
        self.prepare_data()
        models = []
        metrics_list = []
        for i_fold in range(N_FOLDS):
            self.create_fold()
            vocab_size = len(self.lang.word2id)

            if self.model_name == "SentimentGRU":
                model = SentimentGRU(vocab_size, self.model_config)
            elif self.model_name == "SentimentCNN":
                model = SentimentCNN(vocab_size, self.model_config)
            else:
                print("Model name does not exist")
                return

            run = wandb.init(
                project="NLU_SA",
                entity="filippomomesso",
                name=f"{self.model_name}_{i_fold:02d}",
                config={
                    "model": self.model_name,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "loss": "BCELoss",
                    "optimizer": "Adam"
                }
            )
            self.optimizer = optim.Adam(model.parameters(), lr=run.config['lr'])
            self.cost_fn = torch.nn.BCEWithLogitsLoss()  # Because we do not have the pad token

            best_model, metrics = self.training_loop(model, self.train_loader, self.test_loader, run)
            models.append(best_model)
            metrics_list.append(metrics)

        # print average and std for metrics
        metrics_df = pd.DataFrame.from_dict(metrics_list)
        metrics_df.loc["mean"] = metrics_df.mean()
        metrics_df.loc["std"] = metrics_df.std()
        metrics_df.loc["max"] = metrics_df.max()
        metrics_df.loc["min"] = metrics_df.min()
        print(metrics_df)
        metrics_df.to_csv(f"{self.model_name}_stats.csv")

        best_model_overall_idx = metrics_df["acc"].idxmax()
        return models[best_model_overall_idx]


    def training_step(self, model, data_loader, optimizer, cost_function, clip=5, epoch=0):
        n_samples = 0
        cumulative_loss = 0.
        cumulative_accuracy = 0.

        model.train()

        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Training Step", leave=False)):
            seqs, text_lens = inputs

            outputs = model(seqs, text_lens)

            loss = cost_function(outputs, targets.unsqueeze(-1).float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

            # add batch size
            n_samples += seqs.shape[0]
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
                seqs, text_lens = inputs
                outputs = model(seqs, text_lens)
                loss = cost_function(outputs, targets.unsqueeze(-1).float())

                # add batch size
                n_samples += seqs.shape[0]
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

    def load_weights(self, model, optimizer, weights_path, device, scheduler=None):
        checkpoint = torch.load(weights_path, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler'])

        return epoch, model, optimizer, scheduler

    def training_loop(self, model, tr_dl, ts_dl, wandb_run, save=False):
        print(wandb_run.name)
        model.apply(init_weights)
        experiment = wandb_run.name

        optimizer = self.optimizer
        cost_fn = self.cost_fn

        best_loss = 0.
        best_acc = 0.

        print("Start training")
        for e in tqdm(range(wandb_run.config['epochs']), desc="Training Loop"):
            train_metrics = self.training_step(model, tr_dl, optimizer, cost_fn, epoch=e)
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
                    self.save_weights(e, model, optimizer, test_loss, f'/content/drive/MyDrive/weights/ResNet18CAN_{experiment}.pth')
                    artifact = wandb.Artifact(f'ResNet18CAN_{experiment}', type='model', metadata={**wandb_run.config, **metrics})
                    artifact.add_file(f'/content/drive/MyDrive/weights/ResNet18CAN_{experiment}.pth')
                    wandb_run.log_artifact(artifact)

            print('\n Epoch: {:d}'.format(e + 1))
            print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_acc))
            print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_acc))
            print('-----------------------------------------------------')

        #visualize(best_model, ts_dl, wandb_run)
        wandb.summary["test_best_loss"] = best_loss
        wandb.summary["test_best_accuracy"] = best_acc
        wandb.finish()
        print('\t BEST Test loss {:.5f}, Test accuracy {:.2f}'.format(best_loss, best_acc))
        best_metrics = {"loss": best_loss, "acc": best_acc, "f1": best_f1}
        return best_model, best_metrics
