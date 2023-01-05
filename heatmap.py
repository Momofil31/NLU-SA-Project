'''
    Code to create json to be used with neat-vision for generating SVG heatmaps.
    https://cbaziotis.github.io/neat-vision/

    Outputs in ./heatmaps/ json objects in format like:
    [most_confident_neg, most_confident_pos, most_confident_wrong_neg, most_confident_wrong_pos]
'''

import argparse
import json
import wandb
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, f1_score
from experiment import Experiment
from baseline import BaselineExperiment
from models import *
from settings import *

nameToModel = {
    "BiGRUAttention": BiGRU
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["BiGRUAttention"], help="Specify model type. Eg. 'BiGRU'")
    parser.add_argument("task", choices=["polarity", "polarity-filter", "subjectivity"], help="Specify which task to perform.")
    parser.add_argument("--fold_index", type=int, choices=[0, 1, 2, 3, 4],  help="Specifify the fold index to load correct train/test split.")
    parser.add_argument("-pe", "--pretrained_embeddings", action="store_true", help="Specify if use pretrained embeddings.")
    args = parser.parse_args()

    sjv_classifier = None
    sjv_vectorizer = None

    if args.task == "polarity-filter":
        # Run subjectivity
        exp_subjectivity = BaselineExperiment(task="subjectivity")
        sjv_classifier, sjv_vectorizer = exp_subjectivity.run()

    # load model
    api = wandb.Api()
    pe_string = "_pe" if args.pretrained_embeddings else ""
    name = f"{args.task}_{args.model}{pe_string}_fold_{args.fold_index:02d}"

    artifact_name = f'{WANDB_ENTITY}/{WANDB_PROJECT}/{name}:latest'
    print(artifact_name)

    checkpoint = f"{name}.pth"
    print(checkpoint)

    artifact = api.artifact(artifact_name)
    artifact.download(root=WEIGHTS_SAVE_PATH)
    print(artifact.metadata)
    model_config = artifact.metadata

    if model_config.get("vocab_size"):
        model = nameToModel[args.model](model_config["vocab_size"], model_config)
    else:
        raise Exception("Config does not specify vocab_size.")

    checkpoint = torch.load(f"{WEIGHTS_SAVE_PATH}/{checkpoint}", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # load data
    exp = Experiment(args.task, sjv_classifier, sjv_vectorizer)
    exp.model_config = model_config
    exp.prepare_data()
    exp.create_folds()
    exp.create_dataloaders(args.fold_index)
    data_loader = exp.test_loader

    # run model
    y_pred = []
    outputs_list = []
    inputs_list = []
    y_gt = []
    attentions_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Test Step", leave=False)):
            for k in inputs.keys():
                inputs[k] = inputs[k].to(DEVICE)
                targets = targets.to(DEVICE)

            # since this code is meant to run only for attention based models it expects a tuple as output
            outputs, attentions = model(inputs)
            predicted = torch.sigmoid(outputs).round().int()

            outputs_list += torch.sigmoid(outputs).squeeze().tolist()
            y_pred += predicted.squeeze().tolist()
            y_gt += targets.int().tolist()

            attentions_list += attentions.sqrt().permute(1, 0, 2).squeeze().tolist()  # permute attention tensor to batch * seq_len * attention_scalar
            inputs_list += inputs["document"].int().squeeze().tolist()

    # Compute and print metrics
    f1 = f1_score(y_gt, y_pred)
    accuracy = accuracy_score(y_gt, y_pred)
    print('Test accuracy {:.2f}, Test F1 {:.2f}'.format(accuracy, f1))
    #print(len(y_pred), len(outputs_list), len(y_gt), len(attentions_list))

    # Extract most confident samples, both right and wrong
    most_confident_neg = np.argmin(outputs_list)
    most_confident_pos = np.argmax(outputs_list)

    wrong_predictions_neg = [idx for idx, (pred, label) in enumerate(zip(y_pred, y_gt)) if pred != label and label == 0]
    wrong_predictions_pos = [idx for idx, (pred, label) in enumerate(zip(y_pred, y_gt)) if pred != label and label == 1]
    print(wrong_predictions_neg, len(wrong_predictions_neg))
    print(wrong_predictions_pos, len(wrong_predictions_pos))
    wrong_predictions = wrong_predictions_neg + wrong_predictions_pos

    most_confident_wrong_neg = wrong_predictions_neg[0]
    most_confident_wrong_pos = wrong_predictions_pos[0]

    for idx in wrong_predictions_neg:
        if outputs_list[idx] > outputs_list[most_confident_wrong_neg]:
            most_confident_wrong_neg = idx
    for idx in wrong_predictions_pos:
        if outputs_list[idx] < outputs_list[most_confident_wrong_pos]:
            most_confident_wrong_pos = idx

    # Make json
    heatmap_data = []
    for idx in [most_confident_neg, most_confident_pos, most_confident_wrong_neg, most_confident_wrong_pos] + wrong_predictions:
        document = inputs_list[idx]
        document_text = []
        for word_idx in document:
            if word_idx == 0:
                break
            document_text.append(exp.lang.id2word[word_idx])

        heatmap_entry = dict()
        heatmap_entry["text"] = document_text
        heatmap_entry["label"] = y_gt[idx]
        heatmap_entry["prediction"] = y_pred[idx]
        heatmap_entry["posterior"] = outputs_list[idx]
        heatmap_entry["attention"] = attentions_list[idx]
        heatmap_entry["id"] = int(idx)
        heatmap_data.append(heatmap_entry)

    with open(f"heatmaps/{name}.json", "w") as write_file:
        json.dump(heatmap_data, write_file)
