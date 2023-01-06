# Deep Sequence Models for Subjectivity and Polarity Classification

This repository contains the code for the project of Natural Language Understanding A.Y. 2021-2022 of University of Trento. 

In [`report.pdf`](report.pdf) you can find the report for this work.
In [this WandB project](https://wandb.ai/filippomomesso/NLU_SA?workspace=user-filippomomesso) you can find all the training logs, metrics plots and weigths of the trained models. [https://wandb.ai/filippomomesso/NLU_SA?workspace=user-filippomomesso](https://wandb.ai/filippomomesso/NLU_SA?workspace=user-filippomomesso)

## What does this repo do?
In this work, I use various deep learning models to tackle subjectivity detection and polarity classification tasks and compare their performances showing that transfer learning on pre-trained large transformer models gives astonishing results in terms of accuracy, at the cost of higher computational needs, while smaller recurrent neural networks with an attention mechanism offer a reasonable trade-off between the two.

List of trained models:
* Bidirectional GRU
* Bidirectional GRU with soft-attention
* TextCNN
* DistilBERT
* RoBERTa
* Big Bird

## How to run?

1. Create a virtual environment on python 3.8 or greater or a conda environment.
   
    ```
    python -m venv venv
    source venv/bin/activate
    ```
    or 

    ```
    conda create --name nlu-sa python=3.8
    conda activate nlu-sa
    ```

2. Install required dependencies.
    ```
    pip install -r requirements.txt
    ```
3. To run the baseline experiment:
    ```
    python baseline.py
    ```
4. To Run experiments (use `--help` flag to have a list of the available arguments).
   ```
   python main.py <ExperimentName> <Task> <...args>
   ```
   Eg. `python main.py BiGRUAttention polarity -pe`

5. To produce heatmaps `.json` files to be feed into [NeatVision](https://github.com/cbaziotis/neat-vision) for attention heatmaps visualization (`--help` for arguments list).
   ```
   python heatmap.py <Model> <Task> --fold_index <index> <..args>
   ```

## Generated objective and subjective sentences
To corroborate the hypothesis that the baseline model lears to classify the subjectivity of a sentence based on the presence of words that are only present in one of the two classes lexicons I generated a set of sentences that should trick the model using ChatGPT.

* Objective sentences that contain words that are only present in the subjective lexicon:
>"The widely reserved, self-determination and simplicity of the 12-step program have proven to be an effective life-affirming method for those seeking to overcome addiction and achieve reconciliation with themselves and others."

>"The artist-agent's creative approach to marketing and promotion has helped to boost the success and stylishness of numerous music and entertainment projects."

* Subjective sentences that contain words that are only present in the objective lexicon:
> "I was shocked to discover that the financial webcams we had been using were actually part of a scheme known as 'frodes', and I couldn't believe that Daddy's client would scoff at the idea of being caught up in such a bale of trouble."

>"I felt betrayed and stunned, but I knew I had to move on and find a new situation-based opportunity, even if it meant leaving behind the familiar Composers' Castle and the territorial Marjorie and Margaret."

The baseline model is not able to correctly classify these sentences, as expected. To run this experiment, use the following command:
```
python adversarial_examples.py <ModelName> <Task> <...args>
```
                      