import json
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from data.data_preprocessing import preprocess_data, split_data
from utils.evaluators import train_and_save_word2vec
from utils.objectives import objective_markov, objective_lstm, objective_transformer, objective_gpt

# Import your model classes
from models.markov import MarkovModel
from models.lstm import LSTMModel
from models.gpt import GPTModel
from models.transformer import TransformerModel

# Load configuration from JSON file
with open('config.json') as config_file:
    config = json.load(config_file)


# Function to get model class by name
def get_model_class(name):
    if name == "MarkovModel":
        return MarkovModel
    elif name == "LSTMModel":
        return LSTMModel
    elif name == "GPTModel":
        return GPTModel
    elif name == "TransformerModel":
        return TransformerModel
    else:
        raise ValueError(f"Unknown model name: {name}")


def objective(trial, model_config, train_data, test_data, vocab_size, dataset_name):
    model_class = get_model_class(model_config['name'])
    epochs = model_config['epochs']

    if model_class == LSTMModel:
        return objective_lstm(trial, train_data, test_data, vocab_size, dataset_name, epochs=epochs)
    elif model_class == GPTModel:
        return objective_gpt(trial, train_data, test_data, vocab_size, dataset_name, epochs=epochs)
    elif model_class == TransformerModel:
        return objective_transformer(trial, train_data, test_data, vocab_size, dataset_name, epochs=epochs)
    elif model_class == MarkovModel:
        return objective_markov(trial, train_data, test_data, dataset_name)
    else:
        raise ValueError("Invalid model class")


def run_experiment(model_config, dataset_name, train_data, test_data, vocab):
    model_class = get_model_class(model_config['name'])
    study_name = f"{model_class.__name__} optimization"

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=["accuracy", "perplexity", "w2v_similarity"],
        mlflow_kwargs={
            "nested": True
        })

    with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
        study = optuna.create_study(directions=['maximize','minimize', 'maximize'], study_name=study_name)
        study.optimize(
            lambda trial: objective(trial, model_config, train_data, test_data, len(vocab) + 1, dataset_name),
            n_trials=20, callbacks=[mlflow_callback])

        best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
        mlflow.log_metric("best_accuracy", best_accuracy_trial.values[0])
        mlflow.log_metric("best_perplexity", best_accuracy_trial.values[1])
        mlflow.log_metric("best_w2v", best_accuracy_trial.values[2])
        print(f"Best trial for {model_class.__name__}: Metrics: {best_accuracy_trial.values}")


def main():
    for dataset_name in config['datasets']:
        experiment_name = f"Experiments on {dataset_name}"
        mlflow.set_experiment(experiment_name)

        encoded_seqs, vocab, vocab_inv = preprocess_data(dataset_name)
        train_data, test_data = split_data(encoded_seqs)

        train_and_save_word2vec(encoded_seqs, dataset_name)

        with mlflow.start_run(run_name=f"Experiments on {dataset_name}"):
            mlflow.set_tag("dataset", dataset_name)
            for model_config in config['models']:
                run_experiment(model_config, dataset_name, train_data, test_data, vocab)


if __name__ == "__main__":
    main()
