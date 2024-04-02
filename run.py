import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from data.data_preprocessing import preprocess_data
from models.markov import MarkovModel
from models.lstm import LSTMModel
from models.gpt import GPTModel
from data.data_preprocessing import split_data
from utils.evaluators import train_and_save_word2vec
from utils.objectives import objective_markov, objective_lstm, objective_transformer, objective_gpt
from models.transformer import TransformerModel


def objective(trial, model_class, train_data, test_data, vocab_size, dataset_name):
    if model_class == LSTMModel:
        return objective_lstm(trial, train_data, test_data, vocab_size, dataset_name)
    elif model_class == GPTModel:
        return objective_gpt(trial, train_data, test_data, vocab_size, dataset_name)
        pass
    elif model_class == TransformerModel:
        return objective_transformer(trial, train_data, test_data, vocab_size, dataset_name)
        pass
    elif model_class == MarkovModel:
        return objective_markov(trial, train_data, test_data, dataset_name)
        pass
    else:
        raise


def run_experiment(model_class, dataset_name, train_data, test_data, vocab):
    study_name = f"{model_class.__name__} optimization"

    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="perplexity", mlflow_kwargs={
        "nested": True
    })

    with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
        study = optuna.create_study(direction='minimize', study_name=study_name)
        study.optimize(lambda trial: objective(trial, model_class, train_data, test_data, len(vocab)+1, dataset_name), n_trials=20,
                       callbacks=[mlflow_callback])

        best_trial = study.best_trial
        mlflow.log_metric("best_perplexity", best_trial.value)
        print(f"Best trial for {model_class.__name__}: Perplexity: {best_trial.value}")


def main():
    datasets = ["data/mcgill.txt", "data/cocopops.txt"]
    models = [
        MarkovModel,
        LSTMModel,
        GPTModel,
        TransformerModel
    ]

    for dataset_name in datasets:
        experiment_name = f"Experiments on {dataset_name}"
        mlflow.set_experiment(experiment_name)

        encoded_seqs, vocab, vocab_inv = preprocess_data(dataset_name)
        train_data, test_data = split_data(encoded_seqs)

        train_and_save_word2vec(encoded_seqs, dataset_name)

        with mlflow.start_run(run_name=f"Experiments on {dataset_name}"):
            mlflow.set_tag("dataset", dataset_name)
            for model in models:
                run_experiment(model, dataset_name, train_data, test_data, vocab)


if __name__ == "__main__":
    main()
