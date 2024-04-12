import json
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

from data import preprocess_txt_dataset, preprocess_csv_dataset
from models import MODEL_REGISTRY
from utils.objectives import get_objective_function

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)


def get_model_class(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not recognized.")
    return MODEL_REGISTRY[model_name]


def run_experiment(model_config, dataset_info, target_feature=None):
    model_class = get_model_class(model_config['name'])
    study_name = f"{model_class.__name__} optimization"

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=["accuracy", "perplexity", "w2v_similarity"],
        mlflow_kwargs={"nested": True})

    with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
        study = optuna.create_study(directions=['maximize', 'minimize', 'maximize'], study_name=study_name)
        study.optimize(
            lambda trial: get_objective_function(
                trial=trial,
                model_class=model_class,
                train_data=dataset_info['train_data'],
                test_data=dataset_info['test_data'],
                word2vec_model=dataset_info['word2vec_model'],
                vocab=dataset_info['vocab'],
                model_config=model_config,
                target_feature=target_feature
            ),
            n_trials=20, callbacks=[mlflow_callback])

        best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
        mlflow.log_metric("best_accuracy", best_accuracy_trial.values[0])
        mlflow.log_metric("best_perplexity", best_accuracy_trial.values[1])
        mlflow.log_metric("best_w2v", best_accuracy_trial.values[2])
        print(f"Best trial for {model_class.__name__}: Metrics: {best_accuracy_trial.values}")


def process_and_run_experiments(architecture_name, architecture_config):
    for dataset_name in architecture_config['datasets']:
        mlflow.set_experiment(f"{architecture_name} Experiments on {dataset_name}")
        with mlflow.start_run(run_name=f"{architecture_name} Experiments on {dataset_name}"):
            # Data preprocessing based on dataset format
            if dataset_name.endswith('.txt'):
                train_data, test_data, word2vec_model, vocab, avg_seq_len = preprocess_txt_dataset(dataset_name)
            elif dataset_name.endswith('.csv'):
                train_data, test_data, word2vec_model, vocab, avg_seq_len = preprocess_csv_dataset(dataset_name,
                                                                                                   architecture_config,
                                                                                                   architecture_name)
            else:
                raise ValueError('Unknown data type')

            dataset_info = {
                'train_data': train_data,
                'test_data': test_data,
                'word2vec_model': word2vec_model,
                'vocab': vocab,
                'avg_seq_len': avg_seq_len
            }

            for model_config in architecture_config['models']:
                target_feature = architecture_config.get('target_feature')
                run_experiment(model_config, dataset_info, target_feature)


def main():
    for architecture_name, architecture_config in config.items():
        process_and_run_experiments(architecture_name, architecture_config)


if __name__ == "__main__":
    main()
