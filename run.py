import json
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import KFold

from data import preprocess_txt_dataset, preprocess_csv_dataset
from data.stats import get_html_dataset_stats
from data.multi_feature_stats import get_html_multi_feature_dataset_stats
from models import MODEL_REGISTRY
from utils.objectives import get_objective_function

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)


def get_model_class(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not recognized.")
    return MODEL_REGISTRY[model_name]


def log_artifacts(dataset_info, train_data, test_data):
    if isinstance(dataset_info['full_data'], dict):
        dataset_stat_path = get_html_multi_feature_dataset_stats(
            train_data, test_data, dataset_info['vocab']
        )
    else:
        dataset_stat_path = get_html_dataset_stats(train_data, test_data, dataset_info['vocab'])
    mlflow.log_artifact(dataset_stat_path, 'dataset_stats')


def log_metrics(best_accuracy_trial):
    mlflow.log_metric("best_accuracy", best_accuracy_trial.values[0])
    mlflow.log_metric("best_perplexity", best_accuracy_trial.values[1])
    mlflow.log_metric("best_w2v", best_accuracy_trial.values[2])


def run_experiment(model_config, dataset_info, target_feature=None, n_trials=20, n_splits=7):
    model_class = get_model_class(model_config['name'])
    study_name = f"{model_class.__name__} optimization"
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Determine the length of data for splitting
    if isinstance(dataset_info['full_data'], dict):
        sample_data = next(iter(dataset_info['full_data'].values()))
    else:
        sample_data = dataset_info['full_data']
    data_length = len(sample_data)

    # Start a parent MLflow run for the entire experiment
    with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
        mlflow.log_param("model_name", model_class.__name__)

        accuracies, perplexities, w2v_similarities = [], [], []

        # Iterate over each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(range(data_length))):  # Operate on data length directly
            if isinstance(dataset_info['full_data'], dict):
                train_data = {feature: [sequences[i] for i in train_idx] for feature, sequences in
                              dataset_info['full_data'].items()}
                test_data = {feature: [sequences[i] for i in test_idx] for feature, sequences in
                             dataset_info['full_data'].items()}
            else:
                train_data = [dataset_info['full_data'][i] for i in train_idx]
                test_data = [dataset_info['full_data'][i] for i in test_idx]

            # Start a nested run for the current fold
            with mlflow.start_run(run_name=f"Fold {fold + 1}", nested=True):
                study = optuna.create_study(directions=['maximize', 'minimize', 'maximize'], study_name=study_name)
                mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(),
                                                 metric_name=["accuracy", "perplexity", "w2v_similarity"],
                                                 mlflow_kwargs={"nested": True})

                study.optimize(
                    lambda trial: get_objective_function(
                        trial=trial,
                        model_class=model_class,
                        train_data=train_data,
                        test_data=test_data,
                        word2vec_model=dataset_info['word2vec_model'],
                        vocab=dataset_info['vocab'],
                        model_config=model_config,
                        target_feature=target_feature
                    ),
                    n_trials=n_trials, callbacks=[mlflow_callback])

                best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
                log_metrics(best_accuracy_trial)
                log_artifacts(dataset_info, train_data, test_data)

                accuracies.append(best_accuracy_trial.values[0])
                perplexities.append(best_accuracy_trial.values[1])
                w2v_similarities.append(best_accuracy_trial.values[2])

                print(f"Best trial for {model_class.__name__}: Metrics: {best_accuracy_trial.values}")
        # Calculate and log the average metrics across all folds
        avg_accuracy = sum(accuracies) / n_splits
        avg_perplexity = sum(perplexities) / n_splits
        avg_w2v_similarity = sum(w2v_similarities) / n_splits

        mlflow.log_metrics({
            "average_accuracy": avg_accuracy,
            "average_perplexity": avg_perplexity,
            "average_w2v_similarity": avg_w2v_similarity
        })

        print(
            f"Average Metrics - Accuracy: {avg_accuracy}, Perplexity: {avg_perplexity}, W2V Similarity: {avg_w2v_similarity}")


def process_and_run_experiments(experiment_config):
    n_trials = experiment_config.get('number_of_trials', 20)
    n_splits = experiment_config.get('kfold_splits', 7)

    for dataset_name in experiment_config['datasets']:
        feature_dimensions = experiment_config['feature_dimensions']
        mlflow.set_experiment(f"{feature_dimensions} Experiments on {dataset_name}")
        with mlflow.start_run(run_name=f"{feature_dimensions} Experiments on {dataset_name}"):
            mlflow.log_param("n_folds", n_splits)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("augment_by_subsequences", experiment_config.get('augment_by_subsequences', False))
            mlflow.log_param("augment_by_key", experiment_config.get('augment_by_key', False))
            mlflow.log_param("normalize_chords", experiment_config.get('normalize_chords', False))
            # Data preprocessing based on dataset format
            if dataset_name.endswith('.txt'):
                full_data, word2vec_model, vocab, avg_seq_len = preprocess_txt_dataset(
                    dataset_name, experiment_config
                )
                mlflow.log_param("train_data_len", len(full_data))
                mlflow.log_param("vocab_len", len(vocab.keys()))
                mlflow.log_param("avg_seq_len", avg_seq_len)

            elif dataset_name.endswith('.csv'):
                full_data, word2vec_model, vocab, avg_seq_len = preprocess_csv_dataset(dataset_name,
                                                                                       experiment_config,
                                                                                       feature_dimensions)
            else:
                raise ValueError('Unknown data type')

            dataset_info = {
                'full_data': full_data,
                'word2vec_model': word2vec_model,
                'vocab': vocab,
                'avg_seq_len': avg_seq_len
            }

            for model_config in experiment_config['models']:
                target_feature = experiment_config.get('target_feature')
                run_experiment(model_config, dataset_info, target_feature, n_trials, n_splits)


def main():
    for experiment_config in config['experiments']:
        process_and_run_experiments(experiment_config)


if __name__ == "__main__":
    main()
