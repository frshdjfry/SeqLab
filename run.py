import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import KFold

from data import preprocess_txt_dataset, preprocess_csv_dataset, preprocess_separate_train_test_txt_dataset, \
    preprocess_separate_train_test_csv_dataset
from data.stats import get_html_dataset_stats
from data.multi_feature_stats import get_html_multi_feature_dataset_stats
from models import MODEL_REGISTRY
from utils.objectives import get_objective_function
import json
from typing import List, Dict, Any, Optional, Union


class OptimizationConfig:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


class ModelConfig:
    def __init__(self, name: str, epochs: int, optimization_config: Dict[str, Any]):
        self.name = name
        self.epochs = epochs
        self.optimization_config = OptimizationConfig(optimization_config)

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**model_dict)


class ExperimentConfig:
    def __init__(self, feature_dimensions: str, number_of_trials: int, kfold_splits: int, datasets: List[str],
                 models: List[ModelConfig], target_feature: Optional[str] = None,
                 source_features: Optional[List[str]] = None, augment_by_subsequences: bool = False,
                 normalize_chords: bool = False, augment_by_key: bool = False, test_dataset: str = None):
        self.feature_dimensions = feature_dimensions
        self.number_of_trials = number_of_trials
        self.kfold_splits = kfold_splits
        self.datasets = datasets
        self.models = models
        self.target_feature = target_feature
        self.source_features = source_features
        self.augment_by_subsequences = augment_by_subsequences
        self.normalize_chords = normalize_chords
        self.augment_by_key = augment_by_key
        self.test_dataset = test_dataset

    @classmethod
    def from_dict(cls, experiment_dict: Dict[str, Any]) -> 'ExperimentConfig':
        models = [ModelConfig.from_dict(model) for model in experiment_dict['models']]
        experiment_dict.pop('models')
        return cls(models=models, **experiment_dict)


class Config:
    def __init__(self, config_path: str):
        self.experiments = self.load_config(config_path)

    def load_config(self, config_path: str) -> List[ExperimentConfig]:
        with open(config_path) as config_file:
            config_dict = json.load(config_file)
        return [ExperimentConfig.from_dict(experiment) for experiment in config_dict['experiments']]

    def get_experiments(self) -> List[ExperimentConfig]:
        return self.experiments


class DatasetInfo:
    def __init__(self, full_data: Any, word2vec_model: Any, vocab: Dict[str, Any], test_start_index: int = None):
        self.full_data = full_data
        self.word2vec_model = word2vec_model
        self.vocab = vocab
        self.test_start_index = test_start_index


class ExperimentRunner:
    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config
        self.n_trials = experiment_config.number_of_trials
        self.n_splits = experiment_config.kfold_splits

    def get_model_class(self, model_name: str):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} not recognized.")
        return MODEL_REGISTRY[model_name]

    def log_artifacts(self, dataset_info: DatasetInfo, train_data: Any, test_data: Any):
        if isinstance(dataset_info.full_data, dict):
            dataset_stat_path = get_html_multi_feature_dataset_stats(
                train_data, test_data, dataset_info.vocab
            )
        else:
            dataset_stat_path = get_html_dataset_stats(train_data, test_data, dataset_info.vocab)
        mlflow.log_artifact(dataset_stat_path, 'dataset_stats')

    def log_metrics(self, best_accuracy_trial: optuna.Trial):
        mlflow.log_metric("best_accuracy", best_accuracy_trial.values[0])
        mlflow.log_metric("best_perplexity", best_accuracy_trial.values[1])
        mlflow.log_metric("best_w2v", best_accuracy_trial.values[2])

    def run_kfold_experiment(self, model_config: ModelConfig, dataset_info: DatasetInfo,
                             target_feature: Optional[str] = None):
        model_class = self.get_model_class(model_config.name)
        study_name = f"{model_class.__name__} optimization"
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        sample_data = next(iter(dataset_info.full_data.values())) if isinstance(dataset_info.full_data,
                                                                                dict) else dataset_info.full_data
        data_length = len(sample_data)
        word2vec_model = dataset_info.word2vec_model
        vocab = dataset_info.vocab

        with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
            mlflow.log_params({
                "n_folds": self.n_splits,
                "n_trials": self.n_trials,
                "model_name": model_class.__name__
            })

            best_accuracy_trials = []
            for fold, (train_idx, test_idx) in enumerate(kf.split(range(data_length))):
                train_data, test_data = self.split_data(dataset_info.full_data, train_idx, test_idx)

                with mlflow.start_run(run_name=f"Fold {fold + 1}", nested=True):
                    best_accuracy_trial = self.run_optuna_trials(
                        model_class, train_data, test_data, word2vec_model, vocab, model_config, target_feature,
                        study_name
                    )
                    best_accuracy_trials.append(best_accuracy_trial)
                    self.log_metrics(best_accuracy_trial)
                    self.log_artifacts(dataset_info, train_data, test_data)

                    print(
                        f"Best trial for {model_class.__name__} "
                        f"Metrics: {best_accuracy_trial.values}"
                    )

            self.log_average_metrics(best_accuracy_trials)

    def run_single_experiment(
            self, model_config: ModelConfig,
            dataset_info: DatasetInfo,
            target_feature: Optional[str] = None
    ):
        model_class = self.get_model_class(model_config.name)
        study_name = f"HyperParameter Optimization"

        vocab = dataset_info.vocab
        word2vec_model = dataset_info.word2vec_model
        if isinstance(dataset_info.full_data, dict):
            train_data, test_data = {}, {}
            for feature in self.experiment_config.source_features:
                train_data[feature] = dataset_info.full_data[feature][:dataset_info.test_start_index]
                test_data[feature] = dataset_info.full_data[feature][dataset_info.test_start_index:]
        else:
            train_data = dataset_info.full_data[:dataset_info.test_start_index]
            test_data = dataset_info.full_data[dataset_info.test_start_index:]
        with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
            mlflow.log_params({
                "n_folds": self.n_splits,
                "n_trials": self.n_trials,
                "model_name": model_class.__name__
            })

            best_accuracy_trial = self.run_optuna_trials(
                model_class, train_data, test_data, word2vec_model, vocab, model_config, target_feature, study_name
            )

            self.log_metrics(best_accuracy_trial)
            self.log_artifacts(dataset_info, train_data, test_data)
            print(
                f"Best trial for {model_class.__name__} "
                f"Metrics: {best_accuracy_trial.values}"
            )

    def run_optuna_trials(
            self, model_class, train_data, test_data, word2vec_model, vocab, model_config, target_feature, study_name
    ):
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
                word2vec_model=word2vec_model,
                vocab=vocab,
                model_config=model_config,
                target_feature=target_feature
            ),
            n_trials=self.n_trials, callbacks=[mlflow_callback])
        best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
        return best_accuracy_trial

    @staticmethod
    def split_data(full_data: Union[Dict[str, List[List[int]]], List[List[int]]], train_idx: List[int],
                   test_idx: List[int]):
        if isinstance(full_data, dict):
            train_data = {feature: [sequences[i] for i in train_idx] for feature, sequences in full_data.items()}
            test_data = {feature: [sequences[i] for i in test_idx] for feature, sequences in full_data.items()}
        else:
            train_data = [full_data[i] for i in train_idx]
            test_data = [full_data[i] for i in test_idx]
        return train_data, test_data

    def log_average_metrics(self, best_accuracy_trials: List[Any]):
        accuracies, perplexities, w2v_similarities = [], [], []
        for best_accuracy_trial in best_accuracy_trials:
            accuracies.append(best_accuracy_trial.values[0])
            perplexities.append(best_accuracy_trial.values[1])
            w2v_similarities.append(best_accuracy_trial.values[2])

        mlflow.log_metrics({
            "average_accuracy": sum(accuracies) / self.n_splits,
            "average_perplexity": sum(perplexities) / self.n_splits,
            "average_w2v_similarity": sum(w2v_similarities) / self.n_splits
        })

        print(
            f"Average Metrics - Accuracy: {sum(accuracies) / self.n_splits}"
            f", Perplexity: {sum(perplexities) / self.n_splits}"
            f", W2V Similarity: {sum(w2v_similarities) / self.n_splits}"
        )

    def process_and_run_experiments(self):
        if self.experiment_config.test_dataset:
            for dataset_name in self.experiment_config.datasets:
                mlflow.set_experiment(f"{self.experiment_config.feature_dimensions} Experiments on {dataset_name} "
                                      f"and {self.experiment_config.test_dataset}")
                with mlflow.start_run(
                        run_name=f"{self.experiment_config.feature_dimensions} Experiments on {dataset_name} "
                                 f"and {self.experiment_config.test_dataset}"):

                    dataset_info = self.process_separate_train_test_dataset(
                        train_dataset_name=dataset_name,
                        test_dataset_name=self.experiment_config.test_dataset
                    )

                    for model_config in self.experiment_config.models:
                        self.run_single_experiment(
                            model_config,
                            dataset_info,
                            self.experiment_config.target_feature
                        )
        else:
            for dataset_name in self.experiment_config.datasets:
                mlflow.set_experiment(f"{self.experiment_config.feature_dimensions} Experiments on {dataset_name}")
                with mlflow.start_run(
                        run_name=f"{self.experiment_config.feature_dimensions} Experiments on {dataset_name}"):
                    dataset_info = self.preprocess_dataset(dataset_name)
                    for model_config in self.experiment_config.models:
                        self.run_kfold_experiment(model_config, dataset_info, self.experiment_config.target_feature)

    def preprocess_dataset(self, dataset_name: str) -> DatasetInfo:
        if dataset_name.endswith('.txt'):
            full_data, word2vec_model, vocab, avg_seq_len = preprocess_txt_dataset(dataset_name, self.experiment_config)
        elif dataset_name.endswith('.csv'):
            full_data, word2vec_model, vocab, avg_seq_len = preprocess_csv_dataset(
                dataset_name, self.experiment_config,
                self.experiment_config.feature_dimensions
            )
        else:
            raise ValueError('Unknown data type')

        return DatasetInfo(full_data, word2vec_model, vocab)

    def process_separate_train_test_dataset(self, train_dataset_name: str, test_dataset_name: str) -> DatasetInfo:
        if train_dataset_name.endswith('.txt'):
            full_data, word2vec_model, vocab, test_start_index = preprocess_separate_train_test_txt_dataset(
                train_dataset_name, test_dataset_name, self.experiment_config
            )
        elif train_dataset_name.endswith('.csv'):
            full_data, word2vec_model, vocab, test_start_index = preprocess_separate_train_test_csv_dataset(
                train_dataset_name, test_dataset_name, self.experiment_config
            )
        else:
            raise ValueError('Unknown data type')

        return DatasetInfo(full_data, word2vec_model, vocab, test_start_index)


def main():
    experiment_config = Config('config.json')
    for config in experiment_config.get_experiments():
        runner = ExperimentRunner(config)
        runner.process_and_run_experiments()


if __name__ == "__main__":
    main()
