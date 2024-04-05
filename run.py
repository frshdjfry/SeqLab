import json
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from data.data_preprocessing import preprocess_data, split_data
from data.many_to_many_data_preprocessing import preprocess_man_to_many_data, split_multi_feature_data
from models.lstm_attention import LSTMModelWithAttention
from models.multi_lstm_model import MultiLSTMModel
from utils.evaluators import train_and_save_word2vec
from utils.objectives import objective_markov, objective_lstm, objective_transformer, objective_gpt, \
    objective_multi_lstm

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
    elif name == "LSTMModelWithAttention":
        return LSTMModelWithAttention
    elif name == "GPTModel":
        return GPTModel
    elif name == "TransformerModel":
        return TransformerModel
    elif name == "MultiLSTMModel":
        return MultiLSTMModel
    else:
        raise ValueError(f"Unknown model name: {name}")


def objective_one_to_one(trial, model_config, train_data, test_data, vocab_size, dataset_name, vocab_inv):
    model_class = get_model_class(model_config['name'])
    epochs = model_config['epochs']

    if model_class in [LSTMModel, LSTMModelWithAttention]:
        return objective_lstm(trial, train_data, test_data, vocab_size, dataset_name, epochs, vocab_inv)
    elif model_class == GPTModel:
        return objective_gpt(trial, train_data, test_data, vocab_size, dataset_name, epochs, vocab_inv)
    elif model_class == TransformerModel:
        return objective_transformer(trial, train_data, test_data, vocab_size, dataset_name, epochs, vocab_inv)
    elif model_class == MarkovModel:
        return objective_markov(trial, train_data, test_data, dataset_name, vocab_inv)
    else:
        raise ValueError("Invalid model class")


def objective_many_to_one(trial, model_config, train_data, test_data, feature_vocabs, target_feature, dataset_name):
    model_class = get_model_class(model_config['name'])
    epochs = model_config['epochs']

    if model_class == MultiLSTMModel:
        return objective_multi_lstm(
            trial, train_data, test_data, feature_vocabs, target_feature, dataset_name,
            epochs
        )
    else:
        raise ValueError("Invalid model class")


def run_one_to_one_experiment(model_config, dataset_name, train_data, test_data, vocab, vocab_inv):
    model_class = get_model_class(model_config['name'])
    study_name = f"{model_class.__name__} optimization"

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=["accuracy", "perplexity", "w2v_similarity", "final_epoch_loss"],
        mlflow_kwargs={
            "nested": True
        })

    with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
        study = optuna.create_study(directions=['maximize', 'minimize', 'maximize', 'minimize'], study_name=study_name)

        study.optimize(
            lambda trial: objective_one_to_one(trial, model_config, train_data, test_data, len(vocab) + 1,
                                               dataset_name,
                                               vocab_inv),
            n_trials=20, callbacks=[mlflow_callback])

        best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
        mlflow.log_metric("best_accuracy", best_accuracy_trial.values[0])
        mlflow.log_metric("best_perplexity", best_accuracy_trial.values[1])
        mlflow.log_metric("best_w2v", best_accuracy_trial.values[2])
        print(f"Best trial for {model_class.__name__}: Metrics: {best_accuracy_trial.values}")


def run_many_to_one_experiment(model_config, dataset_name, train_data, test_data, vocabs, target_feature):
    model_class = get_model_class(model_config['name'])
    study_name = f"{model_class.__name__} optimization"

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=["accuracy"],
        mlflow_kwargs={
            "nested": True
        })

    with mlflow.start_run(run_name=f"{model_class.__name__} Training", nested=True):
        study = optuna.create_study(directions=['maximize'], study_name=study_name)
        study.optimize(
            lambda trial: objective_many_to_one(
                trial, model_config, train_data, test_data, vocabs, target_feature, dataset_name
            ),
            n_trials=20, callbacks=[mlflow_callback])

        best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
        mlflow.log_metric("best_accuracy", best_accuracy_trial.values[0])
        mlflow.log_metric("best_perplexity", best_accuracy_trial.values[1])
        mlflow.log_metric("best_w2v", best_accuracy_trial.values[2])
        print(f"Best trial for {model_class.__name__}: Metrics: {best_accuracy_trial.values}")


def get_avg_seq_len_multi(encoded_seqs):
    sumof = 0.
    count = 0.
    for k, v in encoded_seqs.items():
        for i in v:
            sumof += len(i)
            count += 1
        break
    print(f'======= ##### {sumof / count}')
    return sumof / count


def get_avg_seq_len_single(encoded_seqs):
    sumof = 0.
    for seq in encoded_seqs:
        sumof += len(seq)
    print(f'======= ##### {sumof / len(encoded_seqs)}')
    return sumof / len(encoded_seqs)


def main():
    for architecture_name, architecture_config in config.items():
        if architecture_name == 'one_to_one':
            for dataset_name in architecture_config['datasets']:
                experiment_name = f"{architecture_name} Experiments on {dataset_name}"
                mlflow.set_experiment(experiment_name)
                if dataset_name.endswith('.txt'):
                    encoded_seqs, vocab, vocab_inv = preprocess_data(dataset_name)
                elif dataset_name.endswith('.csv'):
                    encoded_seqs_dict, vocabs, vocabs_inv = preprocess_man_to_many_data(
                        dataset_name,
                        [architecture_config['target_feature']]
                    )
                    encoded_seqs = encoded_seqs_dict[architecture_config['target_feature']]
                    vocab = vocabs[architecture_config['target_feature']]
                    vocab_inv = vocabs_inv[architecture_config['target_feature']]
                else:
                    raise ValueError('unknown data type')

                train_data, test_data = split_data(encoded_seqs)

                train_and_save_word2vec(encoded_seqs, dataset_name)

                with mlflow.start_run(run_name=f"{architecture_name} Experiments on {dataset_name}"):
                    mlflow.set_tag("dataset", dataset_name)
                    mlflow.set_tag("train_data_len", len(train_data))
                    mlflow.set_tag("target_vocab_size", len(vocab))
                    mlflow.set_tag("avg_seq_len", get_avg_seq_len_single(encoded_seqs))

                    for model_config in architecture_config['models']:
                        run_one_to_one_experiment(model_config, dataset_name, train_data, test_data, vocab, vocab_inv)

        elif architecture_name == 'many_to_one':
            for dataset_name in architecture_config['datasets']:
                experiment_name = f"{architecture_name} Experiments on {dataset_name}"
                mlflow.set_experiment(experiment_name)

                encoded_seqs, vocabs, vocabs_inv = preprocess_man_to_many_data(
                    dataset_name,
                    architecture_config['source_features']
                )
                train_data, test_data = split_multi_feature_data(encoded_seqs)

                train_and_save_word2vec(encoded_seqs[architecture_config['target_feature']], dataset_name)

                with mlflow.start_run(run_name=f"{architecture_name} Experiments on {dataset_name}"):
                    mlflow.set_tag("dataset", dataset_name)
                    mlflow.set_tag("train_data_len", len(encoded_seqs[architecture_config['target_feature']]))
                    mlflow.set_tag("target_vocab_size", len(vocabs[architecture_config['target_feature']]))
                    mlflow.set_tag("avg_seq_len", get_avg_seq_len_multi(encoded_seqs))

                    for model_config in architecture_config['models']:
                        run_many_to_one_experiment(
                            model_config=model_config,
                            dataset_name=dataset_name,
                            train_data=train_data,
                            test_data=test_data,
                            vocabs=vocabs,
                            target_feature=architecture_config['target_feature']
                        )


if __name__ == "__main__":
    main()
