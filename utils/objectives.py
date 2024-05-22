import mlflow

from utils.evaluators import evaluate_model


def suggest_hyperparameters(trial, optimization_config):
    params = {}
    for param, details in optimization_config.items():
        if 'values' in details:
            params[param] = trial.suggest_categorical(param, details['values'])
        elif 'min' in details and 'max' in details:
            if isinstance(details['min'], int):
                params[param] = trial.suggest_int(param, details['min'], details['max'])
            elif isinstance(details['min'], float):
                params[param] = trial.suggest_float(param, details['min'], details['max'])
        else:
            raise ValueError('Wrong hyperparameters in optimization_config.')
    return params


def get_objective_function(trial, model_class, train_data, test_data, word2vec_model, vocab, model_config,
                           target_feature=None):
    params = suggest_hyperparameters(trial, model_config['optimization_config'])
    model = model_class(
        vocab=vocab,
        target_feature=target_feature,
        max_length=get_max_length(train_data, target_feature),
        **params
    )
    saved_model_path = model.train_model(
        train_data,
        test_data,
        epochs=model_config['epochs'],
        **params
    )
    trial.set_user_attr("model_path", saved_model_path)
    trial.set_user_attr("final_epoch_loss", model.final_epoch_loss)

    accuracy, w2v_similarity, perplexity = evaluate_model(
        model_path=saved_model_path,
        test_data=test_data,
        word2vec_model=word2vec_model,
        vocab=get_vocab_inv(vocab),
        target_feature=target_feature
    )
    return accuracy, perplexity, w2v_similarity


def get_vocab_inv(vocab):
    if isinstance(vocab[list(vocab.keys())[0]], dict):
        return get_inverse_vocab_dict(vocab)
    else:
        return get_inverse_vocab(vocab)


def get_inverse_vocab_dict(feature_vocabs):
    vocab_inverse_dict = {}
    for feature, vocab in feature_vocabs.items():
        vocab_inv = {word_id: word for word, word_id in vocab.items()}
        vocab_inverse_dict[feature] = vocab_inv
    return vocab_inverse_dict


def get_inverse_vocab(vocab):
    vocab_inv = {}
    for word, word_id in vocab.items():
        vocab_inv[word_id] = word
    return vocab_inv


def get_max_length(train_data, target_feature):
    if isinstance(train_data, dict):
        max_lengths = {feature: max(len(seq) for seq in sequences) for feature, sequences in
                       train_data.items()}
        return max_lengths[target_feature] - 1
    else:
        return max(len(seq) for seq in train_data) - 1
