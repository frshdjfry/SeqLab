from utils.evaluators import evaluate_model


def suggest_hyperparameters(trial, optimization_config):
    params = {}
    for param, details in optimization_config.items():
        if details['type'] == 'int':
            params[param] = trial.suggest_int(param, details['min'], details['max'])
        elif details['type'] == 'categorical':
            params[param] = trial.suggest_categorical(param, details['values'])
        elif details['type'] == 'float':
            params[param] = trial.suggest_float(param, details['min'], details['max'])
    return params


def get_objective_function(trial, model_class, train_data, test_data, word2vec_model, vocab, model_config,
                           target_feature=None):
    params = suggest_hyperparameters(trial, model_config['optimization_config'])
    model = model_class(vocab=vocab, target_feature=target_feature, **params)
    model.train_model(train_data, epochs=model_config['epochs'], **params)
    print(vocab)
    accuracy, w2v_similarity, perplexity, final_epoch_loss = evaluate_model(
        model=model,
        test_data=test_data,
        word2vec_model=word2vec_model,
        vocab=get_vocab_inv(vocab),
        target_feature=target_feature
    )
    return accuracy, perplexity, w2v_similarity, final_epoch_loss


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
