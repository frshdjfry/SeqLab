from models.gpt import GPTModel
from models.lstm import LSTMModel
from models.markov import MarkovModel
from utils.evaluators import evaluate_model, load_word2vec_model
from models.transformer import TransformerModel


def objective_markov(trial, train_data, test_data, dataset_name):
    alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)

    model = MarkovModel(alpha=alpha)
    model.train_model(train_data)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity = evaluate_model(model, test_data, word2vec_model)

    return accuracy, perplexity, w2v_similarity


def objective_lstm(trial, train_data, test_data, vocab_size, dataset_name, epochs):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])

    # Initialize the LSTM model with the suggested hyperparameters
    model = LSTMModel(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers, lr=lr)

    model.train_model(encoded_seqs=train_data, epochs=epochs)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity = evaluate_model(model, test_data, word2vec_model)

    return accuracy, perplexity, w2v_similarity


def objective_transformer(trial, train_data, test_data, vocab_size, dataset_name, epochs):
    # Hyperparameters to be tuned
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    nhead = trial.suggest_categorical('nhead', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024])

    # Initialize the Transformer model with the suggested hyperparameters
    model = TransformerModel(vocab_size=vocab_size, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
                             lr=lr)

    model.train_model(train_data, epochs=epochs)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity = evaluate_model(model, test_data, word2vec_model)

    return accuracy, perplexity, w2v_similarity


def objective_gpt(trial, train_data, test_data, vocab_size, dataset_name, epochs):
    # Hyperparameters to be tuned
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    nhead = trial.suggest_categorical('nhead', [8, 12, 16])
    num_layers = trial.suggest_int('num_layers', 2, 12)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [1024, 2048, 3072])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    # Initialize the GPT model with the suggested hyperparameters
    model = GPTModel(vocab_size=vocab_size, lr=lr, num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward)

    model.train_model(train_data, epochs=epochs, batch_size=batch_size)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity = evaluate_model(model, test_data, word2vec_model)

    return accuracy, perplexity, w2v_similarity
