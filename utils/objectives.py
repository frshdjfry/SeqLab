from models.gpt import GPTModel
from models.lstm import LSTMModel
from models.markov import MarkovModel
from models.multi_lstm_model import MultiLSTMModel
from utils.evaluators import evaluate_one_to_one_model, load_word2vec_model, evaluate_many_to_one_model
from models.transformer import TransformerModel


def objective_markov(trial, train_data, test_data, dataset_name, vocab_inv):
    alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)

    model = MarkovModel(alpha=alpha)
    model.train_model(train_data)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity, final_epoch_loss = evaluate_one_to_one_model(model, test_data, word2vec_model, vocab_inv)

    return accuracy, perplexity, w2v_similarity, final_epoch_loss


def objective_lstm(trial, train_data, test_data, vocab_size, dataset_name, epochs, vocab_inv):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-2)
    num_layers = trial.suggest_int('num_layers', 1, 2)
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])

    # Initialize the LSTM model with the suggested hyperparameters
    model = LSTMModel(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers, lr=lr)

    model.train_model(encoded_seqs=train_data, epochs=epochs)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity, final_epoch_loss = evaluate_one_to_one_model(model, test_data, word2vec_model, vocab_inv)

    return accuracy, perplexity, w2v_similarity, final_epoch_loss


def objective_transformer(trial, train_data, test_data, vocab_size, dataset_name, epochs, vocab_inv):
    # Hyperparameters to be tuned
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-2)
    nhead = trial.suggest_categorical('nhead', [8, 16, 32])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024])

    # Initialize the Transformer model with the suggested hyperparameters
    model = TransformerModel(vocab_size=vocab_size, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
                             lr=lr)

    model.train_model(train_data, epochs=epochs)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity, final_epoch_loss = evaluate_one_to_one_model(model, test_data, word2vec_model, vocab_inv)

    return accuracy, perplexity, w2v_similarity, final_epoch_loss


def objective_gpt(trial, train_data, test_data, vocab_size, dataset_name, epochs, vocab_inv):
    # Hyperparameters to be tuned
    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024])
    batch_size = 64

    # Initialize the GPT model with the suggested hyperparameters
    model = GPTModel(vocab_size=vocab_size, lr=lr, num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward)

    model.train_model(train_data, epochs=epochs, batch_size=batch_size)

    word2vec_model = load_word2vec_model(dataset_name)
    accuracy, w2v_similarity, perplexity, final_epoch_loss = evaluate_one_to_one_model(model, test_data, word2vec_model, vocab_inv)

    return accuracy, perplexity, w2v_similarity, final_epoch_loss


def objective_multi_lstm(trial, train_data, test_data, feature_vocabs, target_feature, dataset_name,
                         epochs):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])

    # Initialize the LSTM model with the suggested hyperparameters
    model = MultiLSTMModel(feature_vocabs=feature_vocabs, target_feature=target_feature,
                           embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, lr=lr)

    # Train the model on the training data
    model.train_model(train_data, epochs=epochs)

    # Evaluate the model on the test data
    word2vec_model = load_word2vec_model(dataset_name)
    vocab_invs = {}
    for feature, vocab in feature_vocabs.items():
        vocab_inv = {i + 1: value for value, i in vocab.items()}
        vocab_invs[feature] = vocab_inv
    # Note: You'll need to adapt or implement the evaluate_model function to work with the new data format and model outputs
    accuracy, w2v_similarity, perplexity, final_epoch_loss = evaluate_many_to_one_model(model, test_data, word2vec_model, vocab_invs, target_feature)

    # Return the metrics of interest
    return accuracy, perplexity, w2v_similarity, final_epoch_loss
