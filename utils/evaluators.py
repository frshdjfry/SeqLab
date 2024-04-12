import math
import torch

from models import MODEL_REGISTRY
from models.markov import MarkovModel
from scipy.spatial.distance import cosine


def evaluate_model(model_path, test_data, word2vec_model, vocab, target_feature=None):
    model = load_model(model_path)
    predictions, actuals, probabilities, sequences = collect_predictions(
        model,
        test_data,
        target_feature
    )
    avg_similarity = calculate_total_similarity(predictions, actuals, word2vec_model) / len(predictions)
    accuracy = calculate_accuracy(predictions, actuals)
    total_log_likelihood = calculate_total_log_likelihood(probabilities)
    avg_log_likelihood = -total_log_likelihood / len(predictions)
    perplexity = calculate_perplexity(avg_log_likelihood)
    print_predictions(predictions, actuals, sequences, vocab, target_feature)

    return accuracy, avg_similarity, perplexity


def collect_predictions(model, test_data, target_feature):
    predictions, actuals, probabilities_list, sequences = [], [], [], []
    is_many_to_one = isinstance(test_data, dict)
    test_len = len(test_data) if not is_many_to_one else len(next(iter(test_data.values())))
    for i in range(test_len):
        if is_many_to_one:
            current_sequence = [v[i][:-1] for v in test_data.values()]
            actual_next = test_data[target_feature][i][-1]
        else:
            current_sequence, actual_next = test_data[i][:-1], test_data[i][-1]
        if len(current_sequence) <= 1:
            continue
        if isinstance(model, MarkovModel):
            current_chord = current_sequence[-1]
            predicted_next = model.predict(current_sequence)
            probability = model.get_transition_probability(current_chord, actual_next)
        else:
            probabilities = model.predict_with_probabilities(current_sequence)
            predicted_next = probabilities.index(max(probabilities))
            probability = probabilities[actual_next]
        predictions.append(predicted_next)
        actuals.append(actual_next)
        probabilities_list.append(probability)
        sequences.append(current_sequence)

    return predictions, actuals, probabilities_list, sequences


def word2vec_similarity(model, word1, word2):
    if word1 in model.wv.index_to_key and word2 in model.wv.index_to_key:
        similarity = 1 - cosine(model.wv[word1], model.wv[word2])
        return similarity if similarity > 0.0 else 0.0
    else:
        return 0.0


def calculate_total_similarity(predictions, actuals, word2vec_model):
    return sum(word2vec_similarity(word2vec_model, a, p) for p, a in zip(predictions, actuals))


def calculate_accuracy(predictions, actuals):
    return sum(p == a for p, a in zip(predictions, actuals)) / len(predictions) if predictions else 0


def calculate_total_log_likelihood(probabilities):
    return sum(math.log(p) for p in probabilities if p > 0)


def calculate_perplexity(avg_log_likelihood):
    return math.exp(-avg_log_likelihood) if avg_log_likelihood else float('inf')


def print_predictions(predicted_values, actual_values, sequences, vocab, target_feature):
    vocab_inv = vocab[target_feature] if isinstance(vocab[list(vocab.keys())[0]], dict) else vocab
    print("First 20 Predictions vs Actual Values:")
    for i, (pred, actual, seq) in enumerate(zip(predicted_values, actual_values, sequences[:20])):
        if isinstance(seq[0], list):
            seq = seq[0]
        readable_sequence = [vocab_inv.get(item, 'UNK') for item in seq]
        print(
            f"{i + 1:02d}. Sequence: {readable_sequence}, \nPredicted: {vocab_inv.get(pred, 'UNK')}, "
            f"\nActual: {vocab_inv[actual]}")


def load_model(model_path):
    checkpoint = torch.load(model_path)
    model_class = MODEL_REGISTRY[checkpoint['config']['class_name']]
    model = model_class(vocab=checkpoint['vocab'], **checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
