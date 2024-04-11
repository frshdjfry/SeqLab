import os

import math
from gensim.models import Word2Vec

from models.markov import MarkovModel
from scipy.spatial.distance import cosine


def evaluate_model(model, test_data, word2vec_model, vocab, target_feature=None):
    predictions, actuals, probabilities, sequences, final_epoch_loss = collect_predictions(model,
                                                                                           test_data,
                                                                                           target_feature)
    avg_similarity = calculate_total_similarity(predictions, actuals, word2vec_model) / len(predictions)
    accuracy = calculate_accuracy(predictions, actuals)
    total_log_likelihood = calculate_total_log_likelihood(probabilities)
    avg_log_likelihood = -total_log_likelihood / len(predictions)
    perplexity = calculate_perplexity(avg_log_likelihood)
    print_predictions(predictions, actuals, sequences, vocab, target_feature)

    return accuracy, avg_similarity, perplexity, final_epoch_loss


def collect_predictions(model, test_data, target_feature):
    predictions, actuals, probabilities_list, sequences, final_epoch_loss = [], [], [], [], 0
    is_many_to_one = isinstance(test_data, dict)
    test_len = len(test_data) if not is_many_to_one else len(next(iter(test_data.values())))
    for i in range(test_len):
        if is_many_to_one:
            current_sequence = [v[i][:-1] for v in test_data.values()]
            actual_next = test_data[target_feature][i][-1]
        else:
            current_sequence, actual_next = test_data[i][:-1], test_data[i][-1]

        if isinstance(model, MarkovModel):
            current_chord = current_sequence[-1]
            predicted_next = model.predict(current_sequence)
            probability = model.get_transition_probability(current_chord, actual_next)
        else:
            probabilities = model.predict_with_probabilities(current_sequence)
            predicted_next = probabilities.index(max(probabilities))
            probability = probabilities[actual_next]
        final_epoch_loss = model.final_epoch_loss
        predictions.append(predicted_next)
        actuals.append(actual_next)
        probabilities_list.append(probability)
        sequences.append(current_sequence)

    return predictions, actuals, probabilities_list, sequences, final_epoch_loss


def word2vec_similarity(model, word1, word2):
    # print(word1)
    # print(word2)
    # word1 = int(word1)
    # word2 = int(word2)
    if word1 in model.wv.index_to_key and word2 in model.wv.index_to_key:
        similarity = 1 - cosine(model.wv[word1], model.wv[word2])  # Cosine similarity is 1 - cosine distance
        return similarity if similarity > 0.0 else 0.0
    else:
        return 0.0  # Return 0 similarity if any word is not in the vocabulary


def calculate_total_similarity(predictions, actuals, word2vec_model):
    return sum(word2vec_similarity(word2vec_model, a, p) for p, a in zip(predictions, actuals))


def calculate_accuracy(predictions, actuals):
    return sum(p == a for p, a in zip(predictions, actuals)) / len(predictions) if predictions else 0


def calculate_total_log_likelihood(probabilities):
    return sum(math.log(p) for p in probabilities if p > 0)


def calculate_perplexity(avg_log_likelihood):
    return math.exp(-avg_log_likelihood) if avg_log_likelihood else float('inf')


def print_predictions(predicted_values, actual_values, sequences, vocab, target_feature):
    vocab_inv = vocab if not target_feature else vocab[target_feature]
    print("First 20 Predictions vs Actual Values:")
    for i, (pred, actual, seq) in enumerate(zip(predicted_values, actual_values, sequences[:20])):
        if isinstance(seq[0], list):
            seq = seq[0]
        readable_sequence = [vocab_inv.get(item, 'UNK') for item in seq]
        print(
            f"{i + 1:02d}. Sequence: {readable_sequence}, \nPredicted: {vocab_inv.get(pred, 'UNK')}, "
            f"\nActual: {vocab_inv[actual]}")


def train_and_save_word2vec(sentences, dataset_name, models_dir="./"):
    # Ensure the directory for models exists
    os.makedirs(models_dir, exist_ok=True)

    # Define the path for the model based on the dataset name
    model_path = os.path.join(models_dir, f"word2vec_{dataset_name}.model")

    # Train the Word2Vec model
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Save the model
    model.save(model_path)

    print(f"Word2Vec model saved for '{dataset_name}' at '{model_path}'")
    return model


def load_word2vec_model(dataset_name, models_dir="./"):
    model_path = os.path.join(models_dir, f"word2vec_{dataset_name}.model")
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
        print(f"Loaded Word2Vec model from '{model_path}'")
        return model
    else:
        raise FileNotFoundError(f"No pre-trained Word2Vec model found at '{model_path}'. Please train one first.")
