import os

import math
from gensim.models import Word2Vec

from models.markov import MarkovModel
from scipy.spatial.distance import cosine

def word2vec_similarity(model, word1, word2):
    word1 = int(word1)
    word2 = int(word2)
    if word1 in model.wv.index_to_key and word2 in model.wv.index_to_key:
        similarity = 1 - cosine(model.wv[word1], model.wv[word2])  # Cosine similarity is 1 - cosine distance
        return similarity if similarity > 0.0 else 0.0
    else:
        return 0.0  # Return 0 similarity if any word is not in the vocabulary


def evaluate_perplexity(model, test_data):
    total_log_likelihood = 0
    total_elements = 0

    for sequence in test_data:
        for i in range(1, len(sequence)):
            input_seq = sequence[:i]
            actual_next_chord = sequence[i]
            probability = 0

            if model.__class__ == MarkovModel:
                current_chord = input_seq[-1] if input_seq else None
                if current_chord is not None:
                    probability = model.get_transition_probability(current_chord, actual_next_chord)
            else:
                probabilities = model.predict_with_probabilities(input_seq)
                if probabilities and actual_next_chord < len(probabilities):
                    probability = probabilities[actual_next_chord]

            if probability > 0:
                total_log_likelihood += math.log(probability)
                total_elements += 1

    average_neg_log_likelihood = -total_log_likelihood / total_elements if total_elements > 0 else 0
    perplexity = math.exp(average_neg_log_likelihood)
    return perplexity

#
# def evaluate_model(model, test_data, word2vec_model):
#     correct_predictions = 0
#     total_predictions = 0
#     total_similarity = 0
#
#     for sequence in test_data:
#         current_sequence = sequence[:-1]
#         actual_next = sequence[-1]
#         predicted_next = model.predict(current_sequence)
#
#         # Update accuracy calculations
#         if predicted_next == actual_next:
#             correct_predictions += 1
#         total_predictions += 1
#
#         # Calculate Word2Vec similarity for each prediction
#         similarity = word2vec_similarity(word2vec_model, actual_next, predicted_next)
#         total_similarity += similarity
#
#     # Calculate average similarity
#     avg_similarity = total_similarity / total_predictions if total_predictions > 0 else 0
#     accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
#
#     # Assume calculate_perplexity is defined elsewhere
#     perplexity = evaluate_perplexity(model, test_data)
#
#     return accuracy, avg_similarity, perplexity
#

def evaluate_model(model, test_data, word2vec_model):
    correct_predictions = 0
    total_predictions = 0
    total_similarity = 0
    total_log_likelihood = 0

    for sequence in test_data:
        current_sequence = sequence[:-1]
        actual_next = sequence[-1]
        if len(current_sequence) < 1:
            continue
        # Compute prediction once for all metrics
        predicted_next, probability = None, 0

        # Handling predictions for different model types
        if isinstance(model, MarkovModel):

            if current_sequence:
                current_chord = current_sequence[-1]
                predicted_next = model.predict(current_sequence)
                probability = model.get_transition_probability(current_chord, actual_next)
        else:
            probabilities = model.predict_with_probabilities(current_sequence)
            if probabilities:
                predicted_next = probabilities.index(max(probabilities))  # Assuming this returns the index with max probability
                if actual_next < len(probabilities):
                    probability = probabilities[actual_next]

        # Update metrics
        if predicted_next == actual_next:
            correct_predictions += 1
        total_predictions += 1

        if probability > 0:
            total_log_likelihood += math.log(probability)

        # Ensure predicted_next is not a list for w2v similarity
        if predicted_next is not None and isinstance(predicted_next, list):
            predicted_next = predicted_next[0]

        # Calculate Word2Vec similarity, ensuring valid inputs
        if predicted_next is not None:
            similarity = word2vec_similarity(word2vec_model, str(actual_next), str(predicted_next))
            total_similarity += similarity

    # Calculate final metrics
    avg_similarity = total_similarity / total_predictions if total_predictions > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    average_neg_log_likelihood = -total_log_likelihood / total_predictions if total_predictions > 0 else 0
    perplexity = math.exp(average_neg_log_likelihood) if total_predictions > 0 else float('inf')

    return accuracy, avg_similarity, perplexity

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
