# data_preprocessing.py
import os

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


def preprocess_data(filename):
    with open(filename, 'r') as file:
        chord_sequences = [line.strip().split() for line in file.readlines()]
    vocab = {chord: i+1 for i, chord in enumerate(set(chord for seq in chord_sequences for chord in seq))}
    vocab_inv = {i+1: chord for chord, i in vocab.items()}
    encoded_seqs = [[vocab[chord] for chord in seq] for seq in chord_sequences]
    return encoded_seqs, vocab, vocab_inv


def split_data(encoded_seqs, test_size=0.1):
    train_data, test_data = train_test_split(encoded_seqs, test_size=test_size, random_state=42)
    return train_data, test_data


def train_word2vec(dataset, dataset_name):
    # 'dataset' is a list of lists, where each inner list is a sequence of elements (words, chords, etc.)
    model = Word2Vec(sentences=dataset, vector_size=100, window=5, min_count=1, workers=4)

    # Create a unique filename for each model based on the dataset name
    model_filename = f"word2vec_model_{dataset_name}.model"
    model.save(os.path.join("models", model_filename))  # Saving in a 'models' directory
    return model
