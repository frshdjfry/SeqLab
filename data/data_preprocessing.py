# data_preprocessing.py
import os

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from data.many_to_many_data_preprocessing import preprocess_many_to_many_data, split_multi_feature_data


def preprocess_data(filename):
    chord_sequences = []
    with open(filename, 'r') as file:
        for line in file:  # Streamline file reading
            chord_sequences.append(line.strip().split())

    # Stabilize chord order and optimize vocab construction
    unique_chords = sorted(set(chord for seq in chord_sequences for chord in seq))
    vocab = {}
    vocab_inv = {}

    for i, chord in enumerate(unique_chords):
        vocab[chord] = i + 1
        vocab_inv[i + 1] = chord

    encoded_seqs = [[vocab[chord] for chord in seq] for seq in chord_sequences]
    return encoded_seqs, vocab, vocab_inv


def split_data(encoded_seqs, test_size=0.1):
    train_data, test_data = train_test_split(encoded_seqs, test_size=test_size, random_state=42)
    return train_data, test_data


def get_avg_seq_len_multi(encoded_seqs):
    sumof = 0.
    count = 0.
    for k, v in encoded_seqs.items():
        for i in v:
            sumof += len(i)
            count += 1
        break
    return sumof / count


def get_avg_seq_len_single(encoded_seqs):
    sumof = 0.
    for seq in encoded_seqs:
        sumof += len(seq)
    return sumof / len(encoded_seqs)


def preprocess_txt_dataset(dataset_name):
    encoded_seqs, vocab, vocab_inv = preprocess_data(dataset_name)
    word2vec_model = train_and_save_word2vec(encoded_seqs, dataset_name)
    avg_seq_len = get_avg_seq_len_single(encoded_seqs)
    return encoded_seqs, word2vec_model, vocab, avg_seq_len


def preprocess_csv_dataset(dataset_name, architecture_config, architecture_name):
    if architecture_name == 'one_to_one':
        # Process as single feature dataset
        encoded_seqs, vocabs, vocabs_inv = preprocess_many_to_many_data(
            dataset_name,
            [architecture_config.target_feature],
            architecture_config.target_feature
        )
        vocab = vocabs[architecture_config.target_feature]
        encoded_seqs = encoded_seqs[architecture_config.target_feature]
        avg_seq_len = get_avg_seq_len_single(encoded_seqs)
        word2vec_model = train_and_save_word2vec(encoded_seqs, dataset_name)
    else:
        # Process as multi-feature dataset
        encoded_seqs, vocab, vocabs_inv = preprocess_many_to_many_data(
            dataset_name,
            architecture_config.source_features,
            architecture_config.target_feature
        )
        avg_seq_len = get_avg_seq_len_multi(encoded_seqs)
        word2vec_model = train_and_save_word2vec(encoded_seqs[architecture_config.target_feature], dataset_name)

    return encoded_seqs, word2vec_model, vocab, avg_seq_len


def train_and_save_word2vec(sentences, dataset_name, models_dir="./"):
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"word2vec_{dataset_name}.model")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save(model_path)
    print(f"Word2Vec model saved for '{dataset_name}' at '{model_path}'")
    return model
