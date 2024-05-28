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


def remove_duplicates_list(data):
    unique_data = list(set(tuple(seq) for seq in data))
    unique_data = [list(seq) for seq in unique_data]
    return unique_data


def remove_duplicates_from_dict(data):
    # Combine corresponding elements across all keys into tuples
    combined = list(zip(*[data[key] for key in data]))

    # Remove duplicates
    unique_combined = list(set(tuple(map(tuple, item)) for item in combined))

    # Separate back into dictionary format
    unique_data = {key: [] for key in data}
    for item in unique_combined:
        for idx, key in enumerate(data):
            unique_data[key].append(list(item[idx]))

    return unique_data


def extract_subsequences(sequence, min_length=2):
    subsequences = []
    n = len(sequence)
    for length in range(min_length, n + 1):
        for start in range(n - length + 1):
            subsequences.append(sequence[start:start + length])
    return subsequences


def combine_features(data):
    combined_data = list(zip(*[data[key] for key in data]))
    combined_data = [list(zip(*item)) for item in combined_data]
    return combined_data


def extract_subsequences_from_dict(data):
    combined_data = combine_features(data)

    all_subsequences = []
    for seq in combined_data:
        all_subsequences.extend(extract_subsequences(seq))

    # Convert back to the original dictionary format
    augmented_data = {key: [] for key in data}
    for subseq in all_subsequences:
        for key, items in zip(data.keys(), zip(*subseq)):
            augmented_data[key].append(list(items))

    return augmented_data


def extract_subsequences_from_list(data):
    # Extract sub-sequences from all sequences
    all_subsequences = []
    for seq in data:
        all_subsequences.extend(extract_subsequences(seq))
    return all_subsequences


def preprocess_txt_dataset(dataset_name):
    encoded_seqs, vocab, vocab_inv = preprocess_data(dataset_name)
    word2vec_model = train_and_save_word2vec(encoded_seqs, dataset_name)
    avg_seq_len = get_avg_seq_len_single(encoded_seqs)
    unique_encoded_seqs = remove_duplicates_list(encoded_seqs)
    augmented_data = extract_subsequences_from_list(unique_encoded_seqs)
    unique_augmented_data = remove_duplicates_list(augmented_data)
    return unique_augmented_data, word2vec_model, vocab, avg_seq_len


def preprocess_csv_dataset(dataset_name, architecture_config, architecture_name):
    if architecture_name == 'one_to_one':
        # Process as single feature dataset
        encoded_seqs, vocabs, vocabs_inv = preprocess_many_to_many_data(
            dataset_name,
            [architecture_config['target_feature']],
            architecture_config['target_feature']
        )
        vocab = vocabs[architecture_config['target_feature']]
        encoded_seqs = encoded_seqs[architecture_config['target_feature']]
        avg_seq_len = get_avg_seq_len_single(encoded_seqs)
        word2vec_model = train_and_save_word2vec(encoded_seqs, dataset_name)
    else:
        # Process as multi-feature dataset
        encoded_seqs, vocab, vocabs_inv = preprocess_many_to_many_data(
            dataset_name,
            architecture_config['source_features'],
            architecture_config['target_feature']
        )
        avg_seq_len = get_avg_seq_len_multi(encoded_seqs)
        word2vec_model = train_and_save_word2vec(encoded_seqs[architecture_config['target_feature']], dataset_name)

    unique_encoded_seqs = remove_duplicates_from_dict(encoded_seqs)
    augmented_data = extract_subsequences_from_dict(unique_encoded_seqs)
    unique_augmented_data = remove_duplicates_from_dict(augmented_data)
    return unique_augmented_data, word2vec_model, vocab, avg_seq_len


def train_and_save_word2vec(sentences, dataset_name, models_dir="./"):
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"word2vec_{dataset_name}.model")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save(model_path)
    print(f"Word2Vec model saved for '{dataset_name}' at '{model_path}'")
    return model
