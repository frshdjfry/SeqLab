import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any

from data.chord_normalizer import normalize_chord_sequence


def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


def initialize_sequences(features: List[str]) -> Dict[str, List]:
    return {feature: [] for feature in features}


def process_row(row: pd.Series, required_features: List[str], target_feature: str, current_sequences: Dict[str, List],
                sequences: Dict[str, List]) -> None:
    if str(row[target_feature]).startswith('*>'):
        if any(len(current_sequences[feature]) > 0 for feature in required_features):
            for feature in required_features:
                if current_sequences[feature]:
                    sequences[feature].append(current_sequences[feature])
                    current_sequences[feature] = []
    else:
        for feature in required_features:
            if feature in row:
                if feature == '**kern':
                    cln_val = row[feature].split()[0]
                    cln_val = cln_val.strip(']').strip('[').strip('.').strip(';')
                    current_sequences[feature].append(cln_val)
                elif feature == '**duration':
                    current_sequences[feature].append(str(int(row[feature])))
                else:
                    current_sequences[feature].append(row[feature])


def extract_sequences(df: pd.DataFrame, required_features: List[str], target_feature: str) -> Dict[
    str, List[List[str]]]:
    sequences = initialize_sequences(required_features)
    current_sequences = initialize_sequences(required_features)

    for _, row in df.iterrows():
        process_row(row, required_features, target_feature, current_sequences, sequences)

    for feature in required_features:
        if current_sequences[feature]:
            sequences[feature].append(current_sequences[feature])

    return sequences


def encode_sequences(sequences: Dict[str, List[List[str]]]) -> Tuple[
    Dict[str, List[List[int]]], Dict[str, Dict[str, int]], Dict[str, Dict[int, str]]]:
    vocabs = {}
    vocabs_inv = {}
    encoded_seqs = initialize_sequences(list(sequences.keys()))

    for feature in sequences:
        unique_values = sorted(set(value for seq in sequences[feature] for value in seq))
        vocab = {value: i + 1 for i, value in enumerate(unique_values)}
        vocab_inv = {i + 1: value for value, i in vocab.items()}
        vocabs[feature] = vocab
        vocabs_inv[feature] = vocab_inv
        encoded_seqs[feature] = [[vocab[value] for value in seq] for seq in sequences[feature]]

    return encoded_seqs, vocabs, vocabs_inv


def preprocess_many_to_many_data(filename: str, required_features: List[str], target_feature: str) -> Tuple[
    Dict[str, List[List[int]]], Dict[str, Dict[str, int]], Dict[str, Dict[int, str]]]:
    df = read_csv(filename)
    sequences = extract_sequences(df, required_features, target_feature)
    encoded_seqs, vocabs, vocabs_inv = encode_sequences(sequences)
    return encoded_seqs, vocabs, vocabs_inv


def split_multi_feature_data(encoded_seqs: Dict[str, List[List[int]]], test_size: float = 0.1) -> Tuple[
    Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
    train_data = initialize_sequences(list(encoded_seqs.keys()))
    test_data = initialize_sequences(list(encoded_seqs.keys()))

    num_sequences = len(next(iter(encoded_seqs.values())))
    indices = range(num_sequences)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

    for feature, sequences in encoded_seqs.items():
        train_data[feature] = [sequences[i] for i in train_indices]
        test_data[feature] = [sequences[i] for i in test_indices]

    return train_data, test_data


def get_merged_m2m_train_test_datasets(
        train_filename: str,
        test_filename: str,
        required_features: List[str],
        target_feature: str, architecture_config
) -> Tuple[Dict[str, List[List[int]]], Dict[str, Dict[str, int]], Dict[str, Dict[int, str]], int]:
    train_df = read_csv(train_filename)
    train_sequences = extract_sequences(train_df, required_features, target_feature)
    test_df = read_csv(test_filename)
    test_sequences = extract_sequences(test_df, required_features, target_feature)

    full_sequences = {}
    for feature in required_features:
        full_sequences[feature] = train_sequences[feature] + test_sequences[feature]

    if architecture_config.normalize_chords:
        if '**harte' in full_sequences.keys():
            chord_sequences = full_sequences['**harte']

            normalized_sequences = []
            for chord_sequence in chord_sequences:
                normalized_sequences.append(normalize_chord_sequence(chord_sequence))
            full_sequences['**harte'] = normalized_sequences

    test_start_index = len(train_sequences[target_feature])

    encoded_seqs, vocabs, vocabs_inv = encode_sequences(full_sequences)
    return encoded_seqs, vocabs, vocabs_inv, test_start_index

