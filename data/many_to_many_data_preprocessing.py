import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_many_to_many_data(filename, required_features):
    # Read the CSV into a DataFrame
    df = pd.read_csv(filename)

    # Initialize a dictionary to store sequences for each feature
    sequences = {feature: [] for feature in required_features}
    current_sequences = {feature: [] for feature in required_features}

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        if str(row['**harte']).startswith('*>'):
            # If current sequences are not empty, append them to their respective lists
            if any(len(current_sequences[feature]) > 1 for feature in required_features):
                for feature in required_features:
                    sequences[feature].append(current_sequences[feature])
                    current_sequences[feature] = []
        else:
            # Extract and append the required features for this row to the current sequence
            for feature in required_features:
                if feature in row:
                    current_sequences[feature].append(row[feature])

    # Append the last sequences if not empty
    for feature in required_features:
        if current_sequences[feature]:
            sequences[feature].append(current_sequences[feature])

    # Encoding the sequences for each feature
    vocabs = {}
    vocabs_inv = {}
    encoded_seqs = {feature: [] for feature in required_features}

    for feature in required_features:
        unique_values = sorted(set(value for seq in sequences[feature] for value in seq))
        vocab = {value: i + 1 for i, value in enumerate(unique_values)}
        vocab_inv = {i + 1: value for value, i in vocab.items()}
        vocabs[feature] = vocab
        vocabs_inv[feature] = vocab_inv
        encoded_seqs[feature] = [[vocab[value] for value in seq] for seq in sequences[feature]]

    return encoded_seqs, vocabs, vocabs_inv


def split_multi_feature_data(encoded_seqs, test_size=0.1):
    train_data = {feature: [] for feature in encoded_seqs}
    test_data = {feature: [] for feature in encoded_seqs}

    # Assuming all features have the same number of sequences and alignment by index
    num_sequences = len(next(iter(encoded_seqs.values())))  # Get the length of sequences from the first feature

    # Create a unified index list and split it
    indices = range(num_sequences)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

    # Split sequences for each feature based on the train/test indices
    for feature, sequences in encoded_seqs.items():
        train_data[feature] = [sequences[i] for i in train_indices]
        test_data[feature] = [sequences[i] for i in test_indices]

    return train_data, test_data
