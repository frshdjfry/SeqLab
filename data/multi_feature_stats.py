from typing import List, Dict
from collections import Counter
import os
from jinja2 import Environment, FileSystemLoader


# Helper functions
def get_sentence_count(data: Dict[str, List[List[int]]]) -> int:
    feature = next(iter(data))
    return len(data[feature])


def get_avg_sentence_length(data: Dict[str, List[List[int]]]) -> Dict[str, float]:
    avg_lengths = {}
    for feature, sentences in data.items():
        lengths = [len(sentence) for sentence in sentences]
        avg_lengths[feature] = sum(lengths) / len(lengths)
    return avg_lengths


def get_sentence_duplicate_count(data: Dict[str, List[List[int]]]) -> Dict[str, int]:
    duplicate_counts = {}
    for feature, sentences in data.items():
        duplicates = len(sentences) - len(set(map(tuple, sentences)))
        duplicate_counts[feature] = duplicates
    return duplicate_counts


def get_vocab_length(vocab: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    return {feature: len(vocab[feature]) for feature in vocab}


def get_vocab_richness(data: Dict[str, List[List[int]]], vocab: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    richness = {}
    for feature, sentences in data.items():
        word_counts = Counter(word for sentence in sentences for word in sentence)
        richness[feature] = len(word_counts) / len(vocab[feature])
    return richness


def get_train_test_percent(train_data: Dict[str, List[List[int]]], test_data: Dict[str, List[List[int]]]) -> float:
    feature = next(iter(train_data))
    total = len(train_data[feature]) + len(test_data[feature])
    return len(train_data[feature]) / total * 100


def get_longest_sentences(data: List[List[int]], top_n: int = 5) -> List[List[int]]:
    return sorted(data, key=len, reverse=True)[:top_n]


def decode_sentences(sentences: List[List[int]], inv_vocab: Dict[int, str]) -> List[str]:
    return [' '.join(inv_vocab[word_idx] if word_idx in inv_vocab.keys() else '<UNK>' for word_idx in sentence) for sentence in sentences]


def get_most_frequent_words_decoded(data: List[List[int]], inv_vocab: Dict[int, str], top_n: int = 10) -> List[str]:
    word_counts = Counter(word for sentence in data for word in sentence)
    most_common = word_counts.most_common(top_n)
    return [inv_vocab[word_idx] if word_idx in inv_vocab.keys() else '<UNK>' for word_idx, _ in most_common]


def get_least_frequent_words_decoded(data: List[List[int]], inv_vocab: Dict[int, str], top_n: int = 10) -> List[str]:
    word_counts = Counter(word for sentence in data for word in sentence)
    least_common = word_counts.most_common()[:-top_n - 1:-1]
    return [inv_vocab[word_idx] if word_idx in inv_vocab.keys() else '<UNK>' for word_idx, _ in least_common]


def generate_html_stat_file(desc_stats: dict, sample_data: dict, word_freq_data: dict, sentence_lengths: list) -> str:
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('data/templates/multi_feature_stats_template.html')
    html_content = template.render(
        desc_stats=desc_stats,
        sample_data=sample_data,
        word_freq_data=word_freq_data,
        sentence_lengths=sentence_lengths
    )
    html_file_path = 'data/templates/dataset_stats.html'
    with open(html_file_path, 'w') as f:
        f.write(html_content)
    return html_file_path


# Main functions
def get_basic_descriptive_stats(train_data: Dict[str, List[List[int]]], test_data: Dict[str, List[List[int]]], vocab: dict) -> dict:
    full_data = {feature: train_data[feature] + test_data[feature] for feature in train_data}
    return {
        'train_sentence_count': get_sentence_count(train_data),
        'train_sentence_avg_length': get_avg_sentence_length(train_data),
        'train_sentence_duplicate_count': get_sentence_duplicate_count(train_data),
        'test_sentence_count': get_sentence_count(test_data),
        'test_sentence_avg_length': get_avg_sentence_length(test_data),
        'test_sentence_duplicate_count': get_sentence_duplicate_count(test_data),
        'vocab_length': get_vocab_length(vocab),
        'vocab_richness': get_vocab_richness(full_data, vocab),
        'train_test_percent': get_train_test_percent(train_data, test_data),
    }


def get_sample_data(train_data: Dict[str, List[List[int]]], test_data: Dict[str, List[List[int]]], vocab: dict) -> dict:
    full_data = {feature: train_data[feature] + test_data[feature] for feature in train_data}
    vocab_inv = {feature: {v: k for k, v in vocab[feature].items()} for feature in vocab}
    encoded_samples = {feature: get_longest_sentences(train_data[feature]) for feature in train_data}
    decoded_samples = {feature: decode_sentences(get_longest_sentences(train_data[feature]), vocab_inv[feature]) for feature in train_data}
    most_frequent_words = {feature: get_most_frequent_words_decoded(full_data[feature], vocab_inv[feature]) for feature in full_data}
    least_frequent_words = {feature: get_least_frequent_words_decoded(full_data[feature], vocab_inv[feature]) for feature in full_data}
    return {
        "encoded_samples": encoded_samples,
        "decoded_samples": decoded_samples,
        "most_frequent_words": most_frequent_words,
        "least_frequent_words": least_frequent_words,
    }


def get_distribution_data(train_data: Dict[str, List[List[int]]], test_data: Dict[str, List[List[int]]]) -> dict:
    full_data = {feature: train_data[feature] + test_data[feature] for feature in train_data}
    sentence_lengths = [len(sentence) for feature in full_data for sentence in full_data[feature]]
    word_freq_data = {}
    for feature in full_data:
        word_counts = Counter(word for sentence in full_data[feature] for word in sentence)
        word_freq_data[feature] = {
            "word_frequencies": list(word_counts.values())
        }
    return {
        "sentence_lengths": sentence_lengths,
        "word_freq_data": word_freq_data
    }


def get_html_multi_feature_dataset_stats(encoded_train_data: Dict[str, List[List[int]]], encoded_test_data: Dict[str, List[List[int]]], vocab: dict) -> str:
    desc_stats = get_basic_descriptive_stats(encoded_train_data, encoded_test_data, vocab)
    sample_data = get_sample_data(encoded_train_data, encoded_test_data, vocab)
    distribution_data = get_distribution_data(encoded_train_data, encoded_test_data)
    html_file_path = generate_html_stat_file(desc_stats, sample_data, distribution_data["word_freq_data"], distribution_data["sentence_lengths"])
    return html_file_path
