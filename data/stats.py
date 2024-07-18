from typing import List, Dict
from collections import Counter
import os
from jinja2 import Environment, FileSystemLoader


# Helper functions
def get_sentence_count(data: List[List[int]]) -> int:
    return len(data)


def get_avg_sentence_length(data: List[List[int]]) -> float:
    lengths = [len(sentence) for sentence in data]
    return sum(lengths) / len(lengths)


def get_sentence_duplicate_count(data: List[List[int]]) -> int:
    duplicates = len(data) - len(set(map(tuple, data)))
    return duplicates


def get_vocab_length(vocab: Dict[str, int]) -> int:
    return len(vocab)


def get_vocab_richness(data: List[List[int]], vocab: Dict[str, int]) -> float:
    word_counts = Counter(word for sentence in data for word in sentence)
    richness = len(word_counts) / len(vocab)
    return richness


def get_train_test_percent(train_data: List[List[int]], test_data: List[List[int]]) -> float:
    total = len(train_data) + len(test_data)
    return len(train_data) / total * 100


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


def generate_html_stat_file(desc_stats: dict, sample_data: dict, word_frequencies: list, sentence_lengths: list) -> str:
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('data/templates/stats_template.html')

    html_content = template.render(
        desc_stats=desc_stats,
        sample_data=sample_data,
        word_frequencies=word_frequencies,
        sentence_lengths=sentence_lengths
    )
    html_file_path = 'data/templates/dataset_stats.html'
    with open(html_file_path, 'w') as f:
        f.write(html_content)
    return html_file_path


# Main functions
def get_basic_descriptive_stats(train_data: List[List[int]], test_data: List[List[int]], vocab: dict) -> dict:
    full_data = train_data + test_data
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


def get_sample_data(train_data: List[List[int]], test_data: List[List[int]], vocab: dict) -> dict:
    full_data = train_data + test_data
    vocab_inv = {v: k for k, v in vocab.items()}
    return {
        "train_longest_encoded_samples": get_longest_sentences(train_data),
        "train_longest_decoded_samples": decode_sentences(get_longest_sentences(train_data), vocab_inv),
        "test_longest_encoded_samples": get_longest_sentences(test_data),
        "test_longest_decoded_samples": decode_sentences(get_longest_sentences(test_data), vocab_inv),
        "most_frequent_words": get_most_frequent_words_decoded(full_data, vocab_inv),
        "least_frequent_words": get_least_frequent_words_decoded(full_data, vocab_inv),
    }


def get_distribution_data(train_data: List[List[int]], test_data: List[List[int]]) -> dict:
    full_data = train_data + test_data
    sentence_lengths = [len(sentence) for sentence in full_data]
    word_counts = Counter(word for sentence in full_data for word in sentence)
    word_frequencies = list(word_counts.values())
    return {
        "sentence_lengths": sentence_lengths,
        "word_frequencies": word_frequencies
    }


def get_html_dataset_stats(encoded_train_data: List[List[int]], encoded_test_data: List[List[int]], vocab: dict) -> str:
    desc_stats = get_basic_descriptive_stats(encoded_train_data, encoded_test_data, vocab)
    sample_data = get_sample_data(encoded_train_data, encoded_test_data, vocab)
    distribution_data = get_distribution_data(encoded_train_data, encoded_test_data)
    html_file_path = generate_html_stat_file(desc_stats, sample_data, distribution_data['word_frequencies'], distribution_data["sentence_lengths"])
    return html_file_path
