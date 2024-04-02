from collections import defaultdict
import random

from models.model_interface import BaseModel


class MarkovModel(BaseModel):
    def __init__(self, alpha=0.01):
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.vocab = None
        self.vocab_inv = None
        self.alpha = alpha  # Smoothing parameter

    def train_model(self, encoded_seqs):
        for seq in encoded_seqs:
            for i in range(len(seq) - 1):
                current_chord = seq[i]
                next_chord = seq[i + 1]
                self.transition_matrix[current_chord][next_chord] += 1

        for current_chord, next_chords in self.transition_matrix.items():
            total_transitions = sum(next_chords.values()) + len(next_chords) * self.alpha
            for next_chord in next_chords:
                self.transition_matrix[current_chord][next_chord] = (self.transition_matrix[current_chord][next_chord] + self.alpha) / total_transitions

    def predict(self, sequence):
        current_chord = sequence[-1]
        if current_chord not in self.transition_matrix:
            return None

        next_chords = self.transition_matrix[current_chord]
        next_chord = random.choices(list(next_chords.keys()), weights=next_chords.values())[0]
        return next_chord

    def get_transition_probability(self, current_chord, next_chord):
        return self.transition_matrix[current_chord].get(next_chord, 0)

    def set_vocab(self, vocab, vocab_inv):
        self.vocab = vocab
        self.vocab_inv = vocab_inv

    def predict_chord(self, current_chord_name):
        if self.vocab is None or self.vocab_inv is None:
            raise ValueError("Vocabulary not set. Please call set_vocab before predicting.")

        current_chord_id = self.vocab.get(current_chord_name)
        if current_chord_id is None:
            raise ValueError(f"Chord '{current_chord_name}' not found in vocabulary.")

        next_chord_id = self.predict(current_chord_id)
        if next_chord_id is not None:
            return self.vocab_inv[next_chord_id]
        return None

