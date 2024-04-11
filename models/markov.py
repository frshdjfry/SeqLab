from collections import defaultdict
import random

from models.model_interface import BaseModel


class MarkovModel(BaseModel):
    def __init__(self, alpha, **kwargs):
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.alpha = alpha
        self.final_epoch_loss = 0

    def train_model(self, encoded_seqs, **kwargs):
        for seq in encoded_seqs:
            for i in range(len(seq) - 1):
                current_chord = seq[i]
                next_chord = seq[i + 1]
                self.transition_matrix[current_chord][next_chord] += 1

        for current_chord, next_chords in self.transition_matrix.items():
            total_transitions = sum(next_chords.values()) + len(next_chords) * self.alpha
            for next_chord in next_chords:
                self.transition_matrix[current_chord][next_chord] = (self.transition_matrix[current_chord][
                                                                         next_chord] + self.alpha) / total_transitions

    def predict(self, sequence):
        current_chord = sequence[-1]
        if current_chord not in self.transition_matrix:
            return 0

        next_chords = self.transition_matrix[current_chord]
        next_chord = random.choices(list(next_chords.keys()), weights=next_chords.values())[0]
        return next_chord

    def get_transition_probability(self, current_chord, next_chord):
        return self.transition_matrix[current_chord].get(next_chord, 0)
