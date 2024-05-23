from collections import defaultdict
import random
from datetime import datetime

import torch

from models.model_interface import BaseModel


class VariableOrderMarkovModel(BaseModel):
    def __init__(self, vocab, max_context_length=3, alpha=0.01, **kwargs):
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.max_context_length = max_context_length
        self.alpha = alpha
        self.vocab = vocab
        self.config = {
            'class_name': self.__class__.__name__,
            'alpha': self.alpha,
            'max_context_length': self.max_context_length
        }
        self.final_epoch_loss = 0

    def train_model(self, encoded_seqs, validation_encoded_seqs, **kwargs):
        for seq in encoded_seqs:
            for i in range(len(seq)):
                for context_length in range(1, min(self.max_context_length + 1, i + 1)):
                    context = tuple(seq[i-context_length:i])
                    next_chord = seq[i]
                    self.transition_matrix[context][next_chord] += 1

        # Smoothing and normalizing transition probabilities
        for context, next_chords in self.transition_matrix.items():
            total_transitions = sum(next_chords.values()) + len(next_chords) * self.alpha
            for next_chord in next_chords:
                self.transition_matrix[context][next_chord] = (self.transition_matrix[context][next_chord] + self.alpha) / total_transitions

        return self.save_model()

    def predict_with_probability(self, sequence):
        for context_length in range(min(self.max_context_length, len(sequence)), 0, -1):
            context = tuple(sequence[-context_length:])
            if context in self.transition_matrix:
                next_chords = self.transition_matrix[context]
                next_chord = random.choices(list(next_chords.keys()), weights=next_chords.values())[0]
                next_chord_probability = next_chords[next_chord]
                return next_chord, next_chord_probability
        return 0, 0

    def save_model(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        model_path = f"./saved_models/{self.__class__.__name__}_{timestamp}.pth"
        torch.save({
            'model_state_dict': dict(self.transition_matrix),
            'vocab': self.vocab,
            'config': self.config,
        }, model_path)
        return model_path

    def load_state_dict(self, state_dict):
        self.transition_matrix = defaultdict(lambda: defaultdict(int), state_dict)
