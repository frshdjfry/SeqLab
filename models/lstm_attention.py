import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from models.model_interface import BaseModel


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn = nn.Linear(self.hidden_dim, 1).to(self.device)

    def forward(self, lstm_output):
        attention_scores = torch.softmax(self.attn(lstm_output).squeeze(2), dim=1)

        context_vector = torch.bmm(attention_scores.unsqueeze(1), lstm_output).squeeze(1)

        return context_vector, attention_scores


class LSTMModelWithAttention(nn.Module):
    def __init__(self, vocab, embedding_dim=128, hidden_dim=256, num_layers=2, lr=0.001, **kwargs):
        super(LSTMModelWithAttention, self).__init__()
        vocab_size = len(vocab) + 1
        self.vocab = vocab
        self.config = {
            'class_name': self.__class__.__name__,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'lr': lr
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True).to(self.device)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size).to(self.device)
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.final_epoch_loss = None

    def forward(self, x):
        x = self.embedding(x.to(self.device))
        lstm_out, _ = self.lstm(x)

        context_vector, attention_weights = self.attention(lstm_out)

        out = self.fc(context_vector)
        return out

    def train_model(self, encoded_seqs, validation_encoded_seqs, epochs=10, batch_size=64, patience=20, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        train_dataset = self.prepare_dataset(encoded_seqs)
        validation_dataset = self.prepare_dataset(validation_encoded_seqs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = None

        for epoch in range(epochs):
            self.train_mode()
            total_train_loss = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = self.evaluate(validation_loader)

            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if best_model_path is not None:
                    os.remove(best_model_path)  # Delete the previous best model file
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                best_model_path = f"./saved_models/{self.__class__.__name__}_{timestamp}.pth"
                self.save_model(model_path=best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    self.final_epoch_loss = best_val_loss
                    break

            if epoch == epochs - 1:
                self.final_epoch_loss = best_val_loss

        return best_model_path

    def evaluate(self, data_loader):
        self.eval_mode()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict(self, sequence):
        self.eval_mode()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long).to(self.device)
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs, -1)
            return predicted.item()

    def predict_with_probabilities(self, sequence):
        self.eval_mode()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long).to(self.device)
            outputs = self.forward(inputs)
            probabilities = torch.softmax(outputs, dim=-1)
            return probabilities.squeeze().tolist()

    def prepare_dataset(self, encoded_seqs):
        inputs, targets = [], []
        for seq in encoded_seqs:
            inputs.append(torch.tensor(seq[:-1], dtype=torch.long).to(self.device))
            targets.append(seq[-1])

        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.tensor(targets, dtype=torch.long).to(self.device)
        return TensorDataset(inputs_padded, targets)

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()

    def save_model(self, model_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab': self.vocab,
            'config': self.config
        }, model_path)
