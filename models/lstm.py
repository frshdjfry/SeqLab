import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from models.model_interface import BaseModel


class LSTMModel(BaseModel, nn.Module):
    def __init__(self, vocab, embedding_dim=128, hidden_dim=256, num_layers=2, lr=0.001, **kwargs):
        super(LSTMModel, self).__init__()
        vocab_size = len(vocab) + 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_dim, vocab_size).to(self.device)
        self.lr = lr

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None

        self.final_epoch_loss = None

    def forward(self, x):
        x = self.embedding(x.to(self.device))
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1])
        return out

    def train_model(self, encoded_seqs, epochs=10, batch_size=64, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        dataset = self.prepare_dataset(encoded_seqs)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train_mode()
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

            if epoch == epochs - 1:
                self.final_epoch_loss = total_loss / len(train_loader)

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
