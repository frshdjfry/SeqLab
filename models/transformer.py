# transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from models.model_interface import BaseModel


class ChordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size=128, nhead=8, num_layers=3, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        transformed = self.transformer(embedded)
        output = self.fc(transformed)
        return output


class TransformerModel(BaseModel):
    def __init__(self, vocab_size, embed_size=128, nhead=8, num_layers=3, dim_feedforward=512, lr=0.001, **kwargs):  # Accept num_layers and **kwargs
        self.model = ChordPredictor(vocab_size, embed_size=embed_size, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)  # Use num_layers here
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.lr = lr

    def train_model(self, encoded_seqs, epochs=10, batch_size=64):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = self.prepare_dataset(encoded_seqs)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs[:, -1, :], targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs[:, -1, :], dim=1)
            return predicted.item()

    def predict_with_probabilities(self, sequence):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long)
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs[:, -1, :], dim=1)
            return probabilities.squeeze().tolist()

    def prepare_dataset(self, encoded_seqs):
        inputs, targets = [], []
        for seq in encoded_seqs:
            inputs.append(torch.tensor(seq[:-1], dtype=torch.long))
            targets.append(seq[-1])

        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.tensor(targets, dtype=torch.long)
        return TensorDataset(inputs_padded, targets)
