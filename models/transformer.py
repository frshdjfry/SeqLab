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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, embed_size).to(self.device)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True).to(self.device)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers).to(self.device)
        self.fc = nn.Linear(embed_size, vocab_size).to(self.device)

    def forward(self, src):
        src = src.to(self.device)  # Move input to the device
        embedded = self.embedding(src)
        transformed = self.transformer(embedded)
        output = self.fc(transformed)
        return output

class TransformerModel(BaseModel):
    def __init__(self, vocab_size, embed_size=128, nhead=8, num_layers=3, dim_feedforward=512, lr=0.001):
        self.model = ChordPredictor(
            vocab_size, embed_size=embed_size, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
        self.criterion = nn.CrossEntropyLoss().to(self.model.device)  # Use device from ChordPredictor
        self.optimizer = None
        self.lr = lr
        self.final_epoch_loss = None

    def train_model(self, encoded_seqs, epochs=10, batch_size=64):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = self.prepare_dataset(encoded_seqs)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)  # Move data to the device
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs[:, -1, :], targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

            if epoch == epochs - 1:
                self.final_epoch_loss = total_loss / len(train_loader)

    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long).to(self.model.device)  # Move input to the device
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs[:, -1, :], dim=1)
            return predicted.item()

    def predict_with_probabilities(self, sequence):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long).to(self.model.device)  # Move input to the device
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs[:, -1, :], dim=1)
            return probabilities.squeeze().tolist()

    def prepare_dataset(self, encoded_seqs):
        inputs, targets = [], []
        for seq in encoded_seqs:
            inputs.append(torch.tensor(seq[:-1], dtype=torch.long).to(self.model.device))  # Move data to the device
            targets.append(seq[-1])

        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.tensor(targets, dtype=torch.long).to(self.model.device)  # Move data to the device
        return TensorDataset(inputs_padded, targets)
