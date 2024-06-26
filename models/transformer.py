import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from models.model_interface import BaseModel


class TransformerModel(BaseModel):
    def __init__(self, vocab, embed_size=128, nhead=8, num_layers=3, dim_feedforward=512, lr=0.001, **kwargs):
        vocab_size = len(vocab) + 1
        self.vocab = vocab
        self.config = {
            'class_name': self.__class__.__name__,
            'embed_size': embed_size,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'lr': lr
        }
        self.model = ChordPredictor(
            vocab_size, embed_size=embed_size, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
        self.criterion = nn.CrossEntropyLoss().to(self.model.device)
        self.optimizer = None
        self.lr = lr
        self.final_epoch_loss = None

    def train_model(self, encoded_seqs, validation_encoded_seqs, epochs=10, batch_size=64, patience=20, **kwargs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = self.prepare_dataset(encoded_seqs)
        validation_dataset = self.prepare_dataset(validation_encoded_seqs)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = None

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs[:, -1, :], targets)
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
                self.final_epoch_loss = avg_train_loss
        return best_model_path

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs[:, -1, :], targets)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long).to(self.model.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs[:, -1, :], dim=1)
            return predicted.item()

    def predict_with_probabilities(self, sequence):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([sequence], dtype=torch.long).to(self.model.device)
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs[:, -1, :], dim=1)
            return probabilities.squeeze().tolist()

    def prepare_dataset(self, encoded_seqs):
        inputs, targets = [], []
        for seq in encoded_seqs:
            inputs.append(torch.tensor(seq[:-1], dtype=torch.long).to(self.model.device))
            targets.append(seq[-1])

        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.tensor(targets, dtype=torch.long).to(self.model.device)
        return TensorDataset(inputs_padded, targets)

    def save_model(self, model_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'config': self.config
        }, model_path)

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)


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
