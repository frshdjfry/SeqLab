import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from models.model_interface import BaseModel


class MultiLSTMModel(BaseModel, nn.Module):
    def __init__(self, vocab, target_feature, embedding_dim, hidden_dim=256, num_layers=2, lr=0.001, **kwargs):
        super(MultiLSTMModel, self).__init__()
        feature_dims = self.extract_dims(vocab)
        self.vocab = vocab
        self.config = {
            'class_name': self.__class__.__name__,
            'target_feature': target_feature,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'lr': lr
        }

        self.target_feature = target_feature
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = nn.ModuleList([
            nn.Embedding(feature_dim, embedding_dim).to(self.device) for feature_dim in
            feature_dims
        ])

        self.lstms = nn.ModuleList([
            nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True).to(self.device) for _ in
            range(len(vocab))
        ])

        self.fc = nn.Linear(hidden_dim * len(feature_dims), max(feature_dims)).to(
            self.device)

        self.lr = lr
        self.hidden_dim = hidden_dim
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.final_epoch_loss = None

    def forward(self, inputs):
        lstm_outputs = []
        for i, lstm in enumerate(self.lstms):
            embedded = self.embeddings[i](inputs[:, i])
            lstm_out, _ = lstm(embedded)
            lstm_outputs.append(lstm_out[:, -1, :])

        concatenated = torch.cat(lstm_outputs, dim=1)

        final_output = self.fc(concatenated)
        return final_output

    def train_model(self, encoded_seqs_dict, validation_encoded_seqs, epochs=10, batch_size=64, patience=20, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        dataset = self.prepare_dataset(encoded_seqs_dict)
        validation_dataset = self.prepare_dataset(validation_encoded_seqs)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = None

        for epoch in range(epochs):
            self.train_mode()
            total_loss = 0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs.to(self.device))
                loss = self.criterion(outputs, targets.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
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
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def prepare_dataset(self, encoded_seqs_dict):
        feature_tensors = []
        target_values = []
        for feature, sequences in encoded_seqs_dict.items():

            feature_tensor = [torch.tensor(sequence[:-1], dtype=torch.long) for sequence in sequences]

            padded_feature_tensor = pad_sequence(feature_tensor, batch_first=True)

            feature_tensors.append(padded_feature_tensor)
            if feature == self.target_feature:
                target_values = [sequence[-1] for sequence in sequences]

        input_tensor = torch.stack(feature_tensors, dim=1)

        target_tensor = torch.tensor(target_values, dtype=torch.long)

        dataset = TensorDataset(input_tensor, target_tensor)

        return dataset

    def predict(self, sequence):
        self.eval_mode()
        with torch.no_grad():
            feature_tensors = [torch.tensor(feat_sequence, dtype=torch.long).to(self.device) for feat_sequence in
                               sequence]

            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in feature_tensors]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)

            outputs = self.forward(input_tensor)
            _, predicted = torch.max(outputs, -1)
            return predicted.item()

    def predict_with_probabilities(self, sequence):
        self.eval_mode()
        with torch.no_grad():
            feature_tensors = [torch.tensor(feat_sequence, dtype=torch.long).to(self.device) for feat_sequence in
                               sequence]
            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in feature_tensors]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)

            outputs = self.forward(input_tensor)
            probabilities = torch.softmax(outputs, dim=-1)
            return probabilities.squeeze().tolist()

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()

    def extract_dims(self, vocabs):
        feature_dims = []

        for feature, vocab in vocabs.items():
            vocab_size = len(vocab)
            feature_dims.append(vocab_size + 1)

        return feature_dims

    def save_model(self, model_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab': self.vocab,
            'config': self.config
        }, model_path)
