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

    def train_model(self, encoded_seqs_dict, epochs=10, batch_size=64, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        dataset = self.prepare_dataset(encoded_seqs_dict)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

            if epoch == epochs - 1:
                self.final_epoch_loss = total_loss / len(train_loader)

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
