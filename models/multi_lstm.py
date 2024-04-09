import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from models.model_interface import BaseModel


class MultiLSTMModel(BaseModel, nn.Module):
    def __init__(self, feature_vocabs, target_feature, embedding_dim, hidden_dim=256, num_layers=2, lr=0.001):
        super(MultiLSTMModel, self).__init__()
        feature_dims = self.extract_dims(feature_vocabs)
        self.target_feature = target_feature
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assuming feature_dims and embedding_dims are lists of the same length, where each element corresponds to a feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(feature_dim, embedding_dim).to(self.device) for feature_dim in
            feature_dims
        ])

        self.lstms = nn.ModuleList([
            nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True).to(self.device) for _ in
            range(len(feature_vocabs))
        ])

        # Adjust the input dimension of the fully connected layer based on the number of features
        self.fc = nn.Linear(hidden_dim * len(feature_dims), max(feature_dims)).to(
            self.device)  # Adjust the output dimension as needed

        self.lr = lr
        self.hidden_dim = hidden_dim
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.final_epoch_loss = None

    def forward(self, inputs):
        lstm_outputs = []
        for i, lstm in enumerate(self.lstms):
            embedded = self.embeddings[i](inputs[:, i])  # Embedding for the i-th feature across all sequences
            lstm_out, _ = lstm(embedded)  # Add a dimension for batch_first=True
            lstm_outputs.append(lstm_out[:, -1, :])  # Getting the last output of the sequence for each batch

        # Concatenate the outputs from all LSTMs
        concatenated = torch.cat(lstm_outputs, dim=1)

        # Pass through the fully connected layer to predict the next value of the target feature
        final_output = self.fc(concatenated)
        return final_output

    def train_model(self, encoded_seqs_dict, epochs=10, batch_size=64):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        dataset = self.prepare_dataset(encoded_seqs_dict)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train_mode()  # Set the model to training mode
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
        # Initialize an empty list to hold padded tensors for each feature
        feature_tensors = []

        # Loop through each feature in the encoded sequences dictionary
        target_values = []
        for feature, sequences in encoded_seqs_dict.items():
            # Convert the list of sequences for this feature into a tensor
            feature_tensor = [torch.tensor(sequence[:-1], dtype=torch.long) for sequence in sequences]

            # Pad the sequences for this feature to ensure equal length
            padded_feature_tensor = pad_sequence(feature_tensor, batch_first=True)

            # Append the padded tensor for this feature to the list of feature tensors
            feature_tensors.append(padded_feature_tensor)
            if feature == self.target_feature:
                target_values = [sequence[-1] for sequence in sequences]
        # Stack the feature tensors along a new dimension to form the input tensor
        input_tensor = torch.stack(feature_tensors, dim=1)  # Shape: [num_samples, num_features, seq_length (padded)]

        # Convert the target_values list into a tensor
        target_tensor = torch.tensor(target_values, dtype=torch.long)  # Shape: [num_samples]

        # Create a TensorDataset from the input tensor and target tensor
        dataset = TensorDataset(input_tensor, target_tensor)

        return dataset

    def predict(self, sequence):
        self.eval_mode()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            # Assume 'sequence' is a list of lists, where each inner list corresponds to a feature's sequence
            feature_tensors = [torch.tensor(feat_sequence, dtype=torch.long).to(self.device) for feat_sequence in
                               sequence]

            # If necessary, pad and stack the feature sequences
            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in feature_tensors]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)  # Shape: [1, num_features, seq_len]

            outputs = self.forward(input_tensor)
            _, predicted = torch.max(outputs, -1)
            return predicted.item()

    def predict_with_probabilities(self, sequence):
        self.eval_mode()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            # Process the input sequence as in the 'predict' method
            feature_tensors = [torch.tensor(feat_sequence, dtype=torch.long).to(self.device) for feat_sequence in
                               sequence]
            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in feature_tensors]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)  # Shape: [1, num_features, seq_len]

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
            vocab_size = len(vocab)  # Get the size of the vocabulary
            feature_dims.append(vocab_size + 1)  # Add 1 to account for the 0 padding index

        return feature_dims
