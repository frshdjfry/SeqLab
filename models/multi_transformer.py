import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class MultiTransformerModel(nn.Module):
    def __init__(self, feature_vocabs, target_feature, embedding_dim, nhead=8, num_layers=3, dim_feedforward=512,
                 lr=0.001):
        super(MultiTransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.target_feature = target_feature
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        feature_dims = self.extract_dims(feature_vocabs)

        # Embedding layers for each feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(feature_dim, embedding_dim).to(self.device) for feature_dim in feature_dims
        ])

        # Transformer Encoder Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * len(feature_vocabs),  # Adjusted for concatenated features
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        ).to(self.device)

        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers).to(self.device)
        self.transformer = nn.Transformer(
            d_model=embedding_dim * len(feature_vocabs),  # Assuming all features are projected to the same dimension
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        ).to(self.device)

        self.num_classes = len(feature_vocabs[self.target_feature]) + 1

        # Output fully connected layer, adjust according to your output requirements
        self.fc = nn.Linear(embedding_dim * len(feature_vocabs), self.num_classes).to(
            self.device)  # Adjust output_size accordingly

    def train_model(self, encoded_seqs_dict, epochs=10, batch_size=64):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        dataset = self.prepare_dataset(encoded_seqs_dict)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()  # Set the model to training mode
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

    def forward(self, inputs):
        embedded_features = []
        for i, embedding in enumerate(self.embeddings):
            embedded = embedding(inputs[:, i])
            embedded_features.append(embedded)
        concatenated_embeddings = torch.cat(embedded_features, dim=-1)
        tgt_dummy = torch.zeros_like(concatenated_embeddings)
        transformed = self.transformer(concatenated_embeddings, tgt_dummy)

        # Transformer Encoder processing
        # transformed = self.transformer_encoder(concatenated_embeddings)
        last_timestep_output = transformed[:, -1, :]
        # Output layer
        output = self.fc(last_timestep_output)
        return output

    def predict(self, features):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            # Prepare the features for the model
            prepared_features = [torch.tensor(feature, dtype=torch.long).to(self.device) for feature in features]
            # If necessary, pad and stack the feature sequences
            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in
                                      prepared_features]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)  # Shape: [1, num_features, seq_len]
            # Forward pass through the model
            outputs = self.forward(input_tensor)
            # # Convert outputs to predictions
            _, predicted = torch.max(outputs[:, -1, :], dim=1)
            return predicted.item()

    def predict_with_probabilities(self, features):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            # Prepare the features for the model
            prepared_features = [torch.tensor(feature, dtype=torch.long).to(self.device) for feature in features]
            # If necessary, pad and stack the feature sequences
            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in
                                      prepared_features]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)  # Shape: [1, num_features, seq_len]
            # Forward pass through the model
            outputs = self.forward(input_tensor)
            # Calculate probabilities
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.squeeze().tolist()

    def extract_dims(self, vocabs):
        feature_dims = []

        for feature, vocab in vocabs.items():
            vocab_size = len(vocab)  # Get the size of the vocabulary
            feature_dims.append(vocab_size + 1)  # Add 1 to account for the 0 padding index

        return feature_dims
