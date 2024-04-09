import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Config


class MultiGPTModel(nn.Module):
    def __init__(self, feature_vocabs, target_feature, nhead, embedding_dim, num_layers, dim_feedforward,
                 gpt2_model_name='gpt2', lr=0.001):
        super(MultiGPTModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.target_feature = target_feature
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        feature_dims = self.extract_dims(feature_vocabs)
        self.config = GPT2Config(
            vocab_size=sum(feature_dims),
            n_layer=num_layers,
            n_head=nhead,
            n_inner=dim_feedforward,
            pad_token_id=0
        )
        self.gpt2 = GPT2LMHeadModel(self.config).to(self.device)

        self.num_classes = len(feature_vocabs[self.target_feature]) + 1  # Define this based on your dataset
        self.fc = nn.Linear(self.gpt2.config.n_embd, self.num_classes).to(self.device)
        self.final_epoch_loss = 0.0
        self.feature_projection = None
        self.max_lengths = {}

    def forward(self, inputs):
        # Project the input features
        projected_features = self.feature_projection(inputs.float())

        # Process through GPT-2

        gpt2_outputs = self.gpt2(inputs_embeds=projected_features, output_hidden_states=True)
        last_hidden_states = gpt2_outputs.hidden_states[-1]

        # We use the last hidden state for the classification task
        logits = self.fc(last_hidden_states[:, -1, :])

        return logits

    def train_model(self, encoded_seqs_dict, epochs=10, batch_size=64):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.max_lengths = {feature: max(len(seq) for seq in sequences) for feature, sequences in
                            encoded_seqs_dict.items()}
        self.feature_projection = nn.Linear(self.max_lengths[self.target_feature] - 1, self.gpt2.config.n_embd).to(
            self.device)  # -1 for predicting last one

        dataset = self.prepare_dataset(encoded_seqs_dict)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
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
        return TensorDataset(input_tensor, target_tensor)

    def predict(self, features):
        self.eval()
        with torch.no_grad():
            prepared_features = self.prepare_features(features)
            outputs = self.forward(prepared_features)
            _, predicted = torch.max(outputs, dim=1)
            return predicted.item()

    def predict_with_probabilities(self, features):
        self.eval()
        with torch.no_grad():
            prepared_features = self.prepare_features(features)
            outputs = self.forward(prepared_features)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.squeeze().tolist()

    def prepare_features(self, features):
        # Assume 'features' is a list of lists, where each inner list is a feature vector
        prepared_features = [torch.tensor(feature, dtype=torch.long).to(self.device) for feature in features]

        padded_feature_tensors = [pad_sequence([feat], batch_first=True, padding_value=0).to(self.device) for feat in
                                  prepared_features]
        max_length = self.max_lengths[self.target_feature] - 1

        padded_and_reshaped_tensors = [
            feat_tensor if feat_tensor.shape[1] == max_length else torch.nn.functional.pad(feat_tensor,
                                                                                           (0, max_length -
                                                                                            feat_tensor.shape[1]),
                                                                                           "constant", 0) for
            feat_tensor in
            padded_feature_tensors]
        input_tensor = torch.stack(padded_and_reshaped_tensors, dim=1)

        return input_tensor

    def extract_dims(self, vocabs):
        feature_dims = []

        for feature, vocab in vocabs.items():
            vocab_size = len(vocab)  # Get the size of the vocabulary
            feature_dims.append(vocab_size + 1)  # Add 1 to account for the 0 padding index

        return feature_dims
