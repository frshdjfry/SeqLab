import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Config


class MultiGPTModel(nn.Module):
    def __init__(self, vocab, target_feature, nhead, num_layers, dim_feedforward, lr=0.001, **kwargs):
        super(MultiGPTModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.target_feature = target_feature
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        feature_dims = self.extract_dims(vocab)
        self.config = GPT2Config(
            vocab_size=sum(feature_dims),
            n_layer=num_layers,
            n_head=nhead,
            n_inner=dim_feedforward,
            pad_token_id=0
        )
        self.gpt2 = GPT2LMHeadModel(self.config).to(self.device)

        self.num_classes = len(vocab[self.target_feature]) + 1
        self.fc = nn.Linear(self.gpt2.config.n_embd, self.num_classes).to(self.device)
        self.final_epoch_loss = 0.0
        self.feature_projection = None
        self.max_lengths = {}

    def forward(self, inputs):

        projected_features = self.feature_projection(inputs.float())

        gpt2_outputs = self.gpt2(inputs_embeds=projected_features, output_hidden_states=True)
        last_hidden_states = gpt2_outputs.hidden_states[-1]

        logits = self.fc(last_hidden_states[:, -1, :])

        return logits

    def train_model(self, encoded_seqs_dict, validation_encoded_seqs, epochs=10, batch_size=64, patience=10, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.max_lengths = {feature: max(len(seq) for seq in sequences) for feature, sequences in
                            encoded_seqs_dict.items()}
        self.feature_projection = nn.Linear(self.max_lengths[self.target_feature] - 1, self.gpt2.config.n_embd).to(
            self.device)

        dataset = self.prepare_dataset(encoded_seqs_dict)
        validation_dataset = self.prepare_dataset(validation_encoded_seqs)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0

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

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = self.evaluate(validation_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0  # Reset patience
                # torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    self.final_epoch_loss = best_val_loss
                    break

            if epoch == epochs - 1:
                self.final_epoch_loss = best_val_loss

    def evaluate(self, data_loader):
        try:
            self.eval()
            total_loss = 0
            with torch.no_grad():
                for inputs, targets in data_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    print(inputs.shape)
                    outputs = self.forward(inputs)
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()

            return total_loss / len(data_loader)
        except:
            return 0

    def prepare_dataset(self, encoded_seqs_dict):
        feature_tensors = []
        target_values = []

        for feature, sequences in encoded_seqs_dict.items():
            max_length = self.max_lengths[feature] - 1
            feature_tensor = [torch.tensor(sequence[:-1], dtype=torch.long) for sequence in sequences]
            padded_feature_tensor = self.pad_sequence(feature_tensor, batch_first=True, padding_value=0,
                                                      max_length=max_length)
            feature_tensors.append(padded_feature_tensor)
            if feature == self.target_feature:
                target_values = [sequence[-1] for sequence in sequences]

        input_tensor = torch.stack(feature_tensors, dim=1)
        target_tensor = torch.tensor(target_values, dtype=torch.long)
        return TensorDataset(input_tensor, target_tensor)

    def pad_sequence(self, sequences, batch_first, padding_value, max_length):
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq if len(seq) >= max_length else torch.cat(
                [seq, torch.full((max_length - len(seq),), padding_value, dtype=seq.dtype)])
            padded_sequences.append(padded_seq)
        if batch_first:
            return torch.stack(padded_sequences, dim=0)
        else:
            return torch.stack(padded_sequences, dim=1)

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
        prepared_features = [torch.tensor(feature, dtype=torch.long).to(self.device) for feature in features]
        max_length = self.max_lengths[self.target_feature] - 1
        padded_feature_tensors = [
            self.pad_sequence(
                [feat],
                batch_first=True,
                padding_value=0,
                max_length=max_length
            ).to(self.device) for feat in prepared_features]
        input_tensor = torch.stack(padded_feature_tensors, dim=1)
        return input_tensor

    def extract_dims(self, vocabs):
        feature_dims = []
        for feature, vocab in vocabs.items():
            vocab_size = len(vocab)
            feature_dims.append(vocab_size + 1)
        return feature_dims
