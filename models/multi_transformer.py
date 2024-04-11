import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class MultiTransformerModel(nn.Module):
    def __init__(self, vocab, target_feature, embedding_dim, nhead=8, num_layers=3, dim_feedforward=512,
                 lr=0.001, **kwargs):
        super(MultiTransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.target_feature = target_feature
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        feature_dims = self.extract_dims(vocab)

        self.embeddings = nn.ModuleList([
            nn.Embedding(feature_dim, embedding_dim).to(self.device) for feature_dim in feature_dims
        ])

        self.transformer = nn.Transformer(
            d_model=embedding_dim * len(vocab),
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        ).to(self.device)

        self.num_classes = len(vocab[self.target_feature]) + 1
        self.fc = nn.Linear(embedding_dim * len(vocab), self.num_classes).to(self.device)
        self.final_epoch_loss = 0

    def train_model(self, encoded_seqs_dict, validation_encoded_seqs, epochs=10, batch_size=64, patience=10, **kwargs):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

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
        self.eval()
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

    def forward(self, inputs):
        embedded_features = []
        for i, embedding in enumerate(self.embeddings):
            embedded = embedding(inputs[:, i])
            embedded_features.append(embedded)
        concatenated_embeddings = torch.cat(embedded_features, dim=-1)
        tgt_dummy = torch.zeros_like(concatenated_embeddings)
        transformed = self.transformer(concatenated_embeddings, tgt_dummy)
        last_timestep_output = transformed[:, -1, :]
        output = self.fc(last_timestep_output)
        return output

    def predict(self, features):
        self.eval()
        with torch.no_grad():
            prepared_features = [torch.tensor(feature, dtype=torch.long).to(self.device) for feature in features]
            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in
                                      prepared_features]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)  # Shape: [1, num_features, seq_len]
            outputs = self.forward(input_tensor)
            _, predicted = torch.max(outputs[:, -1, :], dim=1)
            return predicted.item()

    def predict_with_probabilities(self, features):
        self.eval()
        with torch.no_grad():
            prepared_features = [torch.tensor(feature, dtype=torch.long).to(self.device) for feature in features]
            padded_feature_tensors = [pad_sequence([feat_tensor], batch_first=True) for feat_tensor in
                                      prepared_features]
            input_tensor = torch.stack(padded_feature_tensors, dim=1)  # Shape: [1, num_features, seq_len]
            outputs = self.forward(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.squeeze().tolist()

    def extract_dims(self, vocabs):
        feature_dims = []
        for feature, vocab in vocabs.items():
            vocab_size = len(vocab)
            feature_dims.append(vocab_size + 1)
        return feature_dims
