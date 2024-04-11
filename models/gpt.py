import datetime
import os
from torch import optim, nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import Trainer, TrainingArguments, GPT2Config, GPT2LMHeadModel, TrainerCallback, EarlyStoppingCallback

from models.model_interface import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
if torch.backends.mps.is_built():
    device = torch.device("mps")


class GPTModel(BaseModel):
    def __init__(self, vocab, lr=0.001, num_layers=12, nhead=12, dim_feedforward=3072, **kwargs):
        vocab_size = len(vocab) + 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=num_layers,
            n_head=nhead,
            n_inner=dim_feedforward,
            pad_token_id=0
        )
        self.model = GPT2LMHeadModel(self.config).to(device)
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.internal_final_epoch_loss = 0
        self.final_epoch_loss = 0
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_model(self, encoded_seqs, validation_encoded_seqs, epochs=10, batch_size=64, patience=20, **kwargs):
        dataset = self.prepare_dataset(encoded_seqs)
        validation_dataset = self.prepare_dataset(validation_encoded_seqs)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs).logits
                outputs =outputs[:, -1, :]
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
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
                self.final_epoch_loss = avg_train_loss

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits[:, -1, :], targets)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def prepare_dataset(self, encoded_seqs):
        inputs, targets = [], []
        for seq in encoded_seqs:
            inputs.append(torch.tensor(seq[:-1], dtype=torch.long).to(self.model.device))
            targets.append(seq[-1])

        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.tensor(targets, dtype=torch.long).to(self.model.device)
        return TensorDataset(inputs_padded, targets)

    def predict(self, input_sequence, num_predictions=1):
        input_ids = torch.tensor([input_sequence], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=len(input_sequence) + num_predictions,
                                          num_return_sequences=1, pad_token_id=self.config.pad_token_id)

        return outputs.tolist()[0][-num_predictions:]

    def predict_with_probabilities(self, input_sequence):
        input_ids = torch.tensor([input_sequence], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            logits[:, :, self.config.pad_token_id] = -float('Inf')
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            return probabilities.squeeze().tolist()


class CaptureFinalLossCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        kwargs['model'].internal_final_epoch_loss = state.log_history[-2]['loss']


class GPTChordDataset(Dataset):
    def __init__(self, encoded_seqs, seq_length=50):
        self.inputs = []
        self.targets = []
        for seq in encoded_seqs:
            if len(seq) > seq_length:
                continue
            padded_seq = seq + [0] * (seq_length - len(seq))
            self.inputs.append(padded_seq[:-1])
            self.targets.append(padded_seq[1:])
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.final_epoch_loss = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.targets[idx]}
