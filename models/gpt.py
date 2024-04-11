import datetime
import os
from torch import optim
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, GPT2Config, GPT2LMHeadModel, TrainerCallback

from models.model_interface import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
if torch.backends.mps.is_built():
    device = torch.device("mps")


class GPTModel(BaseModel):
    def __init__(self, vocab, lr=0.001, num_layers=12, nhead=12, dim_feedforward=3072, **kwargs):
        vocab_size = len(vocab) + 1
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

    def train_model(self, encoded_seqs, epochs=3, batch_size=16, **kwargs):
        dataset = GPTChordDataset(encoded_seqs, self.config.vocab_size)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"./gpt_chords/gpt_chords_{now}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[CaptureFinalLossCallback()]
        )
        trainer.train()
        self.final_epoch_loss = self.model.internal_final_epoch_loss

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
