# gpt.py
from torch import optim
from transformers import Trainer, TrainingArguments, GPT2Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset
from models.model_interface import BaseModel


class GPTChordDataset(Dataset):
    def __init__(self, encoded_seqs, vocab_size, seq_length=50):
        self.inputs = []
        self.targets = []
        for seq in encoded_seqs:
            if len(seq) > seq_length:
                continue  # Skip sequences that are longer than our desired sequence length for simplicity
            padded_seq = seq + [0] * (seq_length - len(seq))  # Pad sequences to ensure consistent length
            self.inputs.append(padded_seq[:-1])
            self.targets.append(padded_seq[1:])
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.targets[idx]}

class GPTModel(BaseModel):
    def __init__(self, vocab_size, lr=0.001, num_layers=12, nhead=12, dim_feedforward=3072):
        super(GPTModel, self).__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=num_layers,
            n_head=nhead,
            n_inner=dim_feedforward,
            pad_token_id=0
        )
        self.model = GPT2LMHeadModel(self.config)
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
        if torch.backends.mps.is_built():
            self.device = torch.device("mps")  # for mac use


    def train_model(self, encoded_seqs, epochs=3, batch_size=16):
        dataset = GPTChordDataset(encoded_seqs, self.config.vocab_size)
        training_args = TrainingArguments(
            output_dir="./gpt_chords",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        trainer.save_model("./gpt_chords")

    def predict(self, input_sequence, num_predictions=1):
        input_ids = torch.tensor([input_sequence], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=len(input_sequence) + num_predictions,
                                          num_return_sequences=1).to(self.device)
        return [self.tokenizer.decode(output_id) for output_id in outputs[0][-num_predictions:]]

    def predict_with_probabilities(self, input_sequence):
        input_ids = torch.tensor([input_sequence], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            return probabilities.squeeze().tolist()
