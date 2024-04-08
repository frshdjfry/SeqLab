import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
import datetime


# Define a dataset class compatible with multiple features
class MultiGPTDataset(Dataset):
    def __init__(self, encoded_seqs_dict, seq_length=50):
        self.inputs = []
        self.targets = []

        # Process each feature
        for feature, sequences in encoded_seqs_dict.items():
            for seq in sequences:
                if len(seq) > seq_length:  # Ensure sequence length consistency
                    continue
                padded_seq = seq + [0] * (seq_length - len(seq))  # Padding
                self.inputs.append(padded_seq[:-1])
                self.targets.append(padded_seq[1:])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.targets[idx]}


# MultiGPTModel definition
class MultiGPTModel(nn.Module):
    def __init__(self, feature_vocabs, target_feature,  embedding_dim, lr=0.001, num_layers=12, nhead=12, dim_feedforward=3072):
        super(MultiGPTModel, self).__init__()
        self.feature_vocabs = feature_vocabs
        self.embedding_dim = embedding_dim
        self.target_feature = target_feature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_built():
            self.device = torch.device("mps")  # for mac use

        # Embedding layers for each feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(vocab) + 1, embedding_dim).to(self.device) for _, vocab in feature_vocabs.items()
        ])

        # Configure GPT-2 model
        self.config = GPT2Config(
            vocab_size=sum(len(vocab) for _, vocab in feature_vocabs.items()) + 1,  # Combined vocab size
            n_layer=num_layers,
            n_head=nhead,
            n_inner=dim_feedforward,
            pad_token_id=0
        )
        self.model = GPT2LMHeadModel(self.config).to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.final_epoch_loss = 0.0
        self.projection_layers = nn.ModuleList([
            nn.Linear(embedding_dim, 768).to(self.device) for _ in feature_vocabs.items()
        ])
    def train_model(self, encoded_seqs_dict, epochs=3, batch_size=16):
        dataset = MultiGPTDataset(encoded_seqs_dict, seq_length=max(
            len(seq) for _, sequences in encoded_seqs_dict.items() for seq in sequences))
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"./multi_gpt/multi_gpt_{now}"

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
            train_dataset=dataset
        )

        trainer.train()


    def forward(self, input_ids):
        # Embedding and concatenating features
        embedded_features = [embedding(input_ids[:, i]) for i, embedding in enumerate(self.embeddings)]

        projected_embeddings = [projection_layer(embedded) for embedded, projection_layer in zip(embedded_features, self.projection_layers)]
        summed_embeddings = sum(projected_embeddings)

        # Ensure the summed embeddings have the correct shape: [batch_size, sequence_length, 768]
        assert summed_embeddings.size(-1) == 768, "Summed embeddings size mismatch with GPT-2 expected input"

        # print(concatenated_embeddings.shape)
        # GPT-2 forward pass

        # GPT-2 forward pass
        outputs = self.model(inputs_embeds=summed_embeddings)
        return outputs.logits

        # outputs = self.model(inputs_embeds=concatenated_embeddings)
        # return outputs.logits

    def predict_with_probabilities(self, features):
        # Prepare input features
        input_ids = self.prepare_features(features)

        # Ensure input is on the correct device
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            logits = self.forward(input_ids)

            # Assuming the target feature's logits are in a specific position or have a specific pattern in the output logits
            target_logits = self.extract_target_logits(logits, self.target_feature)

            # Convert logits to probabilities
            probabilities = torch.softmax(target_logits, dim=-1)
            return probabilities.squeeze().tolist()[0]

    def extract_dims(self, vocabs):
        feature_dims = []

        for feature, vocab in vocabs.items():
            vocab_size = len(vocab)  # Get the size of the vocabulary
            feature_dims.append(vocab_size + 1)  # Add 1 to account for the 0 padding index

        return feature_dims

    def prepare_features(self, features):
        # features is expected to be a list of lists, where each inner list is a sequence of events
        # Each event is a list containing values for each feature
        # Initialize a list to hold the encoded features
        encoded_features = []
        # Iterate over each feature in the event sequence

        # Iterate over the index and values of each feature in the event sequence
        for feature_idx, feature_values in enumerate(zip(*features)):  # Transpose the feature matrix to iterate feature-wise
            if feature_idx == len(self.feature_vocabs):
                break
            vocab = self.feature_vocabs[
                list(self.feature_vocabs.keys())[feature_idx]
            ]  # Get the vocab for the current feature
            encoded_feature = [vocab.get(value, len(vocab)) for value in
                               feature_values]  # Encode feature values, use len(vocab) as OOV index
            encoded_features.append(encoded_feature)

        # Pad the sequences to ensure consistent length across all sequences
        max_length = max(len(seq) for seq in encoded_features)
        padded_features = [seq + [0] * (max_length - len(seq)) for seq in encoded_features]  # Padding with 0, assuming 0 is the pad_token_id

        # Convert to tensor and transpose to get the original shape back (batch_size, seq_length, num_features)
        input_ids = torch.tensor(padded_features, dtype=torch.long).t()

        return input_ids


    def extract_target_logits(self, logits, target_feature):
        # Assuming logits are structured in a way that each feature's predictions are in a block
        start_index = 0
        for feature, vocab in self.feature_vocabs.items():
            if feature == target_feature:
                # Extract the logits block for the target feature
                target_logits = logits[:, start_index:start_index + len(vocab)]
                return target_logits
            start_index += len(vocab)

        raise ValueError(f"Target feature {target_feature} not found in feature vocabularies.")
