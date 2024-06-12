# Models

## Introduction

This section provides detailed documentation for the machine learning models available in SeqLab. Each model has a dedicated page that includes its mathematical representation, data encoding methods, training details, hyperparameters, and example configurations. Additionally, we cover the model saving process and the patience mechanism used during training.

## List of Models

- [MarkovModel](docs/models/markovmodel.md)
- [VariableOrderMarkovModel](docs/models/variableordermarkovmodel.md)
- [LSTMModel](docs/models/lstmmodel.md)
- [LSTMModelWithAttention](docs/models/lstmmodelwithattention.md)
- [TransformerModel](docs/models/transformermodel.md)
- [GPTModel](docs/models/gptmodel.md)
- [MultiLSTMModel](docs/models/multilstmmodel.md)
- [MultiLSTMAttentionModel](docs/models/multilstmattentionmodel.md)
- [MultiTransformerModel](docs/models/multitransformermodel.md)
- [MultiGPTModel](docs/models/multigptmodel.md)

## Model Saving and Inference

SeqLab provides mechanisms for saving the best-performing trained models and loading them for future use. Models are automatically saved based on the experiments set up in the `config.json` file and are not designed to be trained in other ways. However, loading the models for inference is supported.

### Saving the Model

During the training process, SeqLab monitors model performance and saves the best model based on validation loss. The saved model includes:

- **Model State Dictionary**: The parameters of the model at the time of saving.
- **Vocabulary**: The vocabulary used during training, necessary for encoding and decoding sequences.
- **Model Configuration**: The configuration details, including hyperparameters such as embedding dimension, hidden dimension, number of layers, and learning rate.

Additionally, SeqLab saves the best models for **each trial** during hyperparameter tuning and cross-validation. It uses a **patience mechanism** to prevent overfitting during the training of deep models. The patience mechanism stops training if the model's performance on the validation set does not improve after a specified number of epochs.

The **storage locations** of these models are tracked and recorded using MLflow, allowing users to access them through the MLflow panel for further analysis if needed.

### Loading the Model

To load a saved model for inference, the state dictionary, vocabulary, and configuration are used to reinitialize the model. Here is an example of how to load a saved model and use it for making predictions:

```python
import torch
from models.lstm_attention import LSTMModelWithAttention
import os

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = LSTMModelWithAttention(vocab=checkpoint['vocab'], **checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Load the model
model = load_model(os.path.join('saved_models', 'best_model.pth'))

# Encode a sequence using the model's vocabulary
encoded_sequence = [model.vocab[chord] for chord in ['A:min', 'E:min', 'F:maj']]  # Example sequence
probabilities = model.predict_with_probabilities(encoded_sequence)

# Predict the next value
predicted_index = probabilities.index(max(probabilities))

# Decode the predicted next value
predicted_chord = [chord for chord, index in model.vocab.items() if index == predicted_index][0]

print(predicted_chord)
```



