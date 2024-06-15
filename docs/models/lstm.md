### LSTM Model

#### Overview

The LSTM (Long Short-Term Memory) model is a type of recurrent neural network (RNN) that is particularly well-suited to sequential data. It addresses the challenge of long-term dependencies in sequences by leveraging memory cells and gating mechanisms. These mechanisms optimize and control the retention and forgetting of information over long sequences.

#### Theory

LSTM models manage sequence data effectively by using parameters such as embedding dimension, hidden layer dimension, and the number of layers. These parameters are fine-tuned for optimal prediction accuracy. The core LSTM formula encapsulating its gating mechanics can be represented as follows:

$\ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \$

where:
- $\( f_t \)$ is the forget gate's activation.
- $\( \sigma \)$ is the sigmoid function.
- $\( W_f \)$ and $\( b_f \)$ are the weight and bias for the forget gate.
- $\( h_{t-1} \)$ is the previous hidden state.
- $\( x_t \)$ is the current input vector.

#### Data Encoding and Training

1. **Data Encoding**: The states (or categories) are encoded based on the provided vocabulary. Each sequence in the training data is a series of these encoded states.

2. **Model Architecture**: The LSTM model consists of the following components:
    - **Embedding Layer**: Converts input tokens into dense vectors of fixed size (embedding dimension).
    - **LSTM Layers**: Multiple LSTM layers process the embedded sequences, allowing the model to capture long-term dependencies.
    - **Fully Connected Layer**: Maps the output of the LSTM layers to the vocabulary size, producing logits for each token in the vocabulary.

3. **Training Process**:
    - The model is trained using cross-entropy loss and the Adam optimizer.
    - Early stopping is implemented to prevent overfitting. Training stops if the validation loss does not improve after a specified number of epochs (patience).
    - The best model is saved based on validation loss.

4. **Evaluation**: The model's performance is evaluated on a validation set after each epoch. The average validation loss is used to determine the best model.

#### Prediction

For a given sequence, the model predicts the next state by:
1. Encoding the sequence using the model's vocabulary.
2. Passing the encoded sequence through the embedding and LSTM layers.
3. Using the fully connected layer to produce logits for the next state.
4. Applying a softmax function to the logits to obtain probabilities for each possible next state.
5. Selecting the state with the highest probability as the predicted next state.

#### Model Persistence

The LSTM Model in SeqLab supports saving and loading to facilitate persistence and reuse:
- **Saving**: The model's state, including the learned parameters and configuration, is saved to a file.
- **Loading**: A saved model can be loaded back, restoring the parameters and configuration for further use.

#### Example Configuration

Here is an example configuration for using the LSTM Model in SeqLab:

```json
{
  "name": "LSTMModel",
  "epochs": 200,
  "optimization_config": {
    "lr": {"min": 1e-3, "max": 1e-2},
    "num_layers": {"min": 1, "max": 2},
    "hidden_dim": {"values": [512, 1024]},
    "embedding_dim": {"values": [128, 256, 512]},
    "batch_size": {"values": [32, 64]}
  }
}
