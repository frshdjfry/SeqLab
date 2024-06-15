### LSTM Model with Attention

#### Overview

The LSTM Model with Attention enhances the standard LSTM model by incorporating an attention mechanism. This addition allows the model to focus on relevant parts of the sequence, making it better suited for handling long sequences or data with complex dependencies. The attention mechanism enables the model to weigh different parts of the sequence differently, thus improving the prediction accuracy.

#### Theory

The LSTM Model with Attention calculates a context vector $\( c_t \)$ at each time step as a weighted sum of the LSTM's hidden states $\( h_i \)$, using computed attention scores $\( \alpha_t \)$:

$\ c_t = \sum_{i} \alpha_{ti} h_i \$

where:
- $\( \alpha_{ti} \)$ are the attention scores, computed as:

$\ \alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k} \exp(e_{tk})} \$

- $\( e_{ti} \)$ is the alignment score, which measures how well the inputs around position $\( t \)$ and $\( i \)$ match.

The model then combines this context vector with the LSTM's current output through a concatenation operation, followed by a linear transformation to produce the final output $\( y_t \)$. This method allows the model to dynamically focus on relevant portions of the input sequence.

#### Data Encoding and Training

1. **Data Encoding**: The states (or categories) are encoded based on the provided vocabulary. Each sequence in the training data is a series of these encoded states.

2. **Model Architecture**: The LSTM-with-Attention model consists of the following components:
    - **Embedding Layer**: Converts input tokens into dense vectors of fixed size (embedding dimension).
    - **LSTM Layers**: Multiple LSTM layers process the embedded sequences, allowing the model to capture long-term dependencies.
    - **Attention Mechanism**: Computes the attention scores and context vector.
    - **Fully Connected Layer**: Maps the context vector and the LSTM's current output to the vocabulary size, producing logits for each token in the vocabulary.

3. **Training Process**:
    - The model is trained using cross-entropy loss and the Adam optimizer.
    - Early stopping is implemented to prevent overfitting. Training stops if the validation loss does not improve after a specified number of epochs (patience).
    - The best model is saved based on validation loss.

4. **Evaluation**: The model's performance is evaluated on a validation set after each epoch. The average validation loss is used to determine the best model.

#### Prediction

For a given sequence, the model predicts the next state by:
1. Encoding the sequence using the model's vocabulary.
2. Passing the encoded sequence through the embedding and LSTM layers.
3. Computing the attention scores and context vector.
4. Combining the context vector with the LSTM's current output.
5. Using the fully connected layer to produce logits for the next state.
6. Applying a softmax function to the logits to obtain probabilities for each possible next state.
7. Selecting the state with the highest probability as the predicted next state.

#### Model Persistence

The LSTM with Attention Model in SeqLab supports saving and loading to facilitate persistence and reuse:
- **Saving**: The model's state, including the learned parameters and configuration, is saved to a file.
- **Loading**: A saved model can be loaded back, restoring the parameters and configuration for further use.

#### Example Configuration

Here is an example configuration for using the LSTM with Attention Model in SeqLab:

```json
{
  "name": "LSTMModelWithAttention",
  "epochs": 100,
  "optimization_config": {
    "lr": {"min": 1e-3, "max": 1e-2},
    "num_layers": {"min": 1, "max": 2},
    "hidden_dim": {"values": [512, 1024]},
    "embedding_dim": {"values": [128, 256, 512]},
    "batch_size": {"values": [32, 64]}
  }
}
```