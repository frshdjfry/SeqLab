### Variable Order Markov Model

#### Overview

The Variable Order Markov Model extends the first-order Markov Model by considering varying lengths of state sequences (contexts) up to a predefined maximum. This approach enables more fine-grained predictions based on the different contexts encountered leading up to a given state in the training data.

#### Theory

The transition probabilities for moving from context \( C \) (a sequence of states) to state \( j \) after state \( i \) are calculated as:

\[ P_{ij|C} = \frac{N_{ij|C} + \alpha}{\sum_{k} (N_{ik|C} + \alpha)} \]

where:
- \( N_{ij|C} \) represents the frequency of observing state \( j \) after context \( C \) and state \( i \).
- \( \alpha \) is a smoothing parameter to handle unseen transitions.

#### Data Encoding and Training

1. **Data Encoding**: The states (or categories) are encoded based on the provided vocabulary. Each sequence in the training data is a series of these encoded states.

2. **Transition Matrix Construction**: During training, the model constructs a transition matrix by counting the occurrences of each transition between states in various contexts up to the specified maximum context length.

3. **Smoothing**: The model applies the smoothing parameter \( \alpha \) to ensure that every possible transition has a nonzero probability. This is done by adding \( \alpha \) to each transition count.
   - After smoothing, the transition probabilities are recalculated by dividing the adjusted transition counts by the total adjusted transitions for each context. This produces valid probabilities, ensuring the model remains robust.

4. **Normalization**: The total count of transitions from a given context is increased by \( \alpha \) times the number of possible transitions. This ensures that the sum of probabilities from any given context to all possible next states equals 1.

#### Prediction

For a given sequence, the model predicts the next state by:
1. Identifying the current context (a sequence of states leading up to the last state in the sequence).
2. Using the transition matrix to determine the probabilities of transitioning to each possible next state from this context.
3. Selecting the next state based on these probabilities, using random sampling weighted by the transition probabilities. This means that the next state is chosen randomly, but transitions with higher probabilities are more likely to be selected. This approach ensures that the model can account for the inherent uncertainty and variability in sequential data.

#### Model Persistence

The Variable Order Markov Model in SeqLab supports saving and loading to facilitate persistence and reuse:
- **Saving**: The model's state, including the transition matrix and configuration, is saved to a file.
- **Loading**: A saved model can be loaded back, restoring the transition matrix and configuration for further use.

#### Example Configuration

Here is an example configuration for using the Variable Order Markov Model in SeqLab:

```json
{
  "name": "VariableOrderMarkovModel",
  "epochs": 1,
  "optimization_config": {
    "alpha": {"min": 1e-5, "max": 1.0},
    "max_context_length": {"min": 2, "max": 5}
  }
}
```