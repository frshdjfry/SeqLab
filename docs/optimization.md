## Optimization

SeqLab utilizes [Optuna](https://github.com/optuna/optuna), an open-source hyperparameter optimization framework, to enhance the performance of various machine learning models designed for sequence prediction. Optuna systematically searches for the optimal model configurations by adjusting hyperparameters through a series of trials.

### Hyperparameter Optimization with Optuna

Optuna automates the hyperparameter tuning process, aiming to find the best set of hyperparameters that maximize the model's performance. Each model in SeqLab undergoes multiple trials where Optuna adjusts hyperparameters such as the number of layers, embedding dimensions, and learning rates. The primary objective of these trials is to maximize prediction accuracy.

Key aspects of the optimization approach include:

1. **Hyperparameters**: Each model has specific hyperparameters that are optimized. For instance:
    - **LSTM and LSTM with Attention Models**: Number of layers, embedding dimensions, hidden dimensions, and learning rates.
    - **Markov and Variable Order Markov Models**: Smoothing parameters and context lengths.

2. **Number of Trials**: The `number_of_trials` configuration specifies how many different hyperparameter sets Optuna should evaluate. More trials can lead to a better-optimized model but will also require more computational resources.

3. **Objective Function**: The objective function used in Optuna evaluates the performance of each hyperparameter set based on prediction **accuracy**. This function guides the optimization process towards the best performing model configurations.
