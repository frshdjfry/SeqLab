## Monitoring
MLflow is an integral tool in SeqLab for tracking experiments, managing and comparing model configurations, and evaluating performance. It provides a detailed and interactive interface for exploring experiment data.

### Launching the MLflow Dashboard

To begin monitoring experiments with MLflow, launch the MLflow UI:
```
mlflow ui --port=4000
```

This command initializes the MLflow dashboard, which can be accessed by navigating to `127.0.0.1:4000` in your web browser.

### Understanding the MLflow Dashboard Structure

MLflow organizes data in a structured and hierarchical manner that mirrors the organization of your experiments. This structure is crucial for analyzing the details of your model's performance across different configurations and iterations. Here’s how the data is typically structured:

1. **Experiment Level**:
   - **Example**: `one_to_one experiment on 'dataset1.txt'`
   - **Description**: This level aggregates results from all models evaluated on a specific dataset with a certain configuration.

2. **Model Level**:
   - **Example**: `MarkovModel`
   - **Description**: This level shows results for a specific model type. Metrics at this level include aggregated statistics such as average accuracy, average perplexity (ppl), and average Word2Vec similarity (w2v) across all folds and trials.

3. **Fold Level**:
   - **Example**: `Fold 1`
   - **Description**: This level represents one of the k folds in cross-validation. Metrics at this level include the best accuracy, best perplexity, and best Word2Vec similarity achieved within this fold.

4. **Trial Level**:
   - **Example**: `Trial 0`
   - **Description**: This level shows results for a specific trial within a fold. Detailed metrics such as accuracy, perplexity, and Word2Vec similarity for this trial are displayed. Additionally, this level includes the hyperparameters chosen for this trial and tags with additional information such as the path to the saved model and other configurations.

### Visual Representation in MLflow

```
Experiment: one_to_one on 'dataset1.txt'
│
├── Model: MarkovModel
│ ├── Fold 1
│ │ ├── Trial 0
│ │ │ ├── Metrics: Accuracy, Perplexity, W2V Similarity
│ │ │ ├── Parameters: Chosen hyperparameters
│ │ │ └── Tags: saved_model_path, other configs
│ │ ├── Trial 1
│ │ └── ...
│ ├── Fold 2
│ └── Fold 3
│
├── Model: LSTMModel
│ ├── Fold 1
│ │ ├── Trial 0
│ │ ├── ...
│ └── ...
│
└── ... (Additional models or experiments as applicable)
```

![MLflow Experiment Tracking](https://github.com/frshdjfry/SeqLab/blob/master/docs/images/mlflow_video.gif)
*Figure: Visualizing experiment tracking with MLflow in SeqLab. Each experiment set is named after its dimensionality and contains multiple models. Each model is evaluated using different folds of data, with multiple trials per fold to optimize hyperparameters. The MLflow UI stores metrics, evaluation results, and important experiment tags for each run, allowing detailed analysis and comparison of model performance.*
