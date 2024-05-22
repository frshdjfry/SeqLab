# SeqLab

Welcome to **SeqLab**! This project provides a comprehensive framework for training and evaluating various machine learning models, focusing on multi-feature sequential categorical data.

## Introduction

**SeqLab** facilitates the experimentation and benchmarking of machine learning models with ease. Using a configuration-driven approach, users can specify their experiments in a JSON file, ensuring reproducibility and flexibility. The project integrates seamlessly with MLflow for robust experiment tracking and model management.

### Key Features

- **Multiple Model Support**: Includes **Markov**, **LSTM with Attention**, **Transformer**, and **GPT** models.
- **Multi-feature Sequential Categorical Data Handling**
- **Automated Hyperparameter Optimization with Optuna**
- **Experiment Tracking with MLflow**

![MLflow Experiment Tracking](doc/images/mlflow_video.gif)
*Figure: Visualizing experiment tracking with MLflow in SeqLab. Each experiment set is named after its dimensionality and contains multiple models. Each model is evaluated using different folds of data, with multiple trials per fold to optimize hyperparameters. The MLflow UI stores metrics, evaluation results, and important experiment tags for each run, allowing detailed analysis and comparison of model performance.*
