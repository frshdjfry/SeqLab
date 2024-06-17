## Evaluation

Evaluating the performance of machine learning models is crucial to understanding their effectiveness and reliability. In SeqLab, we employ multiple metrics to provide a comprehensive assessment of model performance. These metrics include accuracy, perplexity, and semantic similarity, each offering insights into different aspects of the models' predictive capabilities.

### Accuracy

Our evaluation framework centers around assessing the predictive accuracy of the model on unseen test data. Accuracy measures the performance in correctly predicting the subsequent state in a sequence. It is defined as the proportion of correctly predicted states to the total number of predictions made. While the ground truth typically holds a singular "correct" value, in many contexts other values may substitute equally well.

### Perplexity

In addition to accuracy, we employ perplexity as a secondary metric to evaluate our model's performance. Perplexity measures the model's uncertainty in predicting the next state, offering insight into its probabilistic forecasting efficacy. Lower perplexity values indicate higher confidence in predictions and, consequently, better model performance. Mathematically, perplexity (\(P\)) is defined as the exponential of the average negative log-likelihood of the test set predictions:

$\ P = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log(p(x_i))\right) \$

where $\(N\)$ is the total number of predictions, and $\(p(x_i)\)$ is the probability assigned to the correct state $\(x_i\)$ by the model. Perplexity is particularly informative in the context of generative models, where the goal extends beyond mere classification to include the generation of probable sequences.

### Semantic (Word2Vec) Similarity

Beyond accuracy and perplexity, we utilize semantic similarity metrics to evaluate the closeness between predicted and actual states. This metric is relevant for understanding the model's ability to generate coherent predictions, even when not exactly matching the ground truth. The semantic similarity is computed based on the cosine similarity between the vector representations of predicted and actual states, as obtained from a pre-trained embedding model. The similarity score ranges from 0 to 1, where 1 indicates perfect alignment (or identical vectors) and values closer to 0 denote lower similarity. Formally, the similarity $(\(S\))$ between two states $\(s_1\)$ and $\(s_2\)$ is given by:

$\ S(s_1, s_2) = \max\left(1 - \cos(\vec{s_1}, \vec{s_2}), 0\right) \$

where $\(\vec{s_1}\)$ and $\(\vec{s_2}\)$ are the embeddings of states $\(s_1\)$ and $\(s_2\)$, respectively, and $\(\cos\)$ denotes the cosine distance between the two vectors. This metric is essential for cases where the exact prediction may not be critical, but the semantic or contextual closeness of the prediction to the ground truth holds significance.

### k-Fold Cross-Validation

For model evaluation, SeqLab adopts a k-fold cross-validation approach on the unseen test data, ensuring a robust assessment of each model's accuracy. This method partitions the test data into $\(k\)$ subsets, where each subset serves once as the test set while the remaining $\(k-1\)$ subsets form the training set. The model's accuracy is then averaged over $\(k\)$ runs, providing a comprehensive measure of its generalization capability. The `kfold_splits` configuration specifies the number of folds \(k\) used in cross-validation.
