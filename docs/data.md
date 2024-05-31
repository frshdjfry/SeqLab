# Data

## Overview

The Data section provides comprehensive guidelines on how to prepare, format, and manage your data for use with SeqLab. Proper data preparation is crucial for training robust machine learning models.

## Data Formats

SeqLab accepts data in two primary formats: TXT and CSV. Each format has specific use cases and requirements.

### TXT Format

- **Usage**: Ideal for single feature/dimension data.
- **Format**: Each sequence should be represented in a row, with space-separated values.
- **Example**:
    ```
    A B C D  <--- events of sequence 1
    B C D C  <--- events of sequence 2
    ```

### CSV Format

- **Usage**: Suitable for multi-feature/dimension data.
- **Format**: Features are tab-separated, with sequences separated by rows containing the `>*` symbol. The first line should contain feature names, with each subsequent row representing an event in time. The rows between the separator rows (`>*`) represent sequences.
- **Example**:
    ```
    feature1    feature2    feature3
       A           2           X          <---- Event 1 of sequence 1
       B           5           Y          <---- Event 2 of sequence 1
       >*          >*          >*         <---- Sequence separator
       B           8           Y          <---- Event 1 of sequence 2
       A           11          Z          <---- Event 2 of sequence 2
    ```

## Preparing Your Data

### Steps to Prepare Data

1. **Data Collection**: Gather the data relevant to your sequence modeling tasks.
2. **Data Cleaning**: Remove any inconsistencies or missing values (specially about the `target_feature`) to ensure the data quality.
3. **Formatting**: Organize your data into the accepted formats (TXT or CSV) as described above.
4. **Validation**: Verify that your data adheres to the specified formats and contains no errors.

### Storing Your Data

- Place your prepared data files in a designated folder within your project directory (e.g., `data` folder).
- Ensure that the paths to these data files are correctly specified in your experiment configuration file (`config.json`).

## Example Configuration

Below is an example of how to specify your data paths and feature dimension and target feature (in case of CSV format) in the configuration file:

```json
{
  "experiments": [
    {
      "feature_dimensions": "one_to_one",
      "datasets": [
        "data/some-dataset.csv",
        "data/some-dataset.txt"
      ],
      
  ... Rest of the config
}
```
For multi-feature (many-to-one) data, the configuration is slightly different. You need to specify the source_features which are the input features used for prediction, in addition to the target_feature:

```json
{
  "experiments": [
    {
      "feature_dimensions": "many_to_one",
      "source_features": [
        "some-feature-name",
        "some-feature-name-2"
      ],
      "target_feature": "some-feature-name",
      "datasets": [
        "data/some-dataset.csv"
      ],
      
  ... Rest of the config
}
```