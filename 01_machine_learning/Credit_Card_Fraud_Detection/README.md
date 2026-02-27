# Credit Card Fraud Detection

Machine learning project to detect fraudulent credit card transactions using various classification algorithms.

## Description

This project uses machine learning to identify fraudulent credit card transactions from legitimate ones. The dataset contains anonymized credit card transactions with PCA-transformed features (V1-V28), along with Time and Amount features. The target variable indicates whether a transaction is fraudulent (1) or legitimate (0).

## Problem

Credit card fraud detection is a critical binary classification problem where:
- The dataset is highly imbalanced (fraud cases are rare, typically <1%)
- False negatives (missing fraud) are costly
- False positives (flagging legitimate transactions) impact customer experience
- Need to balance precision and recall

## Dataset

The project uses a credit card fraud detection dataset with the following structure:
- **Features**: Time, V1-V28 (PCA-transformed features), Amount
- **Target**: Class (0 = Legitimate, 1 = Fraud)
- **Dataset**: Highly imbalanced with fraud cases being rare

**Note**: Update the CSV file path in the notebook if your dataset is in a different location.

## Models Used

1. **Logistic Regression** - Linear baseline model with class weights
2. **Random Forest Classifier** - Ensemble method with class balancing
3. **Gradient Boosting Classifier** - Boosting algorithm for improved performance

All models are evaluated with and without hyperparameter tuning.

## Features

- Data exploration and visualization
- Handling imbalanced dataset with class weights
- Feature scaling for optimal performance
- Multiple model comparison
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
- Confusion matrix visualization
- Feature importance analysis
- Model comparison charts

## Evaluation Metrics

For fraud detection, we focus on:
- **Precision**: Minimize false positives (legitimate transactions flagged as fraud)
- **Recall**: Maximize true positives (catch as many fraud cases as possible)
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall model performance on imbalanced data

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the credit card dataset CSV file (`creditcard.csv`)

3. Update the file path in the notebook if needed (Cell 3)

## Usage

1. Open the Jupyter notebook: `Credit Card Fraud Detection Project.ipynb`

2. Run all cells sequentially:
   - The notebook will load the data
   - Perform exploratory data analysis
   - Train multiple models
   - Perform hyperparameter tuning
   - Compare models and display results

3. Review the results:
   - Model comparison table
   - Confusion matrices
   - Feature importance plots
   - Final summary with best model

## Project Structure

```
Credit Card Fraud Detection/
├── Credit Card Fraud Detection Project.ipynb  # Main notebook
├── creditcard.csv                              # Dataset (you need to provide this)
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## Results

The project compares baseline and tuned versions of three models:
- Logistic Regression (baseline and tuned)
- Random Forest (baseline and tuned)
- Gradient Boosting (baseline and tuned)

The best model is selected based on ROC-AUC score, which is the most appropriate metric for imbalanced classification problems.

## Key Findings

1. Hyperparameter tuning improves model performance
2. Random Forest and Gradient Boosting typically perform well on this task
3. Feature scaling is important for optimal performance
4. Class weights help handle the imbalanced dataset

## Technologies Used

- Python 3.x
- Pandas - Data manipulation
- NumPy - Numerical computing
- Matplotlib - Data visualization
- Seaborn - Statistical visualization
- Scikit-learn - Machine learning algorithms and utilities

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Notes

- The dataset is highly imbalanced, so accuracy alone is not a good metric
- ROC-AUC is the primary metric for model selection
- All features are scaled before training
- Hyperparameter tuning may take some time depending on your system

## Author

Credit Card Fraud Detection Project

## License

This project is for educational purposes.
