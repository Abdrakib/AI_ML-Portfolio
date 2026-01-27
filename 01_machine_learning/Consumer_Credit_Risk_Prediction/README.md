# Consumer Credit Risk Prediction

Predicts credit risk for consumers using machine learning algorithms to identify high-risk (likely to default) vs low-risk (good credit) customers.

## Description

This project implements a comprehensive credit risk prediction system using multiple machine learning algorithms. Given consumer financial and personal data, the model predicts whether a customer is high risk (likely to default) or low risk (good credit). This is a critical binary classification problem in the financial industry, helping lenders make informed decisions about loan approvals.

## Problem Definition

Given consumer financial and personal data, can we predict whether a customer is high risk (likely to default) or low risk (good credit)?

## Pipeline

### 1. Load Data
- Load credit risk prediction dataset from CSV file
- Dataset typically contains 300K+ records with 100+ features
- Features include: income, age, employment history, credit history, loan details, etc.

### 2. Data Preprocessing
- **Target Identification**: Automatically detect target column (TARGET, target, Risk, etc.)
- **Missing Value Handling**:
  - Fill numerical missing values with median
  - Fill categorical missing values with mode
- **Categorical Encoding**: One-hot encode categorical variables using pandas get_dummies
- **Feature Preparation**: Handle remaining NaN values

### 3. Data Exploration
- Visualize target distribution (highly imbalanced: ~8% high risk, ~92% low risk)
- Analyze class imbalance
- Understand data characteristics

### 4. Prepare Data for Modeling
- **Train-Test Split**: 80-20 split with stratification to maintain class distribution
- **Feature Scaling**: StandardScaler for numerical features
- **Handle Imbalanced Data**:
  - **Option 1**: Use `class_weight='balanced'` for Logistic Regression and Random Forest
  - **Option 2**: Use `sample_weight` with `compute_sample_weight` for Gradient Boosting
  - **Option 3**: Apply SMOTE (Synthetic Minority Oversampling Technique) for data balancing

### 5. Model Training

#### Baseline Models
- **Logistic Regression**: Linear model with class_weight='balanced'
- **Random Forest**: Ensemble method with class_weight='balanced'
- **Gradient Boosting**: Gradient descent-based boosting with sample_weight

### 6. Hyperparameter Tuning

#### Random Forest Tuning
- **Method**: RandomizedSearchCV with F1-score as primary metric
- **Parameters Tuned**:
  - n_estimators: [200, 400, 600]
  - max_depth: [10, 20, 30, None]
  - min_samples_split: [2, 5, 10, 20]
  - min_samples_leaf: [1, 2, 5, 10]
  - max_features: ["sqrt", "log2", None]
  - max_samples: [0.6, 0.8, 1.0]
  - class_weight: ["balanced"]
- **Cross-Validation**: 3-fold CV
- **Iterations**: 15 random combinations

#### Gradient Boosting Tuning
- **Method**: RandomizedSearchCV with F1-score as primary metric
- **Parameters Tuned**:
  - n_estimators: [100, 200, 400, 600]
  - learning_rate: [0.005, 0.01, 0.05, 0.1, 0.2]
  - max_depth: [1, 2, 3, 4, 5]
  - subsample: [0.6, 0.8, 1.0]
  - min_samples_split: [2, 5, 10, 20, 50]
  - min_samples_leaf: [1, 2, 5, 10, 20]
  - max_features: [None, "sqrt", "log2"]
- **Cross-Validation**: 5-fold CV
- **Iterations**: 20 random combinations
- **Note**: Uses sample_weight for class imbalance (no class_weight parameter)

#### Logistic Regression Tuning
- **Method**: GridSearchCV with F1-score as primary metric
- **Parameters Tuned**:
  - C: [0.001, 0.01, 0.1, 1, 10, 50]
  - penalty: ["l1", "l2"]
  - solver: ["liblinear", "saga"]
- **Cross-Validation**: 5-fold CV
- **Note**: Uses class_weight='balanced' for class imbalance

### 7. Threshold Tuning ⭐

Critical step for imbalanced classification:
- Test thresholds from 0.05 to 0.95 (37 points)
- Optimize for F1-score (can be changed to Recall if catching all high-risk cases is priority)
- Find optimal threshold for each model
- Compare performance at default (0.5) vs optimized threshold

### 8. Model Evaluation

Comprehensive evaluation using multiple metrics:
- **Accuracy**: Overall correctness
- **Precision**: Minimize false positives (low-risk customers flagged as high-risk)
- **Recall**: Maximize true positives (catch as many high-risk cases as possible)
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall model performance on imbalanced data

### 9. Model Comparison

- Compare baseline vs tuned models
- Visualize metrics across all models
- Confusion matrices for tuned models
- Feature importance analysis (from Random Forest)

## Models

### Primary Models
1. **Logistic Regression**
   - Baseline and tuned versions
   - Uses class_weight='balanced' for imbalance
   - L1/L2 regularization with grid search

2. **Random Forest**
   - Baseline and tuned versions
   - Uses class_weight='balanced' for imbalance
   - Extensive hyperparameter search

3. **Gradient Boosting**
   - Baseline and tuned versions
   - Uses sample_weight for imbalance (no class_weight parameter)
   - Gradient descent-based sequential tree building

## Evaluation Metrics

Since credit risk prediction is an imbalanced classification problem, the focus is on:

- **Primary Metric**: **ROC-AUC** - Overall model performance on imbalanced data
- **Secondary Metrics**:
  - **Precision**: Minimize false positives (low-risk customers flagged as high-risk)
  - **Recall**: Maximize true positives (catch as many high-risk cases as possible)
  - **F1-Score**: Balance between precision and recall
  - **Accuracy**: Overall correctness (less important for imbalanced data)

## Key Features

### Data Characteristics
- **Dataset Size**: 300K+ records, 100+ features
- **Class Distribution**: Highly imbalanced (~8% high risk, ~92% low risk)
- **Feature Types**: Numerical and categorical
- **Missing Values**: Handled with median (numerical) and mode (categorical)

### Feature Categories
- **Demographics**: Age, gender, family status
- **Financial**: Income, credit amount, annuity, credit bureau data
- **Employment**: Employment type, organization type
- **Loan Details**: Contract type, loan purpose, payment behavior
- **Documentation**: Various document flags

## Handling Class Imbalance

The project uses multiple strategies to handle the highly imbalanced dataset:

1. **Class Weighting**:
   - `class_weight='balanced'` for Logistic Regression and Random Forest
   - Automatically adjusts weights inversely proportional to class frequencies

2. **Sample Weighting**:
   - `compute_sample_weight(class_weight='balanced')` for Gradient Boosting
   - Required because Gradient Boosting doesn't support class_weight parameter

3. **Threshold Tuning**:
   - Optimize classification threshold (default 0.5 may not be optimal)
   - Test thresholds from 0.05 to 0.95
   - Maximize F1-score or Recall based on business needs

4. **Alternative: SMOTE** (Synthetic Minority Oversampling Technique)
   - Can be used instead of class weighting
   - Synthetically generates minority class samples
   - Note: Memory-intensive for large datasets

## Hyperparameter Tuning Strategy

- **RandomizedSearchCV**: Used for Random Forest and Gradient Boosting (faster, explores more parameter space)
- **GridSearchCV**: Used for Logistic Regression (smaller parameter space)
- **Scoring Metric**: F1-score (can be changed to Recall if catching all high-risk cases is priority)
- **Cross-Validation**: 3-5 folds depending on model
- **Iterations**: 15-20 random combinations for RandomizedSearchCV

## Threshold Optimization

After hyperparameter tuning, optimal classification thresholds are found:
- Test 37 threshold values from 0.05 to 0.95
- Optimize for F1-score (or Recall if needed)
- Compare performance at default (0.5) vs optimized threshold
- Each model gets its own optimal threshold

## Output

The project generates:
1. **Model Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC for all models
2. **Comparison Visualizations**: Bar charts comparing all metrics across models
3. **Confusion Matrices**: Heatmaps for tuned models
4. **Feature Importance**: Top 15 most important features from Random Forest
5. **Best Model Selection**: Model with highest ROC-AUC score

## Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

Optional (for SMOTE):
```txt
imbalanced-learn>=0.8.0
```

## Applications

- **Loan Approval**: Automated credit risk assessment for loan applications
- **Financial Institutions**: Risk management and portfolio optimization
- **Credit Scoring**: Generate credit scores for customers
- **Fraud Detection**: Identify potentially fraudulent applications
- **Regulatory Compliance**: Meet regulatory requirements for risk assessment

## Key Insights

1. **Class Imbalance is Critical**: The dataset is highly imbalanced (~8% high risk), requiring special handling
2. **Threshold Tuning Matters**: Default threshold (0.5) may not be optimal; tuning significantly improves F1-score
3. **Hyperparameter Tuning Improves Performance**: Tuned models consistently outperform baseline models
4. **Multiple Metrics are Important**: ROC-AUC alone isn't enough; Precision and Recall provide business context
5. **Feature Engineering**: One-hot encoding expands feature space significantly (245 features from 122 original columns)
6. **Model Selection**: Best model varies based on business priorities (ROC-AUC vs Recall vs Precision)

## Usage

1. Load your credit risk dataset (update path in the notebook)
2. Run all cells sequentially
3. The notebook will:
   - Preprocess the data
   - Train baseline models
   - Perform hyperparameter tuning
   - Optimize classification thresholds
   - Compare all models
   - Display feature importance
   - Select best model

## Notes

- **Memory Considerations**: For large datasets, use class_weight/sample_weight instead of SMOTE to save memory
- **Scoring Metric**: F1-score is used by default, but can be changed to Recall if catching all high-risk cases is priority
- **Threshold Selection**: Business requirements should guide whether to optimize for F1-score or Recall
- **Feature Selection**: Consider feature selection for very high-dimensional datasets to reduce overfitting
- **Model Interpretability**: Random Forest provides feature importance, useful for understanding model decisions

## Model Performance Summary

After hyperparameter and threshold tuning:
- **Best Model**: Selected based on highest ROC-AUC score
- **Performance Improvement**: Tuned models show significant improvement over baseline
- **Balance**: Models achieve good balance between precision and recall
- **Practical Use**: Models can effectively predict credit risk for real-world applications
