# Customer Churn Prediction

Predicts which customers are likely to churn (leave the service) using machine learning algorithms.

## Description

Classification model to identify customers at risk of churning (leaving the service) based on their historical data and behavior patterns. This is a critical business problem where identifying churning customers early allows companies to take proactive retention measures.

## Problem Definition

Given customer data, can we predict whether a customer will churn (leave) or stay with the service?

## Pipeline

### 1. Load Data
- Load customer churn dataset from CSV file
- Dataset contains customer demographics, service usage, contract details, payment information, etc.
- Automatically detect target column (Churn, churn, Exited, exited, target, Target)

### 2. Explore (EDA)
- **Target Distribution**: Visualize churn distribution (bar charts and pie charts)
- **Dataset Statistics**: Basic statistical summary
- **Column Analysis**: Identify categorical and numerical columns
- **Class Imbalance**: Analyze the ratio of churning vs non-churning customers

### 3. Handle Missing Values
- **Identify Missing Values**: Count and visualize missing values per column
- **Fill Missing Values**:
  - Numerical columns: Fill with median
  - Categorical columns: Fill with mode
- Verify no missing values remain

### 4. Split Data and Encode Categorical Features

**Critical: Split data FIRST to avoid data leakage**

- **Train-Validation-Test Split**: 60-20-20 split with stratification
  - Training set: 60% (for model training)
  - Validation set: 20% (for hyperparameter tuning and threshold optimization)
  - Test set: 20% (for final evaluation - unseen data)
- **Categorical Encoding**: One-hot encode categorical variables using pandas get_dummies
  - Fit encoder on training data only
  - Transform validation and test sets using same encoding
  - Align columns across all sets (add missing columns with 0s)

### 5. Scale Numerical Features

**Fit scaler on training data only to avoid data leakage**

- **StandardScaler**: Standardize all features to have mean=0 and std=1
- Fit on training set, transform all sets (train, validation, test)
- Convert back to DataFrames for easier handling

### 6. Handle Imbalance

Address class imbalance using appropriate techniques:

- **Class Weight Calculation**: Compute balanced class weights
- **Class Weighting**: 
  - `class_weight='balanced'` for Logistic Regression and Random Forest
  - Automatically adjusts weights inversely proportional to class frequencies
- **Sample Weighting**:
  - `compute_sample_weight('balanced')` for XGBoost and LightGBM
  - Required because these models don't support class_weight parameter

### 7. Train Models

Train baseline models on training set and evaluate on validation set:

#### Models:
1. **Logistic Regression**
   - Uses `class_weight='balanced'` for imbalance
   - Linear model with regularization

2. **Random Forest**
   - Uses `class_weight='balanced'` for imbalance
   - Ensemble of decision trees (100 estimators)

3. **XGBoost**
   - Uses `sample_weight` for imbalance
   - Gradient boosting framework
   - eval_metric='logloss'

4. **LightGBM**
   - Uses `sample_weight` for imbalance
   - Fast gradient boosting framework
   - verbose=-1 (silent mode)

**Evaluation on Validation Set**: All models evaluated using Recall, Precision, F1-Score, ROC-AUC, and Accuracy

### 8. Hyperparameter Tuning

Optimize hyperparameters for each model using **RandomizedSearchCV** with **Recall as scoring metric** (primary metric for churn prediction).

#### Logistic Regression Tuning
- **Method**: RandomizedSearchCV
- **Parameters**:
  - C: [0.1, 1, 10, 100]
  - penalty: ['l1', 'l2']
  - solver: ['liblinear', 'lbfgs']
- **CV**: 5-fold
- **Iterations**: 10 random combinations
- **Scoring**: Recall

#### Random Forest Tuning
- **Method**: RandomizedSearchCV
- **Parameters**:
  - n_estimators: [100, 200, 300]
  - max_depth: [10, 20, 30, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
- **CV**: 5-fold
- **Iterations**: 15 random combinations
- **Scoring**: Recall

#### XGBoost Tuning
- **Method**: RandomizedSearchCV
- **Parameters**:
  - n_estimators: [100, 200, 300]
  - max_depth: [3, 5, 7]
  - learning_rate: [0.01, 0.1, 0.2]
  - subsample: [0.8, 0.9, 1.0]
  - colsample_bytree: [0.8, 0.9, 1.0]
- **CV**: 5-fold
- **Iterations**: 15 random combinations
- **Scoring**: Recall
- **Note**: Uses sample_weight during tuning

#### LightGBM Tuning
- **Method**: RandomizedSearchCV
- **Parameters**:
  - n_estimators: [100, 200, 300]
  - max_depth: [3, 5, 7, -1]
  - learning_rate: [0.01, 0.1, 0.2]
  - subsample: [0.8, 0.9, 1.0]
  - colsample_bytree: [0.8, 0.9, 1.0]
  - num_leaves: [31, 50, 70]
- **CV**: 5-fold
- **Iterations**: 15 random combinations
- **Scoring**: Recall
- **Note**: Uses sample_weight during tuning

### 9. Threshold Tuning ⭐

**Critical step**: Optimize classification threshold on validation set to maximize Recall.

- **Threshold Range**: Test thresholds from 0.1 to 0.9 (0.01 increments)
- **Optimization Metric**: Recall (can be changed to F1-score if needed)
- **Process**:
  1. Get probability predictions on validation set for all tuned models
  2. Test each threshold value
  3. Find threshold that maximizes Recall for each model
  4. Apply optimal thresholds to test set for final evaluation
- **Result**: Each model gets its own optimal threshold (often lower than 0.5 to catch more churners)

### 10. Evaluate Models on Test Set

Comprehensive evaluation on **unseen test data**:

- **Metrics Calculated**:
  - Recall (Primary metric - maximize true positives)
  - Precision (Minimize false positives)
  - F1-Score (Balance between precision and recall)
  - ROC-AUC (Overall model performance)
  - Accuracy (Overall correctness)

- **Visualizations**:
  - **Confusion Matrices**: Heatmaps for all models (threshold-optimized)
  - **Precision-Recall Comparison**: Bar charts comparing standard (0.5) vs threshold-optimized predictions
  - **ROC Curves**: ROC curves for all models with AUC scores

- **Feature Importance**: Top 15 most important features from Random Forest

## Models

### Primary Models
1. **Logistic Regression**
   - Baseline and tuned versions
   - Uses `class_weight='balanced'` for imbalance
   - L1/L2 regularization

2. **Random Forest**
   - Baseline and tuned versions
   - Uses `class_weight='balanced'` for imbalance
   - 100-300 estimators (tuned)

3. **XGBoost**
   - Baseline and tuned versions
   - Uses `sample_weight` for imbalance (no class_weight parameter)
   - Gradient boosting with tree-based learning

4. **LightGBM**
   - Baseline and tuned versions
   - Uses `sample_weight` for imbalance (no class_weight parameter)
   - Fast, distributed, high-performance gradient boosting

## Evaluation Metric

**Primary Metric: Recall** - Maximize the ability to identify customers who will churn

### Why Recall is Primary?
- **Business Priority**: Catching churning customers is more important than avoiding false alarms
- **Cost of Missing Churners**: Losing a customer is more expensive than a retention campaign
- **Actionable**: High Recall means we can proactively reach out to at-risk customers

### Secondary Metrics
- **Precision**: Minimize false positives (customers incorrectly flagged as churning)
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall model performance on imbalanced data
- **Accuracy**: Overall correctness (less important for imbalanced data)

## Features

Customer data typically includes:
- **Demographics**: Age, gender, location
- **Service Usage**: Monthly charges, total charges, tenure
- **Contract Details**: Contract type, payment method
- **Payment Information**: Payment history, billing details
- **Account History**: Account age, service subscriptions

## Handling Class Imbalance

The project uses multiple strategies:

1. **Class Weighting**:
   - `class_weight='balanced'` for Logistic Regression and Random Forest
   - Automatically adjusts weights inversely proportional to class frequencies

2. **Sample Weighting**:
   - `compute_sample_weight('balanced')` for XGBoost and LightGBM
   - Required because these models don't support class_weight parameter

3. **Threshold Tuning**:
   - Optimize classification threshold (default 0.5 may not be optimal)
   - Test thresholds from 0.1 to 0.9
   - Maximize Recall (can be changed to F1-score if needed)

## Data Leakage Prevention

Critical practices implemented:

1. **Split Before Encoding**: Data split into train/validation/test BEFORE any encoding
2. **Fit on Training Only**: Encoders and scalers fit only on training data
3. **Transform All Sets**: Validation and test sets transformed using fitted encoders/scalers
4. **Column Alignment**: Ensure all sets have same columns (add missing with 0s)

## Hyperparameter Tuning Strategy

- **Method**: RandomizedSearchCV (faster, explores more parameter space)
- **Scoring Metric**: **Recall** (primary metric for churn prediction)
- **Cross-Validation**: 5-fold CV
- **Iterations**: 10-15 random combinations per model
- **Training**: Models trained on training set, tuned using cross-validation
- **Evaluation**: Best models evaluated on validation set

## Threshold Optimization

After hyperparameter tuning, optimal classification thresholds are found:

- **Range**: 0.1 to 0.9 (0.01 increments = 80 threshold values)
- **Optimization**: Maximize Recall on validation set
- **Application**: Apply optimal thresholds to test set
- **Comparison**: Compare standard (0.5) vs threshold-optimized performance

## Output

The project generates:
1. **Model Performance Metrics**: Recall, Precision, F1-Score, ROC-AUC, Accuracy for all models
2. **Comparison Tables**: Standard vs threshold-optimized metrics
3. **Confusion Matrices**: Heatmaps for all models (threshold-optimized)
4. **Precision-Recall Comparison**: Bar charts showing improvement from threshold tuning
5. **ROC Curves**: ROC curves for all models with AUC scores
6. **Feature Importance**: Top 15 most important features from Random Forest
7. **Best Model Selection**: Model with highest Recall (threshold-optimized)

## Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
xgboost>=1.5.0
lightgbm>=3.3.0
```

## Applications

- **Customer Retention**: Identify at-risk customers for proactive retention campaigns
- **Marketing**: Target retention offers to customers likely to churn
- **Business Intelligence**: Understand factors that lead to customer churn
- **Resource Allocation**: Focus retention efforts on high-risk customers
- **Revenue Protection**: Prevent revenue loss by retaining customers

## Key Insights

1. **Recall is Critical**: For churn prediction, catching churners (high Recall) is more important than avoiding false alarms
2. **Threshold Tuning Matters**: Default threshold (0.5) may not be optimal; tuning significantly improves Recall
3. **Data Leakage Prevention**: Proper train/validation/test split and encoding prevents overfitting
4. **Class Imbalance Handling**: Multiple strategies (class_weight, sample_weight, threshold tuning) work together
5. **Model Diversity**: Different models (linear, tree-based, boosting) provide different perspectives
6. **Validation Set Importance**: Use validation set for tuning, test set only for final evaluation

## Usage

1. Load your customer churn dataset (update path in the notebook)
2. Run all cells sequentially
3. The notebook will:
   - Preprocess the data (handle missing values, encode, scale)
   - Split into train/validation/test sets
   - Train baseline models
   - Perform hyperparameter tuning (optimize for Recall)
   - Optimize classification thresholds
   - Evaluate on test set
   - Display visualizations and feature importance
   - Select best model

## Notes

- **Primary Metric**: Recall is used as primary metric because catching churners is the business priority
- **Threshold Selection**: Lower thresholds increase Recall (catch more churners) but may decrease Precision
- **Business Costs**: Consider cost of false positives vs false negatives when choosing threshold
- **Model Selection**: Best model selected based on Recall, but consider Precision for cost-sensitive scenarios
- **Regular Retraining**: Models should be retrained periodically with new data
- **Feature Engineering**: Consider creating new features (e.g., tenure groups, usage ratios) for better performance

## Model Performance Summary

After hyperparameter and threshold tuning:
- **Best Model**: Selected based on highest Recall (threshold-optimized)
- **Performance Improvement**: Threshold tuning significantly improves Recall for all models
- **Balance**: Models achieve high Recall while maintaining reasonable Precision
- **Practical Use**: Models can effectively identify customers likely to churn for retention campaigns

## Recommendations

1. **Deploy Best Model**: Use the model with highest Recall for production
2. **Monitor Performance**: Track model performance regularly and retrain as needed
3. **Business Integration**: Integrate predictions into customer relationship management (CRM) systems
4. **Retention Campaigns**: Use predictions to trigger automated retention offers
5. **Cost-Benefit Analysis**: Consider business costs when choosing betwen Recall and Precision
