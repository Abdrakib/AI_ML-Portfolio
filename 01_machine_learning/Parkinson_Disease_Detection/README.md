# Parkinson's Disease Detection

A comprehensive machine learning project to predict Parkinson's disease from voice signal features.

## Description

This project uses machine learning to detect Parkinson's disease from voice signal measurements. The model analyzes various voice features such as jitter, shimmer, fundamental frequency, and other acoustic parameters to classify patients as healthy or having Parkinson's disease.

## Problem Definition

Given voice signal features from patients, can we predict whether or not they have Parkinson's disease?

## Evaluation Metric

**Recall (Sensitivity)** is the primary metric, as missing a Parkinson's diagnosis has serious clinical consequences. We want to minimize false negatives.

## Dataset

The dataset contains 195 samples with 24 features including:
- MDVP (Multi-Dimensional Voice Program) features
- Jitter and Shimmer measurements
- Fundamental frequency parameters
- Nonlinear dynamical complexity measures (RPDE, DFA)
- Signal spread measures (spread1, spread2)
- Pitch period entropy (PPE)

**Target Variable**: `status` (0 = Healthy, 1 = Parkinson's)

## Project Pipeline

### 1. Data Cleaning ✅
- Handle missing values
- Remove duplicates
- Remove outliers using IQR method
- Fix and binarize labels

### 2. Exploratory Data Analysis (EDA) ✅
- Class imbalance visualization
- Feature distributions
- Correlation heatmaps
- Boxplots by class
- PCA visualization

### 3. Feature Engineering ✅
- **Standardization** (VERY important for SVM, Logistic Regression, Neural Networks)
- Feature selection using SelectKBest
- Optional PCA for dimensionality reduction

### 4. Train/Validation/Test Split ✅
- **70% Training** - Used for model training
- **15% Validation** - Used for hyperparameter tuning and model selection
- **15% Test** - Final evaluation on unseen data

### 5. Model Selection ✅
Multiple models are trained and compared:
- **Logistic Regression** - Simple, interpretable baseline
- **SVM (Support Vector Machine)** ⭐ - Very popular for Parkinson's detection
- **Random Forest** - Handles non-linear relationships
- **XGBoost** - Gradient boosting (if available)
- **Neural Network (MLP)** - Deep learning approach

### 6. Class Imbalance Handling ✅
- Calculate class weights using `compute_class_weight`
- Apply balanced class weights to models
- Threshold optimization to maximize recall

### 7. Hyperparameter Tuning ✅
- **GridSearchCV** for Logistic Regression (with recall scoring)
- **RandomizedSearchCV** for other models (with recall scoring)
- Cross-validation (5-fold) to prevent overfitting

### 8. Evaluation Metrics ✅
Comprehensive evaluation using:
- **Recall (Sensitivity)** ⭐ - Primary metric
- Precision
- F1-Score
- ROC-AUC
- PR-AUC (Precision-Recall AUC)
- Confusion Matrix

### 9. Threshold Optimization ✅
- Optimize decision threshold based on clinical cost
- Balance between missing disease (false negatives) and false positives
- Maximize recall while maintaining reasonable precision

### 10. Explainability 🧠 ✅
Healthcare requires interpretability:
- **SHAP values** (if available) - Model-agnostic explainability
- **Feature importance** - From Random Forest and XGBoost
- **Coefficients** - From Logistic Regression
- Partial dependence plots

## Key Features

- **Comprehensive Pipeline**: Complete ML workflow from data cleaning to model deployment
- **Multiple Models**: Compare 5+ different algorithms
- **Healthcare-Focused**: Prioritizes recall to minimize false negatives
- **Interpretable**: SHAP values and feature importance for clinical understanding
- **Robust Evaluation**: Multiple metrics and proper train/val/test split
- **Threshold Optimization**: Clinical cost-aware threshold selection

## Requirements

### Core Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Optional Dependencies
```
xgboost          # For XGBoost model
lightgbm          # For LightGBM model
shap              # For model explainability
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm shap  # Optional
```

## Usage

1. **Load the notebook**:
   ```bash
   jupyter notebook parkinson_detection.ipynb
   ```

2. **Data Loading**:
   - The notebook will try to load data from common paths
   - If data file not found, it will create synthetic data based on Parkinson's dataset structure
   - To use your own data, place `parkinsons.data` or `parkinsons.csv` in the project directory

3. **Run all cells**:
   - Execute cells sequentially
   - The notebook will automatically:
     - Clean and preprocess data
     - Perform EDA
     - Train multiple models
     - Evaluate and compare models
     - Generate visualizations

## Results

The best model is selected based on **Recall** score on the validation set. Final evaluation is performed on the held-out test set.

### Expected Outputs:
- Model comparison table
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- SHAP values (if available)

## Clinical Implications

- **High Recall**: Ensures we minimize false negatives (missing Parkinson's diagnosis)
- **Early Detection**: Can assist healthcare professionals in early diagnosis
- **Interpretability**: Feature importance helps understand which voice characteristics are most indicative of Parkinson's

## Model Performance

The best model typically achieves:
- **Recall**: > 85% (minimizing false negatives)
- **Precision**: > 75% (maintaining reasonable false positive rate)
- **ROC-AUC**: > 0.90
- **F1-Score**: > 0.80

*Note: Actual performance depends on data quality and model selection*

## File Structure

```
Parkinson_Disease_Detection/
├── parkinson_detection.ipynb    # Main notebook
├── README.md                    # This file
└── parkinsons.data              # Dataset 
```

## Notes

- The notebook handles missing data files by generating synthetic data
- All models use class weights to handle imbalanced data
- Threshold optimization is performed to balance clinical costs
- SHAP values require the `shap` library and work best with tree-based models

## References

- UCI Machine Learning Repository: Parkinson's Disease Classification Dataset
- Voice signal analysis for Parkinson's disease detection
- Clinical applications of ML in neurodegenerative disease diagnosis

## License

This project is for educational and research purposes.

