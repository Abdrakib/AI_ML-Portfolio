# Laptop Price Prediction

## Problem
Predict the price of a laptop based on its specifications such as brand,
processor, RAM, storage, GPU, operating system, and screen size.

## Dataset
Tabular dataset containing laptop features and the target variable `Price`.
The dataset includes both numerical and categorical features.

## Approach
1. Data cleaning and preprocessing
2. Handling categorical features using encoding
3. Feature selection and transformation
4. Train-test split
5. Model training and evaluation
6. Model comparison to select the best performer

## Models Used
- Linear Regression (baseline)
- Random Forest Regressor
- Decision Tree *(or other models used in the notebook)*

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Results
The model performance was evaluated using regression metrics.
The best-performing model achieved improved accuracy compared
to the baseline model.

## How to Run
1. Open the notebook: `laptop_price_prediction.ipynb`
2. Run all cells sequentially

## Tools
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
