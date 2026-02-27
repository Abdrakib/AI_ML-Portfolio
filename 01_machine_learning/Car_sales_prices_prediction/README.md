# Car Price Prediction (Regression)

## Problem
Predict the selling price of a car based on its attributes such as brand,
model year, mileage, fuel type, transmission, and other specifications.

## Dataset
Tabular dataset containing car features and the target variable
`Selling_Price` (or `Price`).
The dataset includes numerical and categorical features.

## Approach
1. Data cleaning and preprocessing
2. Handling missing values and outliers
3. Encoding categorical variables
4. Feature selection
5. Train-test split
6. Model training and evaluation
7. Hyperparameter tuning (RandomizedCv)

## Models Used
- Linear Regression (baseline)
- Random Forest Regressor *(or other models used in the notebook)*

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Results
The trained models were evaluated using regression metrics.
The improved model achieved better performance compared
to the baseline approach.

## How to Run
1. Open the notebook: `car_price_prediction.ipynb`
2. Run all cells sequentially

## Tools
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
