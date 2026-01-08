# Laptop Price Prediction (Regression)

## Problem
Predict laptop prices based on specifications (brand, CPU, RAM, storage, GPU, screen size, etc.).
This is a regression problem.

## Dataset
Tabular dataset of laptop specifications with a target column `Price`.

## Important Note (Target Transformation)
In this notebook, the target is transformed using:
- `y = np.log(Price)`

So evaluation metrics (MAE and R²) are reported on the **log-price scale**.
To convert predictions back to original price units:
- `price_pred = np.exp(y_pred)`

## Workflow
1. Data cleaning + feature engineering (extract and clean spec columns)
2. Split data into train/test
3. Build a preprocessing + model pipeline
4. Train baseline models (tested multiple regressors)
5. Tune a Random Forest model using RandomizedSearchCV
6. Evaluate the best model on the test set

## Preprocessing
- Categorical features are One-Hot Encoded using a `ColumnTransformer`
- Remaining columns are passed through

## Model
Best performing model:
- **Random Forest Regressor** inside a Scikit-learn `Pipeline`

## Hyperparameter Tuning (RandomizedSearchCV)
- Method: `RandomizedSearchCV`
- Iterations: `n_iter = 30`
- Cross-validation folds: `cv = 5`
- Scoring: `r2`
- Random seed: `random_state = 42`

Parameters searched included:
- `n_estimators`, `max_depth`, `max_features`,
  `min_samples_split`, `min_samples_leaf`, `max_samples`

Best parameters found (from notebook output):
- `n_estimators = 500`
- `max_depth = 30`
- `max_features = 0.5`
- `max_samples = 0.75`
(+ tuned values for `min_samples_split` and `min_samples_leaf`)

## Results (from notebook)
Test R² (best): **0.8950139592049811**  
Test MAE: **0.1536517154334115**  *(log-price scale)*

## Files
- `laptop price predictor.ipynb` — full notebook (preprocessing, training, tuning, evaluation)

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, RandomizedSearchCV)
- Matplotlib / Seaborn
