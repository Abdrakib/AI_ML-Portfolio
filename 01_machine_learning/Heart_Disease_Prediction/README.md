# Heart Disease Prediction Project

A simple Python project that predicts heart disease using machine learning.

## Quick Start

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python heart_disease_prediction.py
```

## Data


- `heart_disease.csv`

The script will automatically detect and use your CSV file. Make sure your CSV has:
- Multiple feature columns (age, sex, cp, etc.)
- A target column (usually the last column) indicating heart disease (0 = no disease, 1 = disease)

## What the Project Does

1. Loads heart disease data (or creates synthetic data if no CSV is found)
2. Splits data into training and test sets
3. Trains three machine learning models:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors
4. Evaluates each model and prints the accuracy
5. Identifies and displays the best performing model

## Output

The script prints:
- Dataset information
- Training progress for each model
- **Final accuracy for each model**
- **Best model with highest accuracy**
- Detailed classification report

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

All requirements are listed in `requirements.txt`.
