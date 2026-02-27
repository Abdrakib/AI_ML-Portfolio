# Stock Return & Direction Prediction (ML + Backtest)

## Goal
Predict **next-day return** and **next-day direction (up/down)** using a leakage-safe ML pipeline built on historical prices.

## What you get
- **Targets**: next-day return (regression) and next-day direction (classification)\n+- **Features**: lagged returns, rolling mean/std (volatility), momentum, RSI, MACD, Bollinger z-score\n+- **Split**: chronological train → validation → test\n+- **Models**: Ridge (return) + Logistic Regression (direction)\n+- **Baselines + evaluation**: directional baseline + error metrics\n+- **Backtest**: simple long/flat strategy with transaction costs (default 5 bps)

## Run (script)
From this folder:

```bash
pip install -r requirements.txt
python stock_ml_project.py --csv stock_data.csv
```

It prints a results table and saves `equity_curve.png` for the best stock column.
