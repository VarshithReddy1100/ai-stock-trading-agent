# AI Stock Trading Agent Using Deep Learning

An educational stock market analysis and trading simulation project that explores historical market data, generates predictive signals, and evaluates a trading strategy through backtesting.

## Overview

The AI Stock Trading Agent uses Python and Streamlit to present market data, model predictions, hypothetical trading signals, and historical strategy performance. It is intended for learning and research, not for making real-world investment decisions.

## Objectives

- Analyze historical stock market data.
- Explore time-series forecasting using deep learning.
- Generate hypothetical BUY, SELL, and HOLD signals.
- Evaluate strategy performance using historical backtesting.
- Visualize market trends and performance metrics.
- Understand the limitations and risks of automated trading systems.

## Features

- Historical market data analysis.
- Data preprocessing and feature engineering.
- Model-based forecasting.
- Rule-based trading signals.
- Historical strategy backtesting.
- Performance and risk metrics.
- Interactive Streamlit dashboard.
- Automated tests for supported project functionality.

The implemented features and results should be verified against the current source code and test suite.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Pytest
- Git and GitHub

Refer to `requirements.txt` for the project's Python dependencies.

## Project Structure

```text
ai-stock-trading-agent/
├── .github/
├── .streamlit/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── app.py
├── train.py
├── pytest.ini
├── requirements.txt
├── README.md
└── .gitignore
```

Generated data, models, and reports may be excluded from version control. Check the `.gitignore` rules before publishing any files needed to reproduce results.

## Installation and Setup

### Prerequisites

- Python compatible with the dependencies in `requirements.txt`
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/VarshithReddy1100/ai-stock-trading-agent.git
cd ai-stock-trading-agent
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python executable directly instead.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

### 5. Run the tests

```powershell
python -m pytest -q
```

### 6. Train the model

Inspect `train.py` and the project instructions to confirm the expected input data and training procedure before running:

```powershell
python train.py
```

## Evaluation

Depending on the available data and implemented components, evaluation may include:

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **Directional accuracy:** How often the predicted direction matches the actual direction.
- **Strategy return:** Return produced by the simulated strategy over the evaluated period.
- **Maximum drawdown:** Largest peak-to-trough decline.
- **Sharpe ratio:** Risk-adjusted performance estimate under the assumptions used.

Prediction accuracy does not necessarily imply profitable trading. Results depend on the selected data, testing period, model, and backtesting assumptions.

## Responsible Use and Limitations

- Historical performance does not guarantee future results.
- Financial markets are affected by unpredictable events.
- Models can overfit and may perform poorly in changing market conditions.
- Chronological train, validation, and test splits are important to reduce data leakage.
- Transaction costs, slippage, and execution assumptions can affect backtest results.
- Simulated trading results should not be treated as guaranteed or expected investment returns.

## Future Improvements

- Compare different forecasting models.
- Evaluate additional market features and technical indicators.
- Improve backtesting with realistic costs and slippage.
- Add reproducible experiments and model comparisons.
- Explore paper trading in a controlled environment.

## Disclaimer

This project is intended solely for educational and research purposes. It does not constitute financial or investment advice. Predictions and simulated trading results may be inaccurate, and profitability is not guaranteed.

## Author

**Varshith Reddy Surakanti**

- GitHub: https://github.com/VarshithReddy1100
- LinkedIn: https://www.linkedin.com/in/varshith-reddy-surakanti-1a6106299/
