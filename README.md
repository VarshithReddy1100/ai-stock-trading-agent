# AI Stock Trading Agent Using Deep Learning

An AI-powered stock market research project: it cleans historical price data, engineers leak-free features, trains an **LSTM** to forecast the next bar's return, converts forecasts into transparent BUY / SELL / HOLD rules, and **backtests** them against buy-and-hold with realistic costs. Results are shown in an interactive Streamlit dashboard.

> **Educational research simulator.** It never places real trades and has no brokerage connection. Nothing here is financial advice.

---

## Status

Features are marked complete only after they were implemented **and** covered by passing tests (`pytest`: 95 tests).

| Area | Status |
|---|---|
| Data loading (Yahoo Finance, CSV), validation, cleaning, caching | ✅ Implemented and tested (live Yahoo download path tested with mocks; verify on your machine) |
| Exploratory data analysis (charts, summary stats, auto-generated findings) | ✅ |
| Feature engineering (returns, lags, SMA/EMA, volatility, RSI, MACD, volume) | ✅ |
| LSTM next-bar-return model (PyTorch), early stopping, save/load | ✅ |
| Baselines (persistence, momentum, ridge regression) and honest comparison | ✅ |
| Rule-based trading agent, position sizing, fees, slippage, stop-loss | ✅ |
| Backtester with buy-and-hold benchmark and risk metrics | ✅ |
| Streamlit dashboard | ✅ |
| Notebook walk-through, CLI, CI workflow | ✅ |
| Paper trading via a broker API | ⏳ Not implemented (future work) |

## What the model predicts (and why)

**Target: the simple return from close *t* to close *t+1*.** Predicting the next *price* is tempting but misleading: prices trend, so a model that just copies yesterday's close looks accurate. Returns are roughly stationary and scale-free, so errors mean something and one model design works across tickers. Direction (up/down) is derived from the sign of the predicted return and reported as *directional accuracy*; the predicted next close is `close_t × (1 + predicted return)`.

## How leakage is prevented

| Risk | Control |
|---|---|
| Features peeking at the future | All rolling/EWM windows are causal. Tests truncate the history and confirm earlier features are identical. |
| Scaler seeing test data | Scalers are fit on training rows only; a test rewrites the test period and confirms the scalers do not change. |
| Random splits | Strictly chronological 70 / 15 / 15 split with a purge gap (the last training label overlaps the first validation bar). Only mini-batches *inside* the training window are shuffled. |
| Tuning on the final test | Thresholds come from **validation** predictions only (`quantile` or `tuned` modes). Headline numbers are on the untouched test split. |
| Trading on information not yet known | A decision at the **close** of bar *t* fills at the **open** of bar *t+1*, after slippage and fees. A test confirms past equity is unchanged when future prices are altered. |
| Impossible trades | Integer shares by default, fees included in the budget, cash can never go negative (tested on random paths). |

## Architecture

```
Data → Clean/validate → Features → Chronological split + scaling → LSTM → Predicted next-bar return
     → Agent (BUY/SELL/HOLD) → Backtester (next-open fills, fees, slippage) → Metrics vs buy-and-hold → Dashboard
```

## Project structure

```
ai-stock-trading-agent/
├── data/                  # cache/ holds downloaded CSVs (git-ignored)
├── notebooks/
│   └── 01_eda_and_experiments.ipynb
├── src/
│   ├── config.py          # Dataclass settings (data, model, strategy)
│   ├── data_collection.py # yfinance / CSV loading, validation, cleaning, caching
│   ├── synthetic.py       # ARTIFICIAL demo data generator (offline use / tests only)
│   ├── preprocessing.py   # Features, target, splits, scaling, sequences
│   ├── eda.py             # Summary tables, Matplotlib/Seaborn charts, findings
│   ├── model.py           # LSTM, training loop, baselines, metrics, save/load bundle
│   ├── trading_agent.py   # Signal rules, thresholds, portfolio & cash management
│   ├── backtesting.py     # Simulation engine, metrics, validation-only threshold tuning
│   ├── pipeline.py        # End-to-end orchestration and report saving
│   └── visualization.py   # Plotly charts for the dashboard
├── tests/                 # pytest suite
├── app.py                 # Streamlit dashboard
├── train.py               # Command-line pipeline
├── requirements.txt
├── pytest.ini
├── .github/workflows/ci.yml
└── README.md
```

## Installation

Requires Python 3.11+ (tested on 3.13), Git and pip.

```bash
git clone https://github.com/VarshithReddy1100/ai-stock-trading-agent.git
cd ai-stock-trading-agent
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Usage

**Dashboard**

```bash
streamlit run app.py
```

Choose Yahoo Finance, an uploaded CSV, or the clearly labelled artificial demo data; adjust the model and strategy in the sidebar; press **Run pipeline**.

**Command line**

```bash
python train.py --ticker AAPL --start 2015-01-01
python train.py --csv data/my_prices.csv --epochs 40 --threshold-mode tuned
python train.py --synthetic                 # offline demo on ARTIFICIAL data
```

Outputs go to `reports/latest/`: `model.pt`, `meta.json` (scaler and feature names), `config.json`, `report.json`, `equity_curve.csv`, `trades.csv`, `test_predictions.csv`, and EDA figures.

**Reuse a trained model**

```python
from src.model import ModelBundle
bundle = ModelBundle.load("reports/latest")
predicted_next_returns = bundle.predict_returns(price_df)   # pandas Series indexed by decision date
```

**Tests**

```bash
pytest -q
```

## Strategy rules

Long-only, one position at a time, driven by the predicted next-bar return *r̂*:

* flat and *r̂* > buy threshold → **BUY** (at next open)
* long and *r̂* < sell threshold → **SELL** (at next open)
* long and close ≤ entry × (1 − stop-loss) → **SELL** (optional stop-loss)
* otherwise **HOLD**

Threshold modes: `quantile` (default; quantiles of validation predictions), `tuned` (grid search on the validation period by Sharpe ratio), `absolute` (you set them). Defaults: 0.10% fee per side, 5 bps slippage, integer shares, $10,000 capital — all configurable.

## Metrics and assumptions

* **MAE / RMSE** on returns, plus next-close error in dollars.
* **Directional accuracy**, compared against 50% using a standard-error check so a coin-flip result is not mistaken for skill.
* **Total return, CAGR, annualised volatility** (bar std × √periods per year; 252 for daily).
* **Sharpe ratio** = mean excess bar return / std × √periods per year, risk-free rate 0% by default.
* **Max drawdown, win rate, profit factor, time in market, fees paid.**
* Neither the strategy nor the benchmark is liquidated at the end, and open trades are excluded from win-rate statistics.

## Limitations

* Daily returns are very noisy. The app reports when the LSTM does **not** beat simple baselines, which is a common and legitimate outcome.
* One test window is a single sample of history; a good result can be luck. Try several tickers and periods.
* The simulator ignores taxes, market impact, liquidity limits, borrowing costs and dividends beyond price adjustment.
* Yahoo Finance data is unofficial, can be delayed or revised, and may be rate limited; download failures raise an error and are never papered over with substitute data.
* Overfitting is possible even with the safeguards here. Treat results as a research exercise, not a trading signal.

## Deploying on Streamlit Community Cloud

1. Push the repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repo with main file `app.py` and Python 3.11 or 3.12.
3. Optional: to keep the build small, add `--extra-index-url https://download.pytorch.org/whl/cpu` as the first line of `requirements.txt` so CPU-only PyTorch is installed.

Free-tier apps have limited memory and CPU, so keep epochs modest (the default dashboard run takes seconds to a minute).

## Future work

* Compare GRU / Transformer forecasters and multi-ticker training.
* Walk-forward (rolling) evaluation across many test windows.
* Portfolio allocation and richer risk management.
* Paper trading through a supported broker or market-data API.

## Disclaimer

This project is intended solely for educational and research purposes. It does not constitute financial or investment advice. Predictions and simulated results may be inaccurate, and no profitability is guaranteed. Always evaluate financial risks independently.

## Author

**Varshith Reddy Surakanti**

* GitHub: [@VarshithReddy1100](https://github.com/VarshithReddy1100)
* LinkedIn: [Varshith Reddy Surakanti](https://www.linkedin.com/in/varshith-reddy-surakanti-1a6106299/)
