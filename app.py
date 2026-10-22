"""Streamlit dashboard for the AI Stock Trading Agent.

Run with:  streamlit run app.py
Educational research simulator -- no real trading, no brokerage connection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import PipelineConfig
from src.data_collection import DataError, load_csv, load_prices
from src.eda import run_eda
from src.model import TrainingError
from src.pipeline import run_pipeline
from src.synthetic import generate_synthetic_ohlcv
from src.visualization import (
    drawdown_chart,
    equity_chart,
    prediction_chart,
    price_with_signals,
    training_chart,
)

st.set_page_config(page_title="AI Stock Trading Agent", page_icon="📈", layout="wide")

PCT_KEYS = {
    "Total return", "Annualized return (CAGR)", "Annualized volatility", "Max drawdown",
    "Time in market", "Win rate", "Avg trade return",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Downloading market data...", ttl=3600)
def cached_prices(ticker: str, start: str, end: str | None, interval: str, min_rows: int):
    return load_prices(ticker, start, end, interval, min_rows=min_rows)


def fmt(key: str, value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a" if not (isinstance(value, float) and np.isinf(value)) else "inf"
    if key in PCT_KEYS:
        return f"{value:.2%}"
    if key in ("Initial equity", "Final equity", "Total fees paid"):
        return f"${value:,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.3f}"


def metrics_table(result) -> pd.DataFrame:
    t = result.backtest.summary_table()
    return t.apply(lambda col: [fmt(k, v) for k, v in zip(t.index, col)]).set_axis(t.index)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.title("📈 AI Stock Trading Agent")
st.caption(
    "LSTM next-day return forecasts → rule-based BUY/SELL/HOLD → historical backtest. "
    "**Educational research simulator — not financial advice.**"
)

with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Yahoo Finance", "Upload CSV", "Synthetic demo (artificial)"])
    cfg = PipelineConfig()
    uploaded = None
    if source == "Yahoo Finance":
        cfg.data.ticker = st.text_input("Ticker", "AAPL").strip().upper()
        c1, c2 = st.columns(2)
        cfg.data.start = str(c1.date_input("Start", pd.Timestamp("2015-01-01")))
        end_date = c2.date_input("End (optional)", value=None)
        cfg.data.end = str(end_date) if end_date else None
        cfg.data.interval = st.selectbox("Interval", ["1d", "1wk"], help="Daily is recommended.")
    elif source == "Upload CSV":
        uploaded = st.file_uploader("OHLCV CSV", type="csv", help="Columns: Date, Open, High, Low, Close, Volume")
    else:
        st.warning("Synthetic data is artificial. Use it only to try the app offline.")

    st.header("2 · Model")
    m = cfg.model
    m.lookback = st.slider("Lookback window (bars)", 10, 90, m.lookback, 5)
    m.hidden_size = st.select_slider("LSTM hidden size", [16, 32, 64, 128], m.hidden_size)
    m.epochs = st.slider("Max epochs", 5, 150, 40, 5)
    m.batch_size = st.select_slider("Batch size", [16, 32, 64, 128], m.batch_size)
    m.learning_rate = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], m.learning_rate)
    m.seed = int(st.number_input("Random seed", 0, 10_000, m.seed))

    st.header("3 · Strategy")
    s = cfg.strategy
    s.initial_capital = float(st.number_input("Initial capital ($)", 1_000.0, 10_000_000.0, 10_000.0, 1_000.0))
    s.threshold_mode = st.selectbox(
        "Signal thresholds",
        ["quantile", "tuned", "absolute"],
        help="quantile: from validation predictions · tuned: grid-searched on validation only · absolute: you choose.",
    )
    if s.threshold_mode == "absolute":
        s.buy_threshold = st.number_input("Buy if predicted return >", value=0.001, format="%.4f", step=0.0005)
        s.sell_threshold = st.number_input("Sell if predicted return <", value=-0.001, format="%.4f", step=0.0005)
    elif s.threshold_mode == "quantile":
        s.buy_quantile = st.slider("Buy quantile", 0.5, 0.95, s.buy_quantile, 0.05)
        s.sell_quantile = st.slider("Sell quantile", 0.05, 0.5, s.sell_quantile, 0.05)
    s.position_fraction = st.slider("Cash deployed per BUY", 0.1, 1.0, 1.0, 0.1)
    s.fee_rate = st.number_input("Fee per side (%)", 0.0, 2.0, 0.10, 0.05) / 100
    s.slippage_bps = st.number_input("Slippage (bps)", 0.0, 100.0, 5.0, 1.0)
    s.fractional_shares = st.checkbox("Allow fractional shares", False)
    sl = st.number_input("Stop-loss (%, 0 = off)", 0.0, 50.0, 0.0, 1.0)
    s.stop_loss_pct = sl / 100 if sl > 0 else None

    run = st.button("▶ Run pipeline", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if run:
    try:
        if source == "Yahoo Finance":
            df, report = cached_prices(cfg.data.ticker, cfg.data.start, cfg.data.end, cfg.data.interval, cfg.data.min_rows)
            label = cfg.data.ticker
        elif source == "Upload CSV":
            if uploaded is None:
                st.error("Please upload a CSV file first.")
                st.stop()
            df, report = load_csv(uploaded, cfg.data.min_rows)
            label = uploaded.name
        else:
            df = generate_synthetic_ohlcv()
            report = {"source": "synthetic (artificial)", "rows_out": len(df),
                      "first_date": str(df.index[0].date()), "last_date": str(df.index[-1].date())}
            label = "SYNTHETIC"

        bar = st.progress(0.0, text="Training LSTM...")

        def on_epoch(epoch, total, tr, va):
            bar.progress(epoch / total, text=f"Epoch {epoch}/{total} · train {tr:.4f} · val {va:.4f}")

        result = run_pipeline(df, cfg, progress=on_epoch)
        bar.empty()
        st.session_state.update(result=result, df=df, report=report, label=label, eda=None)
    except (DataError, TrainingError, ValueError) as exc:
        st.error(f"**Could not complete the run:** {exc}")
        st.stop()

if "result" not in st.session_state:
    st.info("Choose a data source in the sidebar and press **Run pipeline**.")
    st.markdown(
        "**What happens:** data is cleaned → features are engineered → an LSTM is trained on the earliest 70% "
        "of history (validated on the next 15%) → signals are generated → everything is backtested on the final, "
        "untouched 15%, with fees and slippage, against buy-and-hold."
    )
    st.stop()

result, df = st.session_state["result"], st.session_state["df"]
report, label = st.session_state["report"], st.session_state["label"]
bt = result.backtest

if label == "SYNTHETIC":
    st.warning("Showing results on ARTIFICIAL data. They say nothing about real markets.")
st.caption(
    f"**{label}** · {report.get('rows_out', len(df))} bars · {report.get('first_date')} → {report.get('last_date')} "
    f"· source: {report.get('source')}"
)

# Headline numbers
c = st.columns(4)
c[0].metric("Strategy return (test)", fmt("Total return", bt.metrics["Total return"]),
            delta=f"{bt.metrics['Total return'] - bt.benchmark_metrics['Total return']:+.2%} vs buy & hold")
c[1].metric("Buy & hold return", fmt("Total return", bt.benchmark_metrics["Total return"]))
c[2].metric("Max drawdown", fmt("Max drawdown", bt.metrics["Max drawdown"]))
c[3].metric("Sharpe ratio", fmt("Sharpe ratio", bt.metrics["Sharpe ratio"]))

tab_data, tab_model, tab_bt, tab_trades, tab_about = st.tabs(
    ["Data & EDA", "Model", "Backtest", "Trades & signals", "Method & limits"]
)

with tab_data:
    st.subheader("Cleaning report")
    st.json({k: v for k, v in report.items()}, expanded=False)
    if st.session_state.get("eda") is None:
        with st.spinner("Building EDA charts..."):
            st.session_state["eda"] = run_eda(df, periods_per_year=result.config.data.periods_per_year)
    eda = st.session_state["eda"]
    st.subheader("What the data shows")
    for f in eda["findings"]:
        st.markdown(f"- {f}")
    st.pyplot(eda["figures"]["price_ma"])
    col1, col2 = st.columns(2)
    col1.pyplot(eda["figures"]["returns"])
    col2.pyplot(eda["figures"]["volatility"])
    st.pyplot(eda["figures"]["volume"])
    st.pyplot(eda["figures"]["correlation"])
    st.subheader("Summary statistics")
    st.dataframe(eda["summary"].style.format("{:,.4f}"), use_container_width=True)
    st.subheader("Missing values (after cleaning)")
    st.dataframe(eda["missing"], use_container_width=True)

with tab_model:
    st.subheader("Chronological splits")
    st.dataframe(result.split_summary, use_container_width=True)
    st.caption("A one-sample gap is purged between splits because each label looks one bar ahead.")
    st.plotly_chart(training_chart(result.history), use_container_width=True)
    h = result.history
    st.caption(f"Best epoch {h['best_epoch']} · stopped at epoch {h['stopped_epoch']} (early stopping on validation loss).")
    st.subheader("Test-set accuracy vs simple baselines")
    st.dataframe(result.comparison.style.format("{:.5f}", subset=["MAE", "RMSE", "Next-close MAE ($)"])
                 .format("{:.1%}", subset=["Directional accuracy"], na_rep="n/a"), use_container_width=True)
    st.caption("Directional accuracy is n/a for persistence: it always predicts 0, so it never calls a direction.")
    st.info(result.verdict)
    st.plotly_chart(prediction_chart(result.prediction_frame), use_container_width=True)

with tab_bt:
    st.plotly_chart(equity_chart(bt), use_container_width=True)
    st.plotly_chart(drawdown_chart(bt), use_container_width=True)
    st.subheader("Performance metrics (test period)")
    st.dataframe(metrics_table(result), use_container_width=True)
    ti = result.threshold_info
    st.caption(
        f"Thresholds chosen {ti['method']}: BUY when predicted next-bar return > {ti['buy_threshold']:.5f}, "
        f"SELL when < {ti['sell_threshold']:.5f}. They were never fit on the test period. "
        f"Orders fill at the next bar's open with {result.config.strategy.fee_rate:.2%} fees and "
        f"{result.config.strategy.slippage_bps:.0f} bps slippage."
    )
    if result.threshold_grid is not None and len(result.threshold_grid):
        with st.expander("Validation grid used for tuning"):
            st.dataframe(result.threshold_grid.sort_values("sharpe", ascending=False), use_container_width=True)

with tab_trades:
    st.plotly_chart(price_with_signals(df, bt.trades, start=bt.equity.index[0]), use_container_width=True)
    st.subheader("Trade history")
    if len(bt.trades):
        st.dataframe(bt.trades.style.format(
            {"shares": "{:,.2f}", "price": "{:,.2f}", "notional": "{:,.2f}", "fee": "{:,.2f}",
             "cash_after": "{:,.2f}", "pnl": "{:,.2f}", "return_pct": "{:.2%}"}, na_rep="—"),
            use_container_width=True)
        st.download_button("Download trades (CSV)", bt.trades.to_csv(index=False), "trades.csv")
    else:
        st.write("The strategy made no trades with these settings.")
    st.subheader("Daily signals")
    st.dataframe(bt.signals, use_container_width=True)

with tab_about:
    st.markdown(
        """
**Target.** The model predicts the *next-bar simple return* (close *t* → close *t+1*) from the previous
`lookback` bars of scale-free features (returns, lagged returns, SMA/EMA distances, rolling volatility, RSI, MACD,
volume ratio). Returns are stationary; predicting the price level would flatter a model that merely copies
yesterday's close.

**No look-ahead.** Features use only data up to the decision bar. Scalers are fit on the training window only.
Splits are chronological with purge gaps. Signals made at a close are filled at the *next* open.

**Honest comparison.** The LSTM is compared with persistence, momentum and ridge-regression baselines. The app
reports when it does *not* win — which, on noisy daily returns, is common.

**Limits.** Backtests ignore taxes, market impact, dividends beyond price adjustment, borrow costs and liquidity.
Results from a single test window are one sample of history and can be luck. Past performance does not predict
future results. This tool never places real trades.
        """
    )
