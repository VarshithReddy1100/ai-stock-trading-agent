"""Event-driven daily backtester with a buy-and-hold benchmark.

Timeline for every simulated bar ``d`` (in this order):
  1. Execute the order queued at the previous close, at the OPEN of ``d``
     (adverse slippage + fee applied).
  2. Mark the portfolio to market at the CLOSE of ``d``.
  3. If a prediction exists for ``d`` (made from data up to the close of ``d``),
     ask the agent for a decision and queue it for the next bar's open.

Hence no decision ever uses a price that was not yet known, and no fill uses the
same bar's information that generated the signal.

Conventions
-----------
* Neither the strategy nor the benchmark is liquidated at the end of the test, so
  neither pays a final exit fee. Open trades are excluded from win-rate statistics.
* Sharpe = mean(excess bar return) / std(bar return) * sqrt(periods_per_year), using
  the strategy's ``risk_free_rate`` (default 0). Volatility is annualised the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .trading_agent import Portfolio, Signal, TradingAgent, calibrate_thresholds


@dataclass
class BacktestResult:
    equity: pd.DataFrame  # date, cash, shares, close, equity, position
    benchmark: pd.DataFrame  # same layout, buy-and-hold
    trades: pd.DataFrame
    signals: pd.DataFrame  # date, pred_return, action, reason
    metrics: dict
    benchmark_metrics: dict
    thresholds: dict = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame({"Strategy": self.metrics, "Buy & Hold": self.benchmark_metrics})


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def compute_metrics(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    exposure: Optional[pd.Series] = None,
    periods_per_year: float = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    equity = equity.astype(float)
    rets = equity.pct_change().dropna()
    n = len(rets)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = n / periods_per_year if n else 0.0
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 and total_return > -1 else float("nan")
    std = float(rets.std(ddof=1)) if n > 1 else float("nan")
    ann_vol = std * np.sqrt(periods_per_year) if np.isfinite(std) else float("nan")
    if np.isfinite(std) and std > 0:
        sharpe = float((rets.mean() - risk_free_rate / periods_per_year) / std * np.sqrt(periods_per_year))
    else:
        sharpe = float("nan")

    m = {
        "Initial equity": float(equity.iloc[0]),
        "Final equity": float(equity.iloc[-1]),
        "Total return": total_return,
        "Annualized return (CAGR)": cagr,
        "Annualized volatility": float(ann_vol),
        "Sharpe ratio": sharpe,
        "Max drawdown": _max_drawdown(equity),
        "Bars": int(len(equity)),
    }
    if exposure is not None:
        m["Time in market"] = float(exposure.mean())
    if trades is not None:
        sells = trades[trades["side"] == "SELL"]
        m["Round-trip trades"] = int(len(sells))
        m["Win rate"] = float((sells["pnl"] > 0).mean()) if len(sells) else float("nan")
        m["Avg trade return"] = float(sells["return_pct"].mean()) if len(sells) else float("nan")
        gross_win = float(sells.loc[sells["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-sells.loc[sells["pnl"] < 0, "pnl"].sum())
        m["Profit factor"] = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else float("nan"))
        m["Total fees paid"] = float(trades["fee"].sum()) if len(trades) else 0.0
        m["Open position at end"] = bool(len(trades) and trades.iloc[-1]["side"] == "BUY")
    return m


def run_backtest(
    prices: pd.DataFrame,
    predictions: pd.Series,
    config: StrategyConfig,
    buy_threshold: float,
    sell_threshold: float,
    periods_per_year: float = 252,
) -> BacktestResult:
    """Simulate the strategy. ``predictions[t]`` is the forecast of return t->t+1 made at close ``t``."""
    pred = predictions.dropna().sort_index()
    pred = pred[pred.index.isin(prices.index)]
    if len(pred) < 2:
        raise ValueError("Need at least two predictions inside the price history to backtest.")

    dates = prices.index
    first_pos = dates.get_loc(pred.index[0])
    last_pos = dates.get_loc(pred.index[-1])
    end_pos = min(last_pos + 1, len(dates) - 1)  # include the bar that fills the last order
    sim = prices.iloc[first_pos : end_pos + 1]
    if len(sim) < 3:
        raise ValueError("Backtest window is too short.")

    agent = TradingAgent(config, buy_threshold, sell_threshold)
    pf = Portfolio(config)
    pending: Optional[tuple[Signal, str]] = None

    rows, signal_rows = [], []
    for d, bar in sim.iterrows():
        if pending is not None:
            action, reason = pending
            if action is Signal.BUY:
                pf.buy(d, bar["Open"], reason)
            elif action is Signal.SELL:
                pf.sell_all(d, bar["Open"], reason)
            pending = None

        rows.append(
            dict(date=d, cash=pf.cash, shares=pf.shares, close=bar["Close"],
                 equity=pf.equity(bar["Close"]), position=int(pf.shares > 0))
        )

        if d in pred.index:
            action, reason = agent.decide(pred.loc[d], pf.shares > 0, bar["Close"], pf.entry_price)
            signal_rows.append(dict(date=d, pred_return=float(pred.loc[d]), action=action.value, reason=reason))
            if action is not Signal.HOLD:
                pending = (action, reason)

    equity = pd.DataFrame(rows).set_index("date")
    signals = pd.DataFrame(signal_rows).set_index("date")
    trades = pf.trades_frame()

    # ---- Buy & hold benchmark: buys at the first tradable open, same costs, never sells ----
    bh = Portfolio(replace(config, position_fraction=1.0))
    bh_rows = []
    for i, (d, bar) in enumerate(sim.iterrows()):
        if i == 1:  # first order can only fill on the bar after the first decision bar
            bh.buy(d, bar["Open"], "buy_and_hold")
        bh_rows.append(
            dict(date=d, cash=bh.cash, shares=bh.shares, close=bar["Close"],
                 equity=bh.equity(bar["Close"]), position=int(bh.shares > 0))
        )
    benchmark = pd.DataFrame(bh_rows).set_index("date")

    metrics = compute_metrics(equity["equity"], trades, equity["position"], periods_per_year, config.risk_free_rate)
    metrics["Skipped orders"] = int(pf.skipped_orders)
    bh_metrics = compute_metrics(
        benchmark["equity"], bh.trades_frame(), benchmark["position"], periods_per_year, config.risk_free_rate
    )
    return BacktestResult(
        equity=equity,
        benchmark=benchmark,
        trades=trades,
        signals=signals,
        metrics=metrics,
        benchmark_metrics=bh_metrics,
        thresholds={"buy_threshold": float(buy_threshold), "sell_threshold": float(sell_threshold)},
    )


def tune_thresholds_on_validation(
    prices: pd.DataFrame,
    val_predictions: pd.Series,
    config: StrategyConfig,
    periods_per_year: float = 252,
    buy_quantiles: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
    sell_quantiles: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5),
    min_trades: int = 3,
) -> tuple[float, float, pd.DataFrame]:
    """Grid-search the (buy, sell) quantile pair on the VALIDATION period by Sharpe ratio.

    The test period is never touched. Returns ``(buy_threshold, sell_threshold, grid)``;
    falls back to the configured quantiles if no pair produces at least ``min_trades`` trades.
    """
    records = []
    for bq, sq in product(buy_quantiles, sell_quantiles):
        if sq > bq:
            continue
        bt, st = calibrate_thresholds(val_predictions, bq, sq)
        try:
            res = run_backtest(prices, val_predictions, config, bt, st, periods_per_year)
        except ValueError:
            continue
        sharpe = res.metrics["Sharpe ratio"]
        records.append(
            dict(buy_quantile=bq, sell_quantile=sq, buy_threshold=bt, sell_threshold=st,
                 sharpe=sharpe, total_return=res.metrics["Total return"],
                 trades=res.metrics["Round-trip trades"])
        )
    grid = pd.DataFrame(records)
    eligible = grid[(grid["trades"] >= min_trades) & grid["sharpe"].notna()] if len(grid) else grid
    if len(eligible) == 0:
        bt, st = calibrate_thresholds(val_predictions, config.buy_quantile, config.sell_quantile)
        return bt, st, grid
    best = eligible.sort_values("sharpe", ascending=False).iloc[0]
    return float(best["buy_threshold"]), float(best["sell_threshold"]), grid
