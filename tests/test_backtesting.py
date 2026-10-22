import numpy as np
import pandas as pd
import pytest

from src.backtesting import compute_metrics, run_backtest, tune_thresholds_on_validation
from src.config import StrategyConfig
from src.synthetic import generate_synthetic_ohlcv

FRICTIONLESS = dict(initial_capital=1_000, fee_rate=0.0, slippage_bps=0.0, fractional_shares=True)


def toy_prices(n=8):
    idx = pd.bdate_range("2024-01-01", periods=n)
    opens = np.arange(10.0, 10.0 + n)
    return pd.DataFrame(
        {"Open": opens, "High": opens + 1, "Low": opens - 0.5, "Close": opens + 0.5, "Volume": 1e6}, index=idx
    )


def toy_preds(prices, values):
    return pd.Series(values, index=prices.index[: len(values)])


def test_orders_fill_at_next_open_not_same_bar():
    px = toy_prices()
    #                      d0    d1    d2     d3   d4
    preds = toy_preds(px, [0.1, 0.1, -0.1, 0.0, 0.0])
    r = run_backtest(px, preds, StrategyConfig(**FRICTIONLESS), 0.05, -0.05)
    buy, sell = r.trades.iloc[0], r.trades.iloc[1]
    assert buy["side"] == "BUY" and buy["date"] == px.index[1] and buy["price"] == px["Open"].iloc[1]
    assert sell["side"] == "SELL" and sell["date"] == px.index[3] and sell["price"] == px["Open"].iloc[3]
    assert len(r.trades) == 2  # repeated BUY signal while long is ignored
    assert r.equity["cash"].iloc[0] == 1_000 and r.equity["shares"].iloc[0] == 0  # nothing before first fill
    expected_cash = 1_000 / 11.0 * 13.0
    assert r.equity["equity"].iloc[-1] == pytest.approx(expected_cash)
    assert r.metrics["Round-trip trades"] == 1 and r.metrics["Win rate"] == 1.0


def test_equity_is_marked_at_close():
    px = toy_prices()
    r = run_backtest(px, toy_preds(px, [0.1, 0, 0, 0, 0]), StrategyConfig(**FRICTIONLESS), 0.05, -0.05)
    shares = 1_000 / 11.0
    assert r.equity["equity"].iloc[1] == pytest.approx(shares * px["Close"].iloc[1])


def test_past_equity_unaffected_by_future_prices():
    px = toy_prices(12)
    preds = toy_preds(px, [0.1, 0, 0, -0.1, 0.1, 0, 0, -0.1, 0])
    cfg = StrategyConfig(**FRICTIONLESS)
    base = run_backtest(px, preds, cfg, 0.05, -0.05)
    future = px.copy()
    future.iloc[6:, :4] *= 3.0  # wreck prices from bar 6 onward
    alt = run_backtest(future, preds, cfg, 0.05, -0.05)
    pd.testing.assert_frame_equal(base.equity.iloc[:6], alt.equity.iloc[:6])
    assert base.trades[base.trades["date"] < px.index[6]].equals(alt.trades[alt.trades["date"] < px.index[6]])


def test_costs_reduce_buy_fill_and_are_tracked():
    px = toy_prices()
    cfg = StrategyConfig(initial_capital=1_000, fee_rate=0.01, slippage_bps=100, fractional_shares=True)
    r = run_backtest(px, toy_preds(px, [0.1, 0, -0.1, 0, 0]), cfg, 0.05, -0.05)
    buy, sell = r.trades.iloc[0], r.trades.iloc[1]
    assert buy["price"] == pytest.approx(11.0 * 1.01) and sell["price"] == pytest.approx(px["Open"].iloc[3] * 0.99)
    assert r.metrics["Total fees paid"] == pytest.approx(buy["fee"] + sell["fee"])


def test_stop_loss_exits_next_open():
    idx = pd.bdate_range("2024-01-01", periods=7)
    opens = [100, 100, 100, 80, 80, 80, 80]
    closes = [100, 100, 100, 80, 80, 80, 80]
    px = pd.DataFrame({"Open": opens, "High": np.array(opens) + 1, "Low": np.array(opens) - 1,
                       "Close": closes, "Volume": 1e6}, index=idx, dtype=float)
    preds = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], index=idx[:6])  # always bullish
    cfg = StrategyConfig(**FRICTIONLESS, stop_loss_pct=0.1)
    r = run_backtest(px, preds, cfg, 0.05, -0.05)
    sells = r.trades[r.trades["side"] == "SELL"]
    assert len(sells) == 1 and sells.iloc[0]["reason"] == "stop_loss"
    assert sells.iloc[0]["date"] == idx[4]  # close crossed on bar 3, exit fills at bar 4's open


def test_never_spends_more_than_cash_random_walk():
    px = generate_synthetic_ohlcv(n=400, seed=2)
    rng = np.random.default_rng(0)
    preds = pd.Series(rng.normal(0, 0.01, 300), index=px.index[50:350])
    cfg = StrategyConfig(initial_capital=500, fee_rate=0.002, slippage_bps=10)
    r = run_backtest(px, preds, cfg, 0.002, -0.002)
    assert (r.equity["cash"] >= -1e-9).all() and (r.equity["shares"] >= 0).all()
    assert (r.equity["equity"] > 0).all()
    assert r.trades["shares"].apply(lambda s: float(s).is_integer()).all()
    # trade sides alternate BUY, SELL, BUY, ...
    assert list(r.trades["side"]) == ["BUY", "SELL"] * (len(r.trades) // 2) + (["BUY"] if len(r.trades) % 2 else [])


def test_buy_and_hold_benchmark():
    px = toy_prices()
    r = run_backtest(px, toy_preds(px, [-0.1, -0.1, -0.1, -0.1, -0.1]), StrategyConfig(**FRICTIONLESS), 0.05, -0.05)
    assert len(r.trades) == 0 and r.metrics["Total return"] == 0.0  # never bought
    shares = 1_000 / px["Open"].iloc[1]
    last_bar = r.benchmark.index[-1]
    assert last_bar == px.index[5]  # window ends on the bar that would fill the last order
    assert r.benchmark["equity"].iloc[0] == 1_000
    assert r.benchmark["equity"].iloc[-1] == pytest.approx(shares * px.loc[last_bar, "Close"])
    assert r.benchmark_metrics["Total return"] == pytest.approx(shares * px.loc[last_bar, "Close"] / 1_000 - 1)
    assert list(r.benchmark.index) == list(r.equity.index)  # same dates as the strategy


def test_signal_log_records_every_decision():
    px = toy_prices()
    r = run_backtest(px, toy_preds(px, [0.1, 0.1, -0.1, 0, 0]), StrategyConfig(**FRICTIONLESS), 0.05, -0.05)
    assert list(r.signals["action"]) == ["BUY", "HOLD", "SELL", "HOLD", "HOLD"]


def test_too_few_predictions_rejected():
    px = toy_prices()
    with pytest.raises(ValueError):
        run_backtest(px, toy_preds(px, [0.1]), StrategyConfig(), 0.0, 0.0)


def test_compute_metrics_known_values():
    eq = pd.Series([100.0, 110.0, 99.0, 121.0])
    m = compute_metrics(eq, periods_per_year=252)
    rets = eq.pct_change().dropna()
    assert m["Total return"] == pytest.approx(0.21)
    assert m["Max drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["Annualized volatility"] == pytest.approx(rets.std(ddof=1) * np.sqrt(252))
    assert m["Sharpe ratio"] == pytest.approx(rets.mean() / rets.std(ddof=1) * np.sqrt(252))


def test_sharpe_uses_risk_free_rate_and_handles_flat_curve():
    eq = pd.Series([100.0, 101.0, 100.5, 102.0, 103.0])
    s0 = compute_metrics(eq, risk_free_rate=0.0)["Sharpe ratio"]
    s1 = compute_metrics(eq, risk_free_rate=0.05)["Sharpe ratio"]
    assert s1 < s0
    flat = compute_metrics(pd.Series([100.0] * 5))
    assert np.isnan(flat["Sharpe ratio"]) and flat["Max drawdown"] == 0


def test_trade_statistics():
    trades = pd.DataFrame({
        "side": ["BUY", "SELL", "BUY", "SELL", "BUY", "SELL"],
        "pnl": [np.nan, 50.0, np.nan, -20.0, np.nan, 30.0],
        "return_pct": [np.nan, 0.05, np.nan, -0.02, np.nan, 0.03],
        "fee": [1.0] * 6,
    })
    m = compute_metrics(pd.Series([100.0, 101.0, 102.0]), trades)
    assert m["Round-trip trades"] == 3 and m["Win rate"] == pytest.approx(2 / 3)
    assert m["Profit factor"] == pytest.approx(80 / 20) and m["Total fees paid"] == 6.0
    assert m["Avg trade return"] == pytest.approx((0.05 - 0.02 + 0.03) / 3)


def test_threshold_tuning_uses_only_validation_predictions():
    px = generate_synthetic_ohlcv(n=500, seed=4)
    rng = np.random.default_rng(1)
    val = pd.Series(rng.normal(0, 0.01, 150), index=px.index[100:250])
    bt, st, grid = tune_thresholds_on_validation(px, val, StrategyConfig())
    assert st <= bt and val.min() <= st and bt <= val.max()
    assert {"buy_quantile", "sell_quantile", "sharpe", "trades"} <= set(grid.columns)
    # prices after the validation window must not matter
    px2 = px.copy()
    px2.iloc[260:, :4] *= 5
    bt2, st2, _ = tune_thresholds_on_validation(px2, val, StrategyConfig())
    assert (bt, st) == (bt2, st2)
