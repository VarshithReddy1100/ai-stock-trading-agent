import numpy as np
import pytest

from src.config import StrategyConfig
from src.trading_agent import Portfolio, Signal, TradingAgent, calibrate_thresholds


def agent(**kw):
    return TradingAgent(StrategyConfig(**kw), buy_threshold=0.01, sell_threshold=-0.01)


def test_calibrate_thresholds_quantiles():
    preds = np.linspace(-1, 1, 101)
    buy, sell = calibrate_thresholds(preds, 0.75, 0.25)
    assert buy == pytest.approx(0.5) and sell == pytest.approx(-0.5)


def test_calibrate_thresholds_validation():
    with pytest.raises(ValueError):
        calibrate_thresholds([0.1, 0.2], 0.3, 0.7)
    with pytest.raises(ValueError):
        calibrate_thresholds([], 0.6, 0.4)


def test_agent_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        TradingAgent(StrategyConfig(), 0.0, 0.1)


@pytest.mark.parametrize(
    "pred,in_pos,expected",
    [
        (0.02, False, Signal.BUY),
        (0.005, False, Signal.HOLD),
        (-0.02, False, Signal.HOLD),  # long-only: nothing to sell
        (-0.02, True, Signal.SELL),
        (0.02, True, Signal.HOLD),  # already long
        (0.0, True, Signal.HOLD),
    ],
)
def test_decision_table(pred, in_pos, expected):
    sig, _ = agent().decide(pred, in_pos, close=100.0, entry_price=100.0)
    assert sig is expected


def test_stop_loss_overrides_bullish_prediction():
    a = agent(stop_loss_pct=0.1)
    sig, reason = a.decide(0.05, True, close=89.0, entry_price=100.0)
    assert sig is Signal.SELL and reason == "stop_loss"
    sig, _ = a.decide(0.05, True, close=91.0, entry_price=100.0)
    assert sig is Signal.HOLD


def test_portfolio_buy_math_with_costs():
    cfg = StrategyConfig(initial_capital=10_000, fee_rate=0.01, slippage_bps=100, fractional_shares=True)
    pf = Portfolio(cfg)
    assert pf.buy("d1", 50.0)
    fill = 50 * 1.01
    qty = 10_000 / (fill * 1.01)
    assert pf.shares == pytest.approx(qty)
    assert pf.trades[0]["price"] == pytest.approx(fill)
    assert pf.trades[0]["fee"] == pytest.approx(qty * fill * 0.01)
    assert pf.cash == pytest.approx(0.0, abs=1e-6)  # spent everything, fee included, never more


def test_integer_shares_and_cash_never_negative():
    cfg = StrategyConfig(initial_capital=1_000, fee_rate=0.002, slippage_bps=10)
    pf = Portfolio(cfg)
    assert pf.buy("d", 33.33)
    assert pf.shares == int(pf.shares) and pf.cash >= 0


def test_cannot_afford_single_share_is_skipped():
    pf = Portfolio(StrategyConfig(initial_capital=100))
    assert pf.buy("d", 500.0) is False
    assert pf.shares == 0 and pf.cash == 100 and pf.skipped_orders == 1 and not pf.trades


def test_position_fraction_limits_spend():
    pf = Portfolio(StrategyConfig(initial_capital=10_000, position_fraction=0.5, fee_rate=0, slippage_bps=0,
                                  fractional_shares=True))
    pf.buy("d", 100.0)
    assert pf.cash == pytest.approx(5_000) and pf.shares == pytest.approx(50)


def test_cannot_buy_twice_or_sell_when_flat():
    pf = Portfolio(StrategyConfig(initial_capital=10_000))
    assert pf.sell_all("d", 10.0) is False
    assert pf.buy("d", 10.0) is True
    assert pf.buy("d2", 10.0) is False


def test_round_trip_pnl_includes_all_costs():
    cfg = StrategyConfig(initial_capital=10_000, fee_rate=0.001, slippage_bps=10, fractional_shares=True)
    pf = Portfolio(cfg)
    pf.buy("b", 100.0)
    pf.sell_all("s", 110.0)
    sell = pf.trades[-1]
    assert sell["pnl"] == pytest.approx(pf.cash - 10_000)
    assert sell["return_pct"] == pytest.approx(sell["pnl"] / 10_000)
    assert pf.shares == 0 and pf.entry_price is None


def test_losing_trade_with_costs_when_price_flat():
    pf = Portfolio(StrategyConfig(initial_capital=10_000, fee_rate=0.001, slippage_bps=5, fractional_shares=True))
    pf.buy("b", 100.0)
    pf.sell_all("s", 100.0)
    assert pf.trades[-1]["pnl"] < 0  # frictions make a flat round trip a loss


def test_nonpositive_capital_rejected():
    with pytest.raises(ValueError):
        Portfolio(StrategyConfig(initial_capital=0))
