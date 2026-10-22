"""Rule-based trading agent and portfolio / cash management.

The agent is deliberately transparent: a long-only state machine driven by the
model's predicted next-bar return.

Execution timing (no look-ahead)
--------------------------------
A decision made at the CLOSE of bar ``t`` uses only data up to ``t``. The order is
filled at the OPEN of bar ``t+1`` (plus slippage and fees). See ``backtesting``.

This is a research/education simulator. It never places real orders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .config import StrategyConfig


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def calibrate_thresholds(
    reference_predictions: pd.Series | np.ndarray, buy_quantile: float, sell_quantile: float
) -> tuple[float, float]:
    """Thresholds as quantiles of *reference* predictions.

    Callers must pass training/validation-period predictions only -- never the test
    period -- so the thresholds are not tuned on the data used for the final evaluation.
    """
    ref = np.asarray(reference_predictions, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        raise ValueError("No reference predictions available to calibrate thresholds.")
    if not 0.0 <= sell_quantile <= buy_quantile <= 1.0:
        raise ValueError("Require 0 <= sell_quantile <= buy_quantile <= 1.")
    return float(np.quantile(ref, buy_quantile)), float(np.quantile(ref, sell_quantile))


class TradingAgent:
    """Turns a predicted next-bar return into BUY / SELL / HOLD.

    Rules (long-only)
    * Flat and prediction > ``buy_threshold``   -> BUY
    * Long and prediction < ``sell_threshold``  -> SELL (exit)
    * Long and close <= entry * (1 - stop_loss) -> SELL (stop-loss, evaluated at the close)
    * Otherwise HOLD
    """

    def __init__(self, config: StrategyConfig, buy_threshold: float, sell_threshold: float):
        if sell_threshold > buy_threshold:
            raise ValueError("sell_threshold must be <= buy_threshold.")
        self.config = config
        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)

    def decide(
        self,
        predicted_return: float,
        in_position: bool,
        close: float,
        entry_price: Optional[float] = None,
    ) -> tuple[Signal, str]:
        sl = self.config.stop_loss_pct
        if in_position and sl and entry_price and close <= entry_price * (1.0 - sl):
            return Signal.SELL, "stop_loss"
        if not in_position and predicted_return > self.buy_threshold:
            return Signal.BUY, "pred_above_buy_threshold"
        if in_position and predicted_return < self.sell_threshold:
            return Signal.SELL, "pred_below_sell_threshold"
        return Signal.HOLD, ""


class Portfolio:
    """Cash, one long position, fees, slippage and trade log.

    All fills are validated so cash can never go negative.
    """

    def __init__(self, config: StrategyConfig, initial_cash: Optional[float] = None):
        self.config = config
        self.cash = float(config.initial_capital if initial_cash is None else initial_cash)
        if self.cash <= 0:
            raise ValueError("Initial capital must be positive.")
        self.shares = 0.0
        self.entry_price: Optional[float] = None  # average fill price of the open position
        self.cost_basis = 0.0  # cash spent on the open position incl. fees
        self.trades: list[dict] = []
        self.skipped_orders = 0

    @property
    def slippage(self) -> float:
        return self.config.slippage_bps / 1e4

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def buy(self, date, open_price: float, reason: str = "", fraction: Optional[float] = None) -> bool:
        """Spend up to ``fraction`` of cash (default ``position_fraction``). Returns True if filled."""
        if self.shares > 0:
            return False
        frac = self.config.position_fraction if fraction is None else fraction
        budget = self.cash * min(max(frac, 0.0), 1.0)
        fill = open_price * (1.0 + self.slippage)
        unit_cost = fill * (1.0 + self.config.fee_rate)  # price of one share incl. fee
        qty = budget / unit_cost
        if not self.config.fractional_shares:
            qty = math.floor(qty + 1e-12)
        if qty <= 0:
            self.skipped_orders += 1  # cannot afford even one share
            return False
        notional = qty * fill
        fee = notional * self.config.fee_rate
        total = notional + fee
        if total > self.cash + 1e-9:  # defensive: should be impossible by construction
            self.skipped_orders += 1
            return False
        self.cash -= total
        self.shares = float(qty)
        self.entry_price = fill
        self.cost_basis = total
        self.trades.append(
            dict(date=date, side="BUY", shares=qty, price=fill, notional=notional, fee=fee,
                 cash_after=self.cash, reason=reason, pnl=np.nan, return_pct=np.nan)
        )
        return True

    def sell_all(self, date, open_price: float, reason: str = "") -> bool:
        if self.shares <= 0:
            return False
        fill = open_price * (1.0 - self.slippage)
        notional = self.shares * fill
        fee = notional * self.config.fee_rate
        proceeds = notional - fee
        pnl = proceeds - self.cost_basis
        self.cash += proceeds
        self.trades.append(
            dict(date=date, side="SELL", shares=self.shares, price=fill, notional=notional, fee=fee,
                 cash_after=self.cash, reason=reason, pnl=pnl, return_pct=pnl / self.cost_basis)
        )
        self.shares = 0.0
        self.entry_price = None
        self.cost_basis = 0.0
        return True

    def trades_frame(self) -> pd.DataFrame:
        cols = ["date", "side", "shares", "price", "notional", "fee", "cash_after", "reason", "pnl", "return_pct"]
        return pd.DataFrame(self.trades, columns=cols)
