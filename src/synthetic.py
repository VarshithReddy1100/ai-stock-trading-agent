"""Synthetic OHLCV generator for offline demos and unit tests.

IMPORTANT: this data is artificial. It exists so the pipeline can be exercised
without a network connection. It is never used as a fallback when a real download
fails, and any result computed from it says nothing about real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n: int = 1500,
    seed: int = 7,
    start: str = "2018-01-02",
    s0: float = 100.0,
    annual_drift: float = 0.08,
    base_vol: float = 0.20,
    momentum: float = 0.05,
) -> pd.DataFrame:
    """Geometric random walk with volatility clustering and weak return autocorrelation."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n, name="Date")

    daily_drift = annual_drift / 252
    base_daily_vol = base_vol / np.sqrt(252)

    rets = np.zeros(n)
    vol = np.full(n, base_daily_vol)
    for t in range(1, n):
        # GARCH(1,1)-style variance recursion
        var = (
            0.05 * base_daily_vol**2
            + 0.10 * rets[t - 1] ** 2
            + 0.85 * vol[t - 1] ** 2
        )
        vol[t] = np.sqrt(var)
        rets[t] = daily_drift + momentum * rets[t - 1] + vol[t] * rng.standard_normal()

    close = s0 * np.exp(np.cumsum(rets))
    prev_close = np.concatenate([[s0], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0, base_daily_vol * 0.25, n))
    spread_hi = np.abs(rng.normal(0, base_daily_vol * 0.5, n))
    spread_lo = np.abs(rng.normal(0, base_daily_vol * 0.5, n))
    high = np.maximum(open_, close) * (1 + spread_hi)
    low = np.minimum(open_, close) * (1 - spread_lo)
    volume = np.exp(rng.normal(15.0, 0.35, n) + 8.0 * np.abs(rets)).round()

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )
    df.attrs["synthetic"] = True
    return df
