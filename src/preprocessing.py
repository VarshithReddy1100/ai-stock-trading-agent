"""Feature engineering, chronological splitting, scaling and sequence building.

Leakage rules enforced here
---------------------------
* Every feature at row ``t`` uses only data up to and including bar ``t``
  (rolling / EWM windows are causal; nothing is centred or shifted backwards).
* The label at row ``t`` is the return from close ``t`` to close ``t+1`` and is
  never part of the features.
* Splits are chronological with a purge gap, because the label of the last
  training sample overlaps in time with the first validation bar.
* Scalers are fit on rows that end at or before the last *training* sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data_collection import InsufficientDataError

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal technical features from an OHLCV frame.

    Returned columns are scale-free (ratios, returns, bounded oscillators) so a
    single model can be trained on tickers with very different price levels.
    Early rows contain NaN while rolling windows warm up.
    """
    close, open_, high, low, volume = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    # Returns and lagged returns
    ret = close.pct_change()
    out["ret_1"] = ret  # (log return is omitted: EDA shows it is ~100% correlated with ret_1)
    for k in (1, 2, 3, 5):
        out[f"ret_lag_{k}"] = ret.shift(k)

    # Lagged prices, expressed relative to today's close (stationary)
    for k in (5, 10, 20):
        out[f"close_over_lag_{k}"] = close / close.shift(k) - 1

    # Simple and exponential moving averages (as distance from price)
    for w in (5, 10, 20, 50):
        out[f"sma_ratio_{w}"] = close / close.rolling(w).mean() - 1
    for span in (12, 26):
        out[f"ema_ratio_{span}"] = close / close.ewm(span=span, adjust=False).mean() - 1

    # Rolling volatility of returns
    for w in (5, 10, 20):
        out[f"volatility_{w}"] = ret.rolling(w).std()

    # RSI (Wilder smoothing), rescaled to [-1, 1]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    out["rsi_14"] = (rsi - 50.0) / 50.0

    # MACD (12, 26, 9), normalised by price
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / close
    out["macd_signal"] = macd_signal / close
    out["macd_hist"] = (macd - macd_signal) / close

    # Intrabar structure
    out["hl_range"] = (high - low) / close
    out["oc_gap"] = open_ / close.shift(1) - 1  # overnight gap
    out["close_open"] = close / open_ - 1

    # Volume: today's volume relative to its 20-bar average (log ratio)
    out["volume_ratio_20"] = np.log1p(volume) - np.log1p(volume.rolling(20).mean())

    return out.replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward simple return from close ``t`` to close ``t+horizon`` (label for row ``t``).

    Why next-period *return* rather than next price: returns are roughly stationary
    and scale-free, whereas price levels trend and make test-set error look
    deceptively good for a model that merely copies yesterday's close.
    """
    return close.shift(-horizon) / close - 1.0


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
class FeatureScaler:
    """Z-score scaler with optional clipping. Fit on training data only."""

    def __init__(self, clip: Optional[float] = 5.0):
        self.clip = clip
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.mean_ = x.mean(axis=0)
        self.std_ = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been fit.")
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        z = (x.reshape(-1, len(self.mean_)) - self.mean_) / self.std_
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z.reshape(shape).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been fit.")
        z = np.asarray(z, dtype=np.float64)
        shape = z.shape
        return (z.reshape(-1, len(self.mean_)) * self.std_ + self.mean_).reshape(shape)

    def to_dict(self) -> dict:
        return {"clip": self.clip, "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        s = cls(clip=d.get("clip"))
        s.mean_ = np.asarray(d["mean"], dtype=np.float64)
        s.std_ = np.asarray(d["std"], dtype=np.float64)
        return s


# --------------------------------------------------------------------------- #
# Splitting & sequences
# --------------------------------------------------------------------------- #
def chronological_split(
    n_samples: int, train_frac: float, val_frac: float, gap: int = 1
) -> dict[str, np.ndarray]:
    """Index arrays for train / val / test in time order, with ``gap`` samples purged
    between consecutive splits. Never shuffles."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to < 1.")
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    val_start = n_train + gap
    test_start = val_start + n_val + gap
    splits = {
        "train": np.arange(0, n_train),
        "val": np.arange(val_start, val_start + n_val),
        "test": np.arange(test_start, n_samples),
    }
    for name, idx in splits.items():
        if len(idx) < 10:
            raise InsufficientDataError(
                f"Only {len(idx)} samples in the {name} split. Use more history or a shorter lookback."
            )
    return splits


def make_sequences(x: np.ndarray, lookback: int) -> np.ndarray:
    """Sliding windows of shape (n - lookback + 1, lookback, n_features).

    Window ``k`` covers rows ``k .. k+lookback-1`` and is aligned with the label at row
    ``k + lookback - 1`` (its last bar) -- it contains no later rows.
    """
    if len(x) < lookback:
        raise InsufficientDataError(f"Need at least {lookback} rows to build one sequence, have {len(x)}.")
    windows = np.lib.stride_tricks.sliding_window_view(x, lookback, axis=0)  # (M, F, L)
    return np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32)


@dataclass
class PreparedData:
    """Everything the model, baselines and backtester need, aligned by sample."""

    feature_names: list[str]
    lookback: int
    scaler: FeatureScaler
    target_scaler: FeatureScaler
    dates: pd.DatetimeIndex  # decision date of each sample (end of its window)
    X: np.ndarray  # (M, lookback, n_features) scaled
    y: np.ndarray  # (M,) next-bar simple return (unscaled)
    y_scaled: np.ndarray  # (M,) standardised with train statistics
    close: np.ndarray  # (M,) close at the decision date
    raw_ret_1: np.ndarray  # (M,) unscaled return of the decision bar (for momentum baseline)
    splits: dict[str, np.ndarray]

    def split_dates(self, name: str) -> pd.DatetimeIndex:
        return self.dates[self.splits[name]]


def prepare_dataset(df: pd.DataFrame, cfg: ModelConfig) -> PreparedData:
    """Build scaled sequences and chronological splits from a clean OHLCV frame."""
    feats = build_features(df)
    target = make_target(df["Close"])
    frame = feats.join(target.rename("__target__")).dropna()
    if len(frame) < cfg.lookback + 30:
        raise InsufficientDataError(
            f"Only {len(frame)} usable rows after feature warm-up; need at least {cfg.lookback + 30}."
        )

    feature_names = list(feats.columns)
    x_raw = frame[feature_names].to_numpy(dtype=np.float64)
    y = frame["__target__"].to_numpy(dtype=np.float64)
    close = df["Close"].reindex(frame.index).to_numpy(dtype=np.float64)

    n_samples = len(frame) - cfg.lookback + 1
    splits = chronological_split(n_samples, cfg.train_frac, cfg.val_frac, cfg.split_gap)

    # Scalers see only rows that end at or before the last training sample.
    last_train_row = int(splits["train"][-1]) + cfg.lookback - 1
    scaler = FeatureScaler(clip=cfg.feature_clip).fit(x_raw[: last_train_row + 1])
    y_train = y[cfg.lookback - 1 : last_train_row + 1]
    target_scaler = FeatureScaler(clip=None).fit(y_train)

    x_scaled = scaler.transform(x_raw)
    sample_rows = slice(cfg.lookback - 1, None)
    return PreparedData(
        feature_names=feature_names,
        lookback=cfg.lookback,
        scaler=scaler,
        target_scaler=target_scaler,
        dates=frame.index[sample_rows],
        X=make_sequences(x_scaled, cfg.lookback),
        y=y[sample_rows],
        y_scaled=target_scaler.transform(y[sample_rows]).astype(np.float32),
        close=close[sample_rows],
        raw_ret_1=frame["ret_1"].to_numpy(dtype=np.float64)[sample_rows],
        splits=splits,
    )
