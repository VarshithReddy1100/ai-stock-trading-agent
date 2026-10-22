"""Central configuration for the AI Stock Trading Agent.

All tunable settings live in plain dataclasses so they can be serialised to JSON
(saved next to the model for reproducible inference) and edited from the dashboard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

# Bars per year used to annualise volatility / Sharpe for each Yahoo Finance interval.
# Intraday values assume 6.5 trading hours per US session and are approximations.
PERIODS_PER_YEAR: dict[str, float] = {
    "1d": 252,
    "5d": 52,
    "1wk": 52,
    "1mo": 12,
    "1h": 252 * 6.5,
    "30m": 252 * 13,
    "15m": 252 * 26,
    "5m": 252 * 78,
}


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: Optional[str] = None  # None -> up to the latest available bar
    interval: str = "1d"
    min_rows: int = 300  # refuse to train on less data than this
    cache_dir: str = "data/cache"
    cache_ttl_hours: float = 12.0

    @property
    def periods_per_year(self) -> float:
        return PERIODS_PER_YEAR.get(self.interval, 252)


@dataclass
class ModelConfig:
    """LSTM next-period-return model + chronological split settings."""

    lookback: int = 30  # sequence length (bars of history per sample)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8  # early-stopping patience (epochs without val improvement)
    grad_clip: float = 1.0
    loss: str = "huber"  # "huber" or "mse"
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15  # test fraction is the remainder
    split_gap: int = 1  # samples purged between consecutive splits (label overlap)
    feature_clip: float = 5.0  # scaled features clipped to +/- this many train std devs


@dataclass
class StrategyConfig:
    """Rule-based long-only strategy and execution assumptions."""

    initial_capital: float = 10_000.0
    # How thresholds on the predicted next-bar return are chosen:
    #   "absolute"  -> use buy_threshold / sell_threshold as given
    #   "quantile"  -> quantiles of VALIDATION-period predictions (no test peeking)
    #   "tuned"     -> grid-search the quantile pair on the VALIDATION period only
    threshold_mode: str = "quantile"
    buy_threshold: float = 0.001  # used when threshold_mode == "absolute"
    sell_threshold: float = -0.001
    buy_quantile: float = 0.60
    sell_quantile: float = 0.40
    position_fraction: float = 1.0  # fraction of available cash deployed on a BUY
    fee_rate: float = 0.001  # 0.10% of traded notional, each side
    slippage_bps: float = 5.0  # adverse price move applied to every fill
    fractional_shares: bool = False
    stop_loss_pct: Optional[float] = None  # e.g. 0.08 = exit if close <= entry * 0.92
    risk_free_rate: float = 0.0  # annual, used in the Sharpe ratio


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        def build(klass, values):
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in (values or {}).items() if k in names})

        return cls(
            data=build(DataConfig, d.get("data")),
            model=build(ModelConfig, d.get("model")),
            strategy=build(StrategyConfig, d.get("strategy")),
        )
