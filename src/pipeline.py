"""End-to-end pipeline: data -> features -> LSTM -> signals -> backtest -> report."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .backtesting import BacktestResult, run_backtest, tune_thresholds_on_validation
from .config import PipelineConfig
from .model import (
    ModelBundle,
    ProgressFn,
    compare_models,
    predict_scaled,
    train_model,
)
from .preprocessing import PreparedData, prepare_dataset
from .trading_agent import calibrate_thresholds

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    config: PipelineConfig
    prepared: PreparedData
    bundle: ModelBundle
    history: dict
    predictions: pd.Series  # LSTM next-bar-return forecast for every sample
    comparison: pd.DataFrame  # LSTM vs baselines on the test split
    verdict: str
    threshold_info: dict
    backtest: BacktestResult
    prediction_frame: pd.DataFrame  # test split: actual vs predicted returns / closes
    split_summary: pd.DataFrame
    threshold_grid: Optional[pd.DataFrame] = None


def resolve_thresholds(
    cfg: PipelineConfig,
    prices: pd.DataFrame,
    val_pred: pd.Series,
) -> tuple[float, float, dict, Optional[pd.DataFrame]]:
    """Choose (buy, sell) thresholds using validation-period information only."""
    s = cfg.strategy
    ppy = cfg.data.periods_per_year
    grid = None
    if s.threshold_mode == "absolute":
        bt, st, how = s.buy_threshold, s.sell_threshold, "absolute (user supplied)"
    elif s.threshold_mode == "tuned":
        bt, st, grid = tune_thresholds_on_validation(prices, val_pred, s, ppy)
        how = "grid-searched on validation period (Sharpe)"
    elif s.threshold_mode == "quantile":
        bt, st = calibrate_thresholds(val_pred, s.buy_quantile, s.sell_quantile)
        how = f"validation-prediction quantiles ({s.buy_quantile:.2f} / {s.sell_quantile:.2f})"
    else:
        raise ValueError(f"Unknown threshold_mode '{s.threshold_mode}'.")
    return bt, st, {"mode": s.threshold_mode, "method": how, "buy_threshold": bt, "sell_threshold": st}, grid


def run_pipeline(
    df: pd.DataFrame,
    cfg: Optional[PipelineConfig] = None,
    save_dir: Optional[str | Path] = None,
    progress: Optional[ProgressFn] = None,
) -> PipelineResult:
    """Train on the chronological train split, tune/choose thresholds on validation,
    and report all headline numbers on the untouched test split."""
    cfg = cfg or PipelineConfig()
    data = prepare_dataset(df, cfg.model)
    model, history = train_model(data, cfg.model, progress)
    bundle = ModelBundle(model, data.scaler, data.target_scaler, data.feature_names, cfg.model, history)

    pred_all = data.target_scaler.inverse_transform(predict_scaled(model, data.X)).reshape(-1)
    predictions = pd.Series(pred_all, index=data.dates, name="pred_return")

    comparison, verdict = compare_models(data, pred_all, "test")

    val_pred = predictions.iloc[data.splits["val"]]
    test_pred = predictions.iloc[data.splits["test"]]
    buy_th, sell_th, th_info, grid = resolve_thresholds(cfg, df, val_pred)

    backtest = run_backtest(df, test_pred, cfg.strategy, buy_th, sell_th, cfg.data.periods_per_year)

    t = data.splits["test"]
    pred_frame = pd.DataFrame(
        {
            "close": data.close[t],
            "actual_return": data.y[t],
            "pred_return": pred_all[t],
            "actual_next_close": data.close[t] * (1 + data.y[t]),
            "pred_next_close": data.close[t] * (1 + pred_all[t]),
        },
        index=data.dates[t],
    )
    split_summary = pd.DataFrame(
        {
            name: {
                "samples": len(idx),
                "start": str(data.dates[idx[0]].date()),
                "end": str(data.dates[idx[-1]].date()),
            }
            for name, idx in data.splits.items()
        }
    ).T

    result = PipelineResult(
        config=cfg,
        prepared=data,
        bundle=bundle,
        history=history,
        predictions=predictions,
        comparison=comparison,
        verdict=verdict,
        threshold_info=th_info,
        backtest=backtest,
        prediction_frame=pred_frame,
        split_summary=split_summary,
        threshold_grid=grid,
    )
    if save_dir is not None:
        save_report(result, save_dir)
    return result


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def save_report(result: PipelineResult, directory: str | Path) -> Path:
    """Persist model artifacts and evaluation outputs (all reproducible from ``config.json``)."""
    d = Path(directory)
    result.bundle.save(d)
    bt = result.backtest
    (d / "config.json").write_text(json.dumps(result.config.to_dict(), indent=2))
    report = {
        "split_summary": result.split_summary.to_dict(orient="index"),
        "model_comparison_test": result.comparison.to_dict(orient="index"),
        "verdict": result.verdict,
        "thresholds": result.threshold_info,
        "strategy_metrics": bt.metrics,
        "buy_and_hold_metrics": bt.benchmark_metrics,
    }
    (d / "report.json").write_text(json.dumps(report, indent=2, default=_jsonable))
    bt.equity.to_csv(d / "equity_curve.csv")
    bt.benchmark.to_csv(d / "benchmark_curve.csv")
    bt.trades.to_csv(d / "trades.csv", index=False)
    result.prediction_frame.to_csv(d / "test_predictions.csv")
    return d
