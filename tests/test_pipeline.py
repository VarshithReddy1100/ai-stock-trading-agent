import json

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.model import ModelBundle
from src.pipeline import run_pipeline
from train import main as train_main


@pytest.fixture(scope="module")
def result(ohlcv):
    cfg = PipelineConfig()
    cfg.model.lookback, cfg.model.hidden_size, cfg.model.num_layers = 20, 16, 1
    cfg.model.epochs, cfg.model.batch_size, cfg.model.patience = 5, 32, 3
    return run_pipeline(ohlcv, cfg)


def test_pipeline_outputs_present(result):
    assert {"LSTM", "Persistence (price)", "Momentum (last return)", "Ridge regression"} == set(result.comparison.index)
    assert len(result.prediction_frame) == len(result.prepared.splits["test"])
    assert set(result.backtest.summary_table().index) >= {"Total return", "Sharpe ratio", "Max drawdown", "Win rate"}
    assert isinstance(result.verdict, str) and result.verdict


def test_backtest_runs_only_on_the_test_period(result):
    test_dates = result.prepared.split_dates("test")
    assert result.backtest.equity.index[0] == test_dates[0]
    assert result.backtest.signals.index[0] == test_dates[0]
    assert result.backtest.signals.index[-1] == test_dates[-1]
    train_end = result.prepared.split_dates("train")[-1]
    assert result.backtest.equity.index[0] > train_end


def test_thresholds_come_from_validation_not_test(result):
    val = result.predictions.iloc[result.prepared.splits["val"]]
    ti = result.threshold_info
    assert ti["mode"] == "quantile"
    assert ti["buy_threshold"] == pytest.approx(np.quantile(val, 0.60))
    assert ti["sell_threshold"] == pytest.approx(np.quantile(val, 0.40))


def test_pipeline_is_reproducible(ohlcv, result):
    cfg = PipelineConfig.from_dict(result.config.to_dict())
    again = run_pipeline(ohlcv, cfg)
    pd.testing.assert_series_equal(result.predictions, again.predictions, atol=1e-6)
    assert result.backtest.metrics["Final equity"] == pytest.approx(again.backtest.metrics["Final equity"])


@pytest.mark.parametrize("mode", ["absolute", "tuned"])
def test_other_threshold_modes(ohlcv, mode):
    cfg = PipelineConfig()
    cfg.model.lookback, cfg.model.hidden_size, cfg.model.num_layers = 20, 8, 1
    cfg.model.epochs = 3
    cfg.strategy.threshold_mode = mode
    r = run_pipeline(ohlcv, cfg)
    assert r.threshold_info["mode"] == mode
    if mode == "tuned":
        assert r.threshold_grid is not None and len(r.threshold_grid) > 0


def test_unknown_threshold_mode(ohlcv):
    cfg = PipelineConfig()
    cfg.model.lookback, cfg.model.epochs = 20, 2
    cfg.strategy.threshold_mode = "magic"
    with pytest.raises(ValueError):
        run_pipeline(ohlcv, cfg)


def test_save_report_and_reload(result, tmp_path, ohlcv):
    from src.pipeline import save_report

    save_report(result, tmp_path)
    for name in ("model.pt", "meta.json", "config.json", "report.json", "equity_curve.csv",
                 "benchmark_curve.csv", "trades.csv", "test_predictions.csv"):
        assert (tmp_path / name).exists(), name
    report = json.loads((tmp_path / "report.json").read_text())
    assert "strategy_metrics" in report and "buy_and_hold_metrics" in report
    cfg = PipelineConfig.from_dict(json.loads((tmp_path / "config.json").read_text()))
    assert cfg.model.lookback == 20
    reloaded = ModelBundle.load(tmp_path).predict_returns(ohlcv)
    pd.testing.assert_series_equal(reloaded.loc[result.predictions.index], result.predictions,
                                   check_names=False, atol=1e-6)


def test_cli_synthetic_run(tmp_path, capsys):
    code = train_main(["--synthetic", "--epochs", "3", "--lookback", "20", "--hidden-size", "8",
                       "--out", str(tmp_path / "run"), "--no-eda"])
    assert code == 0
    assert (tmp_path / "run" / "report.json").exists()
    assert "Educational research only" in capsys.readouterr().out


def test_cli_reports_data_errors(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("Date,Open\n2020-01-01,1\n")
    code = train_main(["--csv", str(bad), "--out", str(tmp_path / "x")])
    assert code == 1 and "ERROR" in capsys.readouterr().err
