from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from src.eda import describe_findings, feature_correlation, missing_report, run_eda, summary_stats
from src.visualization import equity_chart, drawdown_chart, prediction_chart, price_with_signals, training_chart

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_eda_outputs(ohlcv, tmp_path):
    out = run_eda(ohlcv, tmp_path)
    assert {"price_ma", "returns", "volume", "volatility", "correlation"} == set(out["figures"])
    assert len(out["findings"]) >= 4 and all(isinstance(f, str) for f in out["findings"])
    assert (tmp_path / "eda_price_ma.png").exists() and (tmp_path / "eda_findings.txt").exists()
    assert "Daily return" in summary_stats(ohlcv).index
    assert (missing_report(ohlcv)["missing"] == 0).all()
    corr = feature_correlation(ohlcv)
    assert corr.shape[0] == corr.shape[1] and "target_next_return" in corr.columns
    plt.close("all")


def test_findings_are_computed_from_data(ohlcv):
    text = " ".join(describe_findings(ohlcv))
    assert str(ohlcv.index[0].date()) in text and str(len(ohlcv)) in text


def test_charts_build(ohlcv, tiny_pipeline_cfg):
    from src.pipeline import run_pipeline

    r = run_pipeline(ohlcv, tiny_pipeline_cfg)
    bt = r.backtest
    for fig in (
        equity_chart(bt), drawdown_chart(bt), prediction_chart(r.prediction_frame),
        training_chart(r.history), price_with_signals(ohlcv, bt.trades, start=bt.equity.index[0]),
    ):
        assert len(fig.data) >= 1


def test_dashboard_runs_end_to_end_on_synthetic_data():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Run pipeline" in b.label for b in at.button)  # landing state shows the run button

    next(r for r in at.radio if r.label == "Source").set_value("Synthetic demo (artificial)")
    next(s for s in at.slider if s.label == "Max epochs").set_value(5)
    next(s for s in at.slider if s.label == "Lookback window (bars)").set_value(20)
    at.run()
    next(b for b in at.button if "Run pipeline" in b.label).click()
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.metric) == 4
    assert any("ARTIFICIAL" in w.value for w in at.warning)
    assert len(at.tabs) == 5


def test_dashboard_shows_clean_error_for_missing_csv():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    next(r for r in at.radio if r.label == "Source").set_value("Upload CSV")
    at.run()
    next(b for b in at.button if "Run pipeline" in b.label).click()
    at.run()
    assert not at.exception
    assert any("upload a CSV" in e.value for e in at.error)
