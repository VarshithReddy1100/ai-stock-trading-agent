import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ModelConfig
from src.model import (
    LSTMRegressor,
    ModelBundle,
    TrainingError,
    baseline_predictions,
    compare_models,
    predict_scaled,
    regression_metrics,
    train_model,
)
from src.preprocessing import prepare_dataset


@pytest.fixture(scope="module")
def trained(ohlcv):
    cfg = ModelConfig(lookback=20, hidden_size=16, num_layers=1, epochs=4, batch_size=32, patience=3)
    data = prepare_dataset(ohlcv, cfg)
    model, history = train_model(data, cfg)
    return data, cfg, model, history


def test_model_io_shapes():
    m = LSTMRegressor(n_features=7, hidden_size=8, num_layers=2, dropout=0.1)
    out = m(torch.zeros(5, 12, 7))
    assert out.shape == (5,)


def test_training_records_history(trained):
    _, cfg, _, h = trained
    assert len(h["train_loss"]) == len(h["val_loss"]) == h["stopped_epoch"] <= cfg.epochs
    assert 1 <= h["best_epoch"] <= h["stopped_epoch"]
    assert all(np.isfinite(h["train_loss"])) and all(np.isfinite(h["val_loss"]))


def test_early_stopping_triggers(ohlcv):
    cfg = ModelConfig(lookback=20, hidden_size=16, num_layers=1, epochs=60, batch_size=32, patience=1,
                      learning_rate=0.01)
    data = prepare_dataset(ohlcv, cfg)
    _, h = train_model(data, cfg)
    assert h["stopped_epoch"] < 60


def test_progress_callback_called(ohlcv):
    cfg = ModelConfig(lookback=20, hidden_size=8, num_layers=1, epochs=2, batch_size=32)
    data = prepare_dataset(ohlcv, cfg)
    calls = []
    train_model(data, cfg, progress=lambda e, t, tr, va: calls.append((e, t)))
    assert calls == [(1, 2), (2, 2)]


def test_training_is_reproducible_with_seed(ohlcv):
    cfg = ModelConfig(lookback=20, hidden_size=16, num_layers=1, epochs=3, batch_size=32, seed=5)
    data = prepare_dataset(ohlcv, cfg)
    m1, _ = train_model(data, cfg)
    m2, _ = train_model(data, cfg)
    np.testing.assert_allclose(predict_scaled(m1, data.X), predict_scaled(m2, data.X), atol=1e-6)


def test_different_seeds_differ(ohlcv):
    cfg1 = ModelConfig(lookback=20, hidden_size=16, num_layers=1, epochs=3, batch_size=32, seed=1)
    cfg2 = ModelConfig(lookback=20, hidden_size=16, num_layers=1, epochs=3, batch_size=32, seed=2)
    data = prepare_dataset(ohlcv, cfg1)
    p1 = predict_scaled(train_model(data, cfg1)[0], data.X)
    p2 = predict_scaled(train_model(data, cfg2)[0], data.X)
    assert not np.allclose(p1, p2)


def test_save_load_roundtrip(trained, ohlcv, tmp_path):
    data, cfg, model, h = trained
    bundle = ModelBundle(model, data.scaler, data.target_scaler, data.feature_names, cfg, h)
    before = bundle.predict_returns(ohlcv)
    bundle.save(tmp_path / "m")
    after = ModelBundle.load(tmp_path / "m").predict_returns(ohlcv)
    pd.testing.assert_series_equal(before, after, atol=1e-7)
    assert (tmp_path / "m" / "model.pt").exists() and (tmp_path / "m" / "meta.json").exists()


def test_load_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelBundle.load(tmp_path / "nope")


def test_inference_path_matches_training_path(trained, ohlcv):
    """Dashboard/inference predictions must equal the pipeline's predictions on the same dates."""
    data, cfg, model, h = trained
    bundle = ModelBundle(model, data.scaler, data.target_scaler, data.feature_names, cfg, h)
    served = bundle.predict_returns(ohlcv)
    direct = pd.Series(
        data.target_scaler.inverse_transform(predict_scaled(model, data.X)).reshape(-1), index=data.dates
    )
    pd.testing.assert_series_equal(served.loc[direct.index], direct, check_names=False, atol=1e-6)


def test_inference_needs_enough_history(trained, ohlcv):
    data, cfg, model, h = trained
    bundle = ModelBundle(model, data.scaler, data.target_scaler, data.feature_names, cfg, h)
    from src.data_collection import InsufficientDataError

    with pytest.raises(InsufficientDataError):
        bundle.predict_returns(ohlcv.iloc[:60])


def test_prediction_uses_no_future_information(trained, ohlcv):
    data, cfg, model, h = trained
    bundle = ModelBundle(model, data.scaler, data.target_scaler, data.feature_names, cfg, h)
    full = bundle.predict_returns(ohlcv)
    trunc = bundle.predict_returns(ohlcv.iloc[:600])
    pd.testing.assert_series_equal(full.loc[trunc.index], trunc, atol=1e-6)


def test_regression_metrics_known_values():
    y = np.array([0.01, -0.02, 0.03, 0.0])
    p = np.array([0.02, 0.01, 0.03, 0.5])
    m = regression_metrics(y, p, close=np.array([100.0, 100, 100, 100]))
    assert m["MAE"] == pytest.approx((0.01 + 0.03 + 0 + 0.5) / 4)
    assert m["RMSE"] == pytest.approx(np.sqrt((0.01**2 + 0.03**2 + 0 + 0.5**2) / 4))
    assert m["Directional accuracy"] == pytest.approx(2 / 3)  # zero-return sample ignored
    assert m["Next-close MAE ($)"] == pytest.approx(100 * m["MAE"])


def test_zero_predictions_have_undefined_directional_accuracy():
    m = regression_metrics(np.array([0.01, -0.02, 0.03]), np.zeros(3))
    assert np.isnan(m["Directional accuracy"])  # no direction called, so not "0% accurate"


def test_baselines_and_comparison(trained):
    data, *_ = trained
    base = baseline_predictions(data)
    assert set(base) == {"Persistence (price)", "Momentum (last return)", "Ridge regression"}
    assert all(len(v) == len(data.y) for v in base.values())
    assert (base["Persistence (price)"] == 0).all()
    noisy = np.random.default_rng(0).normal(0, 0.005, len(data.y))  # a non-degenerate "LSTM"
    table, verdict = compare_models(data, noisy)
    assert list(table.index)[0] == "LSTM" and len(table) == 4
    assert table.loc["LSTM", "Directional accuracy"] == table.loc["LSTM", "Directional accuracy"]  # not NaN
    assert np.isnan(table.loc["Persistence (price)", "Directional accuracy"])
    assert "LSTM" in verdict and ("coin flip" in verdict or "standard errors" in verdict)


def test_verdict_does_not_overclaim():
    """When the LSTM is worse than a baseline the verdict must say so."""
    from src.preprocessing import FeatureScaler, PreparedData

    n = 400
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, n)
    splits = {"train": np.arange(0, 200), "val": np.arange(201, 250), "test": np.arange(251, 400)}
    data = PreparedData([], 1, FeatureScaler(), FeatureScaler(), pd.date_range("2020", periods=n),
                        rng.normal(size=(n, 1, 3)).astype(np.float32), y, y.astype(np.float32),
                        np.full(n, 100.0), np.zeros(n), splits)
    bad_lstm = rng.normal(0, 0.05, n)  # far noisier than predicting zero
    _, verdict = compare_models(data, bad_lstm)
    assert "did NOT beat" in verdict
