import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig
from src.data_collection import InsufficientDataError
from src.preprocessing import (
    FeatureScaler,
    build_features,
    chronological_split,
    make_sequences,
    make_target,
    prepare_dataset,
)
from src.synthetic import generate_synthetic_ohlcv


def test_features_have_no_lookahead_truncation(ohlcv):
    """Features computed on a truncated history must equal the same rows of the full history."""
    full = build_features(ohlcv)
    for k in (120, 400, 700):
        part = build_features(ohlcv.iloc[:k])
        pd.testing.assert_frame_equal(part, full.iloc[:k], check_exact=False, atol=1e-10)


def test_features_unchanged_when_future_prices_change(ohlcv):
    altered = ohlcv.copy()
    altered.iloc[-100:, :4] *= 3.0
    altered.iloc[-100:, 4] *= 5.0
    a, b = build_features(ohlcv), build_features(altered)
    pd.testing.assert_frame_equal(a.iloc[:-100], b.iloc[:-100], check_exact=False, atol=1e-10)


def test_expected_feature_families_present(ohlcv):
    cols = set(build_features(ohlcv).columns)
    for needle in ("ret_1", "ret_lag_1", "close_over_lag_5", "sma_ratio_20", "ema_ratio_12",
                   "volatility_10", "rsi_14", "macd", "macd_hist", "volume_ratio_20"):
        assert needle in cols


def test_rsi_bounds_and_flat_series():
    df = generate_synthetic_ohlcv(n=200, seed=1)
    rsi = build_features(df)["rsi_14"].dropna()
    assert rsi.between(-1, 1).all()
    flat = df.copy()
    flat[["Open", "High", "Low", "Close"]] = 100.0
    assert (build_features(flat)["rsi_14"].dropna() == 0.0).all()  # no gains or losses -> neutral


def test_target_alignment(ohlcv):
    close = ohlcv["Close"]
    t = make_target(close)
    i = 123
    assert t.iloc[i] == pytest.approx(close.iloc[i + 1] / close.iloc[i] - 1)
    assert np.isnan(t.iloc[-1])


def test_chronological_split_is_ordered_disjoint_and_purged():
    s = chronological_split(1000, 0.7, 0.15, gap=2)
    assert s["train"][-1] + 2 < s["val"][0] <= s["val"][-1] < s["test"][0] - 1
    assert set(s["train"]).isdisjoint(s["val"]) and set(s["val"]).isdisjoint(s["test"])
    assert s["test"][-1] == 999
    for idx in s.values():
        assert (np.diff(idx) == 1).all()  # contiguous, never shuffled


def test_split_too_small_raises():
    with pytest.raises(InsufficientDataError):
        chronological_split(30, 0.7, 0.15)


def test_split_invalid_fractions():
    with pytest.raises(ValueError):
        chronological_split(1000, 0.9, 0.2)


def test_make_sequences_alignment():
    x = np.arange(50, dtype=float).reshape(25, 2)
    seq = make_sequences(x, 5)
    assert seq.shape == (21, 5, 2)
    np.testing.assert_array_equal(seq[0], x[0:5])
    np.testing.assert_array_equal(seq[7][-1], x[11])  # window k ends at row k+lookback-1


def test_make_sequences_too_short():
    with pytest.raises(InsufficientDataError):
        make_sequences(np.zeros((3, 2)), 5)


def test_scaler_roundtrip_and_constant_column():
    x = np.column_stack([np.arange(100.0), np.full(100, 7.0)])
    sc = FeatureScaler(clip=None).fit(x)
    z = sc.transform(x)
    assert np.allclose(z[:, 0].mean(), 0, atol=1e-6) and np.allclose(z[:, 1], 0)
    np.testing.assert_allclose(sc.inverse_transform(z), x, atol=1e-4)
    again = FeatureScaler.from_dict(sc.to_dict())
    np.testing.assert_allclose(again.transform(x), z)


def test_scaler_clips_outliers():
    sc = FeatureScaler(clip=3.0).fit(np.random.default_rng(0).normal(size=(500, 1)))
    assert sc.transform(np.array([[1000.0]])).max() == pytest.approx(3.0)


def test_prepared_shapes_and_label_alignment(ohlcv):
    cfg = ModelConfig(lookback=20)
    p = prepare_dataset(ohlcv, cfg)
    assert p.X.shape[1:] == (20, len(p.feature_names))
    assert len(p.X) == len(p.y) == len(p.dates) == len(p.close)
    # label for sample k is the realised return from that decision date to the next bar
    pos = ohlcv.index.get_indexer(p.dates)
    nxt = ohlcv["Close"].to_numpy()[pos + 1]
    np.testing.assert_allclose(p.y, nxt / p.close - 1)
    # splits follow time order
    assert p.split_dates("train")[-1] < p.split_dates("val")[0] < p.split_dates("val")[-1] < p.split_dates("test")[0]


def test_scalers_fit_on_training_data_only(ohlcv):
    """Rewriting the TEST period must not change anything the scalers learn."""
    cfg = ModelConfig(lookback=20)
    base = prepare_dataset(ohlcv, cfg)
    altered = ohlcv.copy()
    altered.iloc[-100:, :4] *= 4.0
    alt = prepare_dataset(altered, cfg)
    np.testing.assert_allclose(base.scaler.mean_, alt.scaler.mean_)
    np.testing.assert_allclose(base.scaler.std_, alt.scaler.std_)
    np.testing.assert_allclose(base.target_scaler.mean_, alt.target_scaler.mean_)
    np.testing.assert_allclose(base.target_scaler.std_, alt.target_scaler.std_)


def test_train_sequences_independent_of_future(ohlcv):
    cfg = ModelConfig(lookback=20)
    base = prepare_dataset(ohlcv, cfg)
    altered = ohlcv.copy()
    altered.iloc[-60:, :4] *= 2.0
    alt = prepare_dataset(altered, cfg)
    n = len(base.X) - 70
    np.testing.assert_allclose(base.X[:n], alt.X[:n], atol=1e-6)
    np.testing.assert_allclose(base.y[:n], alt.y[:n])


def test_too_little_data_raises():
    with pytest.raises(InsufficientDataError):
        prepare_dataset(generate_synthetic_ohlcv(n=100), ModelConfig(lookback=30))
