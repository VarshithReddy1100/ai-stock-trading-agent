import io
import os
import time

import numpy as np
import pandas as pd
import pytest

from src import data_collection as dc
from src.synthetic import generate_synthetic_ohlcv


def _raw(n=400):
    return generate_synthetic_ohlcv(n=n, seed=3)


def test_clean_removes_duplicates_sorts_and_drops_invalid():
    df = _raw(50)
    messy = pd.concat([df.iloc[10:], df.iloc[:10], df.iloc[[12]]])  # unsorted + duplicate timestamp
    messy.iloc[3, messy.columns.get_loc("Close")] = np.nan
    messy.iloc[5, messy.columns.get_loc("Open")] = -1.0
    clean, report = dc.clean_ohlcv(messy)
    assert clean.index.is_monotonic_increasing and not clean.index.has_duplicates
    assert report["duplicate_timestamps_removed"] == 1
    assert report["rows_with_missing_values_dropped"] == 1
    assert report["invalid_value_rows_dropped"] == 1
    assert report["was_unsorted"] is True
    assert not clean.isna().any().any()


def test_clean_drops_high_below_low_and_negative_volume():
    df = _raw(30)
    df.iloc[2, df.columns.get_loc("High")] = df.iloc[2]["Low"] - 1
    df.iloc[4, df.columns.get_loc("Volume")] = -5
    clean, report = dc.clean_ohlcv(df)
    assert report["invalid_value_rows_dropped"] == 2
    assert len(clean) == 28


def test_missing_volume_set_to_zero_not_dropped():
    df = _raw(30)
    df.iloc[7, df.columns.get_loc("Volume")] = np.nan
    clean, report = dc.clean_ohlcv(df)
    assert report["missing_volume_set_to_zero"] == 1
    assert len(clean) == 30 and clean.iloc[7]["Volume"] == 0


def test_missing_columns_raise():
    with pytest.raises(dc.DataValidationError):
        dc.clean_ohlcv(_raw(20).drop(columns=["Close"]))


def test_multiindex_columns_are_flattened():
    df = _raw(20)
    df.columns = pd.MultiIndex.from_product([df.columns, ["AAPL"]])
    clean, _ = dc.clean_ohlcv(df)
    assert list(clean.columns) == dc.REQUIRED_COLUMNS


def test_insufficient_rows_raise():
    clean, _ = dc.clean_ohlcv(_raw(50))
    with pytest.raises(dc.InsufficientDataError):
        dc.validate_ohlcv(clean, min_rows=300)


def test_validate_rejects_unsorted():
    df = _raw(50).iloc[::-1]
    with pytest.raises(dc.DataValidationError):
        dc.validate_ohlcv(df, min_rows=10)


def test_load_csv_roundtrip():
    df = _raw(350)
    buf = io.StringIO()
    df.to_csv(buf)
    buf.seek(0)
    loaded, report = dc.load_csv(buf, min_rows=300)
    pd.testing.assert_frame_equal(loaded, df, check_freq=False, check_exact=False, atol=1e-6)
    assert report["source"] == "csv"


def test_load_csv_lowercase_headers():
    df = _raw(320).reset_index()
    df.columns = [c.lower() for c in df.columns]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    loaded, _ = dc.load_csv(buf, min_rows=300)
    assert list(loaded.columns) == dc.REQUIRED_COLUMNS


@pytest.mark.parametrize("bad", ["", "AA PL", "DROP;TABLE", "x" * 40])
def test_invalid_ticker_rejected_before_network(bad):
    with pytest.raises(dc.InvalidTickerError):
        dc.load_prices(bad)


def test_failed_download_raises_and_never_fabricates(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise dc.DataUnavailableError("offline")

    monkeypatch.setattr(dc, "download_prices", boom)
    with pytest.raises(dc.DataUnavailableError):
        dc.load_prices("AAPL", cache_dir=tmp_path)


def test_fresh_cache_used_without_network(tmp_path, monkeypatch):
    df = _raw(350)
    path = dc._cache_path(tmp_path, "AAPL", "2015-01-01", None, "1d")
    df.to_csv(path)
    monkeypatch.setattr(dc, "download_prices", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    loaded, report = dc.load_prices("AAPL", "2015-01-01", None, "1d", cache_dir=tmp_path, min_rows=300)
    assert report["source"] == "cache" and len(loaded) == 350


def test_stale_cache_only_when_explicitly_allowed(tmp_path, monkeypatch):
    df = _raw(350)
    path = dc._cache_path(tmp_path, "AAPL", "2015-01-01", None, "1d")
    df.to_csv(path)
    old = time.time() - 10 * 24 * 3600
    os.utime(path, (old, old))

    def boom(*a, **k):
        raise dc.DataUnavailableError("offline")

    monkeypatch.setattr(dc, "download_prices", boom)
    with pytest.raises(dc.DataUnavailableError):
        dc.load_prices("AAPL", "2015-01-01", None, "1d", cache_dir=tmp_path, min_rows=300)
    loaded, report = dc.load_prices("AAPL", "2015-01-01", None, "1d", cache_dir=tmp_path,
                                    min_rows=300, allow_stale_cache=True)
    assert report["source"] == "stale_cache" and "offline" in report["warning"]


def test_empty_download_is_invalid_ticker(tmp_path, monkeypatch):
    import types, sys

    fake = types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    with pytest.raises(dc.InvalidTickerError):
        dc.download_prices("ZZZZ", "2020-01-01", None, "1d")


def test_rate_limit_message_mapped(monkeypatch):
    import types, sys

    def limited(*a, **k):
        raise RuntimeError("Too Many Requests. Rate limited.")

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=limited))
    with pytest.raises(dc.DataUnavailableError, match="rate-limited"):
        dc.download_prices("AAPL", "2020-01-01", None, "1d")


def test_unsupported_interval():
    with pytest.raises(dc.DataValidationError):
        dc.download_prices("AAPL", "2020-01-01", None, "3d")
