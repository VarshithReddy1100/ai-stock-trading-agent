"""Market data retrieval, validation and cleaning.

Design rules:
* Real data comes only from the provider (Yahoo Finance via ``yfinance``) or from a
  CSV the user supplies.
* If a download fails we raise a descriptive error. We never fabricate data or
  silently substitute a different ticker / period.
* A stale cache is used only when the caller explicitly opts in
  (``allow_stale_cache=True``) and this is logged.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import IO, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SUPPORTED_INTERVALS = ("1d", "5d", "1wk", "1mo", "1h", "30m", "15m", "5m")
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")
_CANONICAL = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "adjclose": "Adj Close",
}


class DataError(Exception):
    """Base class for data problems."""


class InvalidTickerError(DataError):
    """Ticker is malformed or the provider returned nothing for it."""


class DataUnavailableError(DataError):
    """Provider could not be reached, was rate limited, or returned an error."""


class DataValidationError(DataError):
    """Data is structurally invalid (missing columns, bad index, ...)."""


class InsufficientDataError(DataError):
    """Not enough observations to train / evaluate."""


# --------------------------------------------------------------------------- #
# Cleaning & validation
# --------------------------------------------------------------------------- #
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and map names to canonical OHLCV names."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [_CANONICAL.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    # If two columns collapse to the same name keep the first.
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a clean, chronologically ordered OHLCV frame and a cleaning report.

    Steps: normalise columns, parse/sort the index, drop duplicate timestamps
    (keep last), coerce to numeric, drop rows with missing/non-positive prices or
    ``High < Low`` or negative volume. Missing volume is set to 0.
    Prices are never interpolated or forward-filled.
    """
    df = normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {missing}. Found: {list(df.columns)}"
        )

    report: dict = {"rows_in": int(len(df))}

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:  # noqa: BLE001
            raise DataValidationError(f"Index could not be parsed as dates: {exc}") from exc
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)  # keep local wall-clock time
    df.index.name = "Date"

    keep = REQUIRED_COLUMNS + (["Adj Close"] if "Adj Close" in df.columns else [])
    df = df[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    nat = df.index.isna()
    report["bad_timestamp_rows_dropped"] = int(nat.sum())
    df = df[~nat]

    dup = df.index.duplicated(keep="last")
    report["duplicate_timestamps_removed"] = int(dup.sum())
    df = df[~dup]

    report["was_unsorted"] = bool(not df.index.is_monotonic_increasing)
    df = df.sort_index()

    vol_nan = df["Volume"].isna() & df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    report["missing_volume_set_to_zero"] = int(vol_nan.sum())
    df.loc[vol_nan, "Volume"] = 0.0

    price_cols = ["Open", "High", "Low", "Close"]
    nan_rows = df[price_cols + ["Volume"]].isna().any(axis=1)
    report["rows_with_missing_values_dropped"] = int(nan_rows.sum())
    df = df[~nan_rows]

    invalid = (df[price_cols] <= 0).any(axis=1) | (df["High"] < df["Low"]) | (df["Volume"] < 0)
    report["invalid_value_rows_dropped"] = int(invalid.sum())
    df = df[~invalid]

    report["rows_out"] = int(len(df))
    if len(df):
        report["first_date"] = str(df.index[0].date())
        report["last_date"] = str(df.index[-1].date())
        gaps = df.index.to_series().diff().dt.days.dropna()
        report["max_gap_days"] = int(gaps.max()) if len(gaps) else 0
    dropped = report["rows_in"] - report["rows_out"]
    if dropped:
        logger.warning("clean_ohlcv dropped %d of %d rows: %s", dropped, report["rows_in"], report)
    return df, report


def validate_ohlcv(df: pd.DataFrame, min_rows: int = 300) -> None:
    """Raise if ``df`` is not a usable, chronologically ordered OHLCV frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("Index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise DataValidationError("Index must be sorted in chronological order.")
    if df.index.has_duplicates:
        raise DataValidationError("Index contains duplicate timestamps.")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise DataValidationError("Data contains missing values after cleaning.")
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Only {len(df)} valid rows available; at least {min_rows} are required. "
            "Use a longer date range or a finer interval."
        )


# --------------------------------------------------------------------------- #
# CSV input
# --------------------------------------------------------------------------- #
def load_csv(
    source: Union[str, Path, IO], min_rows: int = 300
) -> tuple[pd.DataFrame, dict]:
    """Load OHLCV data from a CSV path or file-like object (e.g. a Streamlit upload)."""
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError(f"Could not read CSV: {exc}") from exc
    if raw.empty:
        raise DataValidationError("CSV file is empty.")

    date_col = next(
        (c for c in raw.columns if str(c).strip().lower() in ("date", "datetime", "timestamp", "time")),
        raw.columns[0],
    )
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce", utc=False)
    raw = raw.set_index(date_col)
    df, report = clean_ohlcv(raw)
    validate_ohlcv(df, min_rows)
    report["source"] = "csv"
    return df, report


# --------------------------------------------------------------------------- #
# Yahoo Finance download with caching
# --------------------------------------------------------------------------- #
def _check_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not _TICKER_RE.match(t):
        raise InvalidTickerError(
            f"'{ticker}' is not a valid ticker symbol (letters, digits, '.', '-', '^', '=' only)."
        )
    return t


def download_prices(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    interval: str = "1d",
) -> pd.DataFrame:
    """Download raw adjusted OHLCV from Yahoo Finance. Raises on any failure."""
    ticker = _check_ticker(ticker)
    if interval not in SUPPORTED_INTERVALS:
        raise DataValidationError(f"Unsupported interval '{interval}'. Use one of {SUPPORTED_INTERVALS}.")
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailableError("yfinance is not installed (pip install yfinance).") from exc

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,  # OHLC adjusted for splits and dividends
            progress=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001 - provider raises many exception types
        msg = str(exc)
        if "rate" in msg.lower() or "too many" in msg.lower() or "429" in msg:
            raise DataUnavailableError(
                "Yahoo Finance rate-limited this request. Wait a few minutes and retry."
            ) from exc
        raise DataUnavailableError(f"Could not download data for {ticker}: {msg}") from exc

    if raw is None or raw.empty:
        raise InvalidTickerError(
            f"No data returned for '{ticker}' ({start} -> {end or 'latest'}, interval {interval}). "
            "Check the symbol, date range and interval, and your internet connection."
        )
    return raw


def _cache_path(cache_dir: Path, ticker: str, start, end, interval) -> Path:
    key = f"{ticker}_{interval}_{start or 'min'}_{end or 'latest'}".replace(":", "-").replace("/", "-")
    return cache_dir / f"{key}.csv"


def load_prices(
    ticker: str,
    start: Optional[str] = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache_dir: Union[str, Path] = "data/cache",
    cache_ttl_hours: float = 12.0,
    min_rows: int = 300,
    force_refresh: bool = False,
    allow_stale_cache: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Load cleaned OHLCV for ``ticker`` (cached on disk).

    Returns ``(df, report)``. ``report`` describes cleaning and whether the cache was used.
    """
    ticker = _check_ticker(ticker)
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, ticker, start, end, interval)

    fresh = (
        path.exists()
        and not force_refresh
        and (time.time() - path.stat().st_mtime) < cache_ttl_hours * 3600
    )
    if fresh:
        logger.info("Using cached data %s", path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df, report = clean_ohlcv(df)
        validate_ohlcv(df, min_rows)
        report.update(source="cache", ticker=ticker)
        return df, report

    try:
        raw = download_prices(ticker, start, end, interval)
    except (DataUnavailableError, InvalidTickerError) as exc:
        if allow_stale_cache and path.exists():
            logger.warning("Download failed (%s); using STALE cache %s as explicitly allowed.", exc, path)
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df, report = clean_ohlcv(df)
            validate_ohlcv(df, min_rows)
            report.update(source="stale_cache", ticker=ticker, warning=str(exc))
            return df, report
        raise

    df, report = clean_ohlcv(raw)
    validate_ohlcv(df, min_rows)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except OSError as exc:  # caching is best-effort
        logger.warning("Could not write cache file %s: %s", path, exc)
    report.update(source="yfinance", ticker=ticker)
    return df, report


def load_data(
    ticker: Optional[str] = None,
    csv: Optional[Union[str, Path, IO]] = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Single entry point: a CSV if given, otherwise Yahoo Finance for ``ticker``."""
    if csv is not None:
        return load_csv(csv, min_rows=kwargs.get("min_rows", 300))
    if not ticker:
        raise DataValidationError("Provide either a ticker or a CSV file.")
    return load_prices(ticker, **kwargs)
