"""Exploratory data analysis: summary tables, static charts (Matplotlib/Seaborn) and
auto-generated plain-English findings. Used by the notebook and the dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .preprocessing import build_features  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for OHLCV plus daily returns."""
    ret = df["Close"].pct_change().rename("Daily return")
    return pd.concat([df[["Open", "High", "Low", "Close", "Volume"]], ret], axis=1).describe().T


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value counts per column (on the cleaned frame) plus calendar-gap info."""
    out = pd.DataFrame({"missing": df.isna().sum(), "pct": df.isna().mean() * 100})
    return out


def feature_correlation(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    feats = build_features(df)
    if with_target:
        feats["target_next_return"] = df["Close"].pct_change().shift(-1)
    return feats.dropna().corr()


def plot_price_with_averages(df: pd.DataFrame, windows=(20, 50, 200)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["Close"], label="Close", lw=1.2)
    for w in windows:
        if len(df) > w:
            ax.plot(df.index, df["Close"].rolling(w).mean(), label=f"SMA {w}", lw=1)
    ax.set(title="Closing price with moving averages", xlabel="Date", ylabel="Price")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_returns(df: pd.DataFrame) -> plt.Figure:
    ret = df["Close"].pct_change().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ret.index, ret.values, lw=0.7)
    axes[0].set(title="Daily returns", xlabel="Date", ylabel="Return")
    sns.histplot(ret, bins=60, kde=True, ax=axes[1])
    axes[1].axvline(ret.mean(), color="red", ls="--", label=f"mean {ret.mean():.4f}")
    axes[1].set(title="Return distribution", xlabel="Daily return", ylabel="Count")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_volume(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(df.index, df["Volume"], width=1.0, alpha=0.7)
    ax.plot(df.index, df["Volume"].rolling(20).mean(), color="C1", label="20-bar average")
    ax.set(title="Trading volume", xlabel="Date", ylabel="Shares")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_volatility(df: pd.DataFrame, windows=(20, 60), periods_per_year: float = 252) -> plt.Figure:
    ret = df["Close"].pct_change()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for w in windows:
        ax.plot(df.index, ret.rolling(w).std() * np.sqrt(periods_per_year), label=f"{w}-bar rolling")
    ax.set(title="Annualised rolling volatility", xlabel="Date", ylabel="Volatility")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame) -> plt.Figure:
    corr = feature_correlation(df)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Feature correlation (including next-bar return)")
    fig.tight_layout()
    return fig


def describe_findings(df: pd.DataFrame, periods_per_year: float = 252) -> list[str]:
    """Plain-English observations computed from the data (not hard-coded claims)."""
    ret = df["Close"].pct_change().dropna()
    total = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    skew, kurt = ret.skew(), ret.kurt()
    corr = feature_correlation(df)
    target_corr = corr["target_next_return"].drop("target_next_return").abs().sort_values(ascending=False)
    feat_corr = corr.drop(index="target_next_return", columns="target_next_return")
    pairs = feat_corr.where(np.triu(np.ones(feat_corr.shape, dtype=bool), k=1)).stack().abs().sort_values(ascending=False)
    worst_dd = float((df["Close"] / df["Close"].cummax() - 1).min())

    findings = [
        f"The series covers {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}; "
        f"price moved {total:+.1%} over the period with a worst peak-to-trough decline of {worst_dd:.1%}.",
        f"Annualised volatility is about {ann_vol:.1%}. Return skew is {skew:+.2f} and excess kurtosis is "
        f"{kurt:.1f}; "
        + ("fat tails (kurtosis well above 0) mean large moves are more common than a normal distribution implies."
           if kurt > 1 else "tails are close to normal."),
        f"Volatility clusters: the autocorrelation of absolute returns at lag 1 is "
        f"{ret.abs().autocorr(1):+.2f} (raw returns: {ret.autocorr(1):+.2f}).",
        f"The feature most correlated (in magnitude) with the next-bar return is '{target_corr.index[0]}' "
        f"at |r| = {target_corr.iloc[0]:.3f}; "
        + ("this is a weak linear relationship, so any predictive edge is likely small."
           if target_corr.iloc[0] < 0.2 else "a moderate linear relationship worth checking for leakage or overfitting."),
        f"The most redundant feature pair is '{pairs.index[0][0]}' / '{pairs.index[0][1]}' "
        f"(|r| = {pairs.iloc[0]:.2f}); highly correlated inputs add little new information.",
    ]
    return findings


def run_eda(df: pd.DataFrame, out_dir: Optional[str | Path] = None, periods_per_year: float = 252) -> dict:
    """Generate all EDA figures/tables; optionally save PNGs and CSVs into ``out_dir``."""
    figs = {
        "price_ma": plot_price_with_averages(df),
        "returns": plot_returns(df),
        "volume": plot_volume(df),
        "volatility": plot_volatility(df, periods_per_year=periods_per_year),
        "correlation": plot_correlation(df),
    }
    result = {
        "figures": figs,
        "summary": summary_stats(df),
        "missing": missing_report(df),
        "findings": describe_findings(df, periods_per_year),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out / f"eda_{name}.png", dpi=130)
        result["summary"].to_csv(out / "eda_summary.csv")
        (out / "eda_findings.txt").write_text("\n".join(f"- {f}" for f in result["findings"]))
    return result
