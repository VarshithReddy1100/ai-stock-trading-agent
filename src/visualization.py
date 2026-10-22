"""Interactive Plotly charts for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtesting import BacktestResult

_BLUE, _GREY, _GREEN, _RED, _ORANGE = "#2563eb", "#6b7280", "#16a34a", "#dc2626", "#f59e0b"


def _base(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
        template="plotly_white",
    )
    return fig


def price_with_signals(df: pd.DataFrame, trades: pd.DataFrame, start=None) -> go.Figure:
    """Candlestick-free close-price line with executed BUY / SELL markers."""
    d = df if start is None else df[df.index >= start]
    fig = go.Figure(go.Scatter(x=d.index, y=d["Close"], name="Close", line=dict(color=_BLUE, width=1.5)))
    if len(trades):
        t = trades[trades["date"].isin(d.index)]
        for side, color, symbol in (("BUY", _GREEN, "triangle-up"), ("SELL", _RED, "triangle-down")):
            s = t[t["side"] == side]
            fig.add_trace(go.Scatter(
                x=s["date"], y=s["price"], mode="markers", name=f"{side} fill",
                marker=dict(color=color, size=11, symbol=symbol, line=dict(width=1, color="white")),
                hovertemplate="%{x|%Y-%m-%d}<br>fill %{y:.2f}<extra>" + side + "</extra>",
            ))
    fig.update_yaxes(title="Price")
    return _base(fig, "Price and executed trades (test period)")


def prediction_chart(pred_frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Next-bar close: actual vs predicted", "Next-bar return: actual vs predicted"))
    fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame["actual_next_close"], name="Actual next close",
                             line=dict(color=_BLUE)), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame["pred_next_close"], name="Predicted next close",
                             line=dict(color=_ORANGE, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame["actual_return"], name="Actual return",
                             line=dict(color=_BLUE, width=1), opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame["pred_return"], name="Predicted return",
                             line=dict(color=_ORANGE, width=1.5)), row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return", tickformat=".1%", row=2, col=1)
    return _base(fig, "Model predictions on the held-out test period", height=560)


def equity_chart(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity["equity"], name="Strategy",
                             line=dict(color=_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark["equity"], name="Buy & hold",
                             line=dict(color=_GREY, width=2, dash="dash")))
    fig.update_yaxes(title="Portfolio value ($)")
    return _base(fig, "Portfolio value: strategy vs buy-and-hold")


def drawdown_chart(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    for name, frame, color in (("Strategy", result.equity, _BLUE), ("Buy & hold", result.benchmark, _GREY)):
        dd = frame["equity"] / frame["equity"].cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, line=dict(color=color), fill="tozeroy",
                                 opacity=0.6 if name == "Strategy" else 0.4))
    fig.update_yaxes(title="Drawdown", tickformat=".0%")
    return _base(fig, "Drawdown from running peak", height=320)


def training_chart(history: dict) -> go.Figure:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train loss", line=dict(color=_BLUE)))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Validation loss", line=dict(color=_ORANGE)))
    if history.get("best_epoch"):
        fig.add_vline(x=history["best_epoch"], line_dash="dot", line_color=_GREEN,
                      annotation_text="best epoch", annotation_position="top")
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss (standardised target)")
    return _base(fig, "Training and validation loss", height=340)
