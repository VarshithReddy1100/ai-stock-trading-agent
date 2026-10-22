"""LSTM next-bar-return model, training loop, baselines, metrics and persistence.

Target: the simple return from close ``t`` to close ``t+1``, predicted from the
previous ``lookback`` bars of scale-free features (see ``preprocessing``).
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelConfig
from .data_collection import InsufficientDataError
from .preprocessing import FeatureScaler, PreparedData, build_features, make_sequences

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, float, float], None]


class TrainingError(RuntimeError):
    """Training could not complete (e.g. diverged to NaN)."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class LSTMRegressor(nn.Module):
    """Input ``(batch, lookback, n_features)`` -> output ``(batch,)`` standardised next-bar return."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(
    data: PreparedData,
    cfg: ModelConfig,
    progress: Optional[ProgressFn] = None,
) -> tuple[LSTMRegressor, dict]:
    """Train on the chronological train split with early stopping on the val split.

    Mini-batches *within the training window* are shuffled (each window is an
    independent sample); the split itself is strictly chronological.
    Returns the model restored to its best-validation weights and a history dict.
    """
    tr, va = data.splits["train"], data.splits["val"]
    if len(tr) < max(50, cfg.batch_size):
        raise InsufficientDataError(
            f"Only {len(tr)} training samples; need at least {max(50, cfg.batch_size)}."
        )

    set_seed(cfg.seed)
    n_features = data.X.shape[2]
    model = LSTMRegressor(n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, cfg.patience // 2)
    )
    loss_fn: nn.Module = nn.HuberLoss(delta=1.0) if cfg.loss == "huber" else nn.MSELoss()

    train_loader = _loader(data.X[tr], data.y_scaled[tr], cfg.batch_size, True, cfg.seed)
    val_loader = _loader(data.X[va], data.y_scaled[va], 512, False, cfg.seed)

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0, "stopped_epoch": 0}
    best_val, best_state, stale = float("inf"), None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise TrainingError(
                    f"Loss became non-finite at epoch {epoch}. Try a lower learning rate."
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
        train_loss = total / count

        model.eval()
        with torch.no_grad():
            vt, vc = 0.0, 0
            for xb, yb in val_loader:
                vt += loss_fn(model(xb), yb).item() * len(xb)
                vc += len(xb)
            val_loss = vt / vc
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["stopped_epoch"] = epoch
        if progress:
            progress(epoch, cfg.epochs, train_loss, val_loss)

        if val_loss < best_val - 1e-6:
            best_val, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= cfg.patience:
                logger.info("Early stopping at epoch %d (best epoch %d).", epoch, history["best_epoch"])
                break

    if best_state is None:
        raise TrainingError("Training produced no valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_scaled(model: LSTMRegressor, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Raw (standardised) model output for sequences ``x``."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            outs.append(model(torch.from_numpy(np.ascontiguousarray(x[i : i + batch_size]))).numpy())
    return np.concatenate(outs) if outs else np.array([], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Bundle: everything needed for reproducible inference
# --------------------------------------------------------------------------- #
@dataclass
class ModelBundle:
    model: LSTMRegressor
    scaler: FeatureScaler
    target_scaler: FeatureScaler
    feature_names: list[str]
    config: ModelConfig
    history: dict = field(default_factory=dict)

    def predict_returns(self, df: pd.DataFrame) -> pd.Series:
        """Predicted next-bar return for every bar in ``df`` that has a full lookback window.

        The value at date ``t`` uses only data up to ``t`` and forecasts close(t+1)/close(t)-1.
        """
        feats = build_features(df)[self.feature_names].dropna()
        L = self.config.lookback
        if len(feats) < L:
            raise InsufficientDataError(
                f"Need at least {L} complete feature rows for inference, have {len(feats)}."
            )
        x = make_sequences(self.scaler.transform(feats.to_numpy()), L)
        z = predict_scaled(self.model, x)
        preds = self.target_scaler.inverse_transform(z).reshape(-1)
        return pd.Series(preds, index=feats.index[L - 1 :], name="pred_return")

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / "model.pt")
        meta = {
            "feature_names": self.feature_names,
            "model_config": asdict(self.config),
            "scaler": self.scaler.to_dict(),
            "target_scaler": self.target_scaler.to_dict(),
            "history": self.history,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "ModelBundle":
        d = Path(directory)
        try:
            meta = json.loads((d / "meta.json").read_text())
            cfg = ModelConfig(**meta["model_config"])
            model = LSTMRegressor(len(meta["feature_names"]), cfg.hidden_size, cfg.num_layers, cfg.dropout)
            model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No saved model found in '{d}'.") from exc
        model.eval()
        return cls(
            model=model,
            scaler=FeatureScaler.from_dict(meta["scaler"]),
            target_scaler=FeatureScaler.from_dict(meta["target_scaler"]),
            feature_names=meta["feature_names"],
            config=cfg,
            history=meta.get("history", {}),
        )


# --------------------------------------------------------------------------- #
# Metrics & baselines
# --------------------------------------------------------------------------- #
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, close: Optional[np.ndarray] = None) -> dict:
    """MAE / RMSE on returns, directional accuracy, and (if ``close`` given) next-close MAE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    # Directional accuracy only counts bars where both a direction was called (pred != 0)
    # and the market actually moved (true != 0). A model that always predicts 0 (price
    # persistence) makes no directional call, so its accuracy is undefined (NaN), not 0%.
    called = (y_true != 0) & (y_pred != 0)
    dir_acc = float(np.mean(np.sign(y_pred[called]) == np.sign(y_true[called]))) if called.any() else float("nan")
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Directional accuracy": dir_acc,
    }
    if close is not None:
        close = np.asarray(close, float)
        out["Next-close MAE ($)"] = float(np.mean(np.abs(close * err)))
    return out


def baseline_predictions(data: PreparedData) -> dict[str, np.ndarray]:
    """Predictions for every sample from simple reference models.

    * ``Persistence (price)``: tomorrow's close equals today's -> predicted return 0.
    * ``Momentum (last return)``: tomorrow's return equals today's.
    * ``Ridge regression``: linear model on the latest bar's scaled features,
      fit on the training split only.
    """
    n = len(data.y)
    tr = data.splits["train"]
    last_bar = data.X[:, -1, :]
    ridge = Ridge(alpha=10.0).fit(last_bar[tr], data.y[tr])
    return {
        "Persistence (price)": np.zeros(n),
        "Momentum (last return)": data.raw_ret_1.copy(),
        "Ridge regression": ridge.predict(last_bar),
    }


def compare_models(
    data: PreparedData, lstm_pred: np.ndarray, split: str = "test"
) -> tuple[pd.DataFrame, str]:
    """Metrics table (LSTM + baselines) on ``split`` and an honest plain-English verdict."""
    idx = data.splits[split]
    preds = {"LSTM": np.asarray(lstm_pred)} | baseline_predictions(data)
    rows = {
        name: regression_metrics(data.y[idx], p[idx], data.close[idx]) for name, p in preds.items()
    }
    table = pd.DataFrame(rows).T

    n = len(idx)
    se = np.sqrt(0.25 / n)
    da = table.loc["LSTM", "Directional accuracy"]
    best_base = table.drop(index="LSTM")["RMSE"].idxmin()
    lstm_rmse, base_rmse = table.loc["LSTM", "RMSE"], table.loc[best_base, "RMSE"]

    parts = []
    if lstm_rmse < base_rmse:
        parts.append(
            f"On the {split} set the LSTM has a lower RMSE ({lstm_rmse:.5f}) than the best baseline "
            f"'{best_base}' ({base_rmse:.5f})."
        )
    else:
        parts.append(
            f"On the {split} set the LSTM did NOT beat the best baseline '{best_base}' on RMSE "
            f"({lstm_rmse:.5f} vs {base_rmse:.5f})."
        )
    if np.isfinite(da):
        if abs(da - 0.5) <= 2 * se:
            parts.append(
                f"Its directional accuracy ({da:.1%}, n={n}) is within about two standard errors "
                "of 50%, so it is not distinguishable from a coin flip."
            )
        else:
            side = "above" if da > 0.5 else "below"
            parts.append(
                f"Its directional accuracy ({da:.1%}, n={n}) is {side} 50% by more than two "
                "standard errors, but that alone does not imply a profitable strategy."
            )
    parts.append("Returns are very noisy, so differences this small should not be over-interpreted.")
    return table, " ".join(parts)
