"""Предсказание удобоукладываемости (осадка конуса, мм) бетонной смеси.

Модуль обучает простую BNN (MC Dropout) на внешней базе данных
Normal_Concrete_DB (873 записи со значениями осадки конуса).

Это БОНУСНАЯ цель: параметр не зависит от времени и характеризует
технологичность укладки смеси.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .scaler import StandardScaler


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class WorkabilityConfig:
    # Data
    data_path: str = "data/workability_data.csv"
    # Model
    hidden_layers: list[int] = field(default_factory=lambda: [128, 128, 64])
    dropout: float = 0.15
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 300
    early_stopping_rounds: int = 30
    validation_split: float = 0.20
    seed: int = 42
    # Inference
    mc_samples: int = 64

    @classmethod
    def from_dict(cls, d: dict) -> "WorkabilityConfig":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ── Model ─────────────────────────────────────────────────────────────────────

class WorkabilityMLP(nn.Module):
    """MLP with MC Dropout for workability (slump) regression.

    At inference, call with ``training=True`` mode N times to get MC samples.
    """

    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for w in hidden_layers:
            layers += [nn.Linear(prev, w), nn.SiLU(), nn.Dropout(dropout)]
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_mc(self, x: torch.Tensor, mc_samples: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and std in MPa/mm units (after inverse_transform applied by caller)."""
        self.train()  # enable dropout
        preds = torch.stack([self(x) for _ in range(mc_samples)], dim=0)  # (S, N)
        mean = preds.mean(0).detach().numpy()
        std  = preds.std(0).detach().numpy()
        return mean, std


# ── Feature engineering ───────────────────────────────────────────────────────

_COMPONENT_COLS = ["cement", "slag_kg", "fly_ash", "microsilica_kg",
                   "water", "plasticizer_kg", "gravel", "sand"]

def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from workability dataframe."""
    def _col(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name].astype(str).str.replace(",", "."),
                                 errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.zeros(len(df), dtype=float)

    cement  = _col("cement")
    water   = _col("water")
    plasticizer = _col("plasticizer_kg")
    fly_ash = _col("fly_ash")
    slag    = _col("slag_kg")
    micro   = _col("microsilica_kg")
    sand    = _col("sand")
    gravel  = _col("gravel")

    eps = 1e-4
    # binder = total cementitious content
    binder = cement + 0.4 * fly_ash + 0.6 * slag + micro
    np.clip(binder, eps, None, out=binder)

    wc   = water / np.clip(cement, eps, None)
    wb   = water / binder
    pb   = plasticizer / binder * 100.0          # plasticizer % of binder
    agg  = (sand + gravel) / binder              # aggregate/binder
    total = cement + slag + fly_ash + micro + water + plasticizer + sand + gravel

    blocks: list[np.ndarray] = [
        cement[:, None], slag[:, None], fly_ash[:, None], micro[:, None],
        water[:, None],  plasticizer[:, None],
        sand[:, None],   gravel[:, None],
        wc[:, None],     wb[:, None],
        pb[:, None],     agg[:, None],
        total[:, None],
    ]
    names = [
        "cement", "slag_kg", "fly_ash", "microsilica_kg",
        "water", "plasticizer_kg",
        "sand", "gravel",
        "wc_ratio", "wb_ratio",
        "plasticizer_binder_pct", "aggregate_binder_ratio",
        "total_mass",
    ]
    X = np.hstack(blocks)
    return X, names


def _load_workability_data(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    # target
    slump = pd.to_numeric(df["slump"].astype(str).str.replace(",", "."),
                          errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(slump)
    df = df[valid].reset_index(drop=True)
    slump = slump[valid]

    X, feature_names = _build_features(df)
    return X, slump[:, None], feature_names   # y shape: (N, 1)


# ── Training ──────────────────────────────────────────────────────────────────

def _metrics_workability(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.ravel(); y_pred = y_pred.ravel()
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1.0 - ss_res / (ss_tot + 1e-10))
    mask = np.abs(y_true) > 1.0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def run_train_workability(
    *,
    config_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train MC-Dropout MLP to predict slump (workability) from mix composition."""

    config_path = Path(config_path)
    config = WorkabilityConfig.from_dict(json.loads(config_path.read_text()))

    root_dir = Path(artifacts_dir)
    out_dir  = Path(output_dir) if output_dir else root_dir / "train_workability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    (out_dir / "workability_config.resolved.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2)
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, feature_names = _load_workability_data(config.data_path)
    print(f"[workability] Loaded {len(X)} samples, {X.shape[1]} features, "
          f"slump {y.min():.0f}–{y.max():.0f} mm")

    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * config.validation_split))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    x_train_raw, y_train_raw = X[train_idx], y[train_idx]
    x_val_raw,   y_val_raw   = X[val_idx],   y[val_idx]

    x_scaler = StandardScaler.fit(x_train_raw)
    y_scaler = StandardScaler.fit(y_train_raw)

    x_train = torch.tensor(x_scaler.transform(x_train_raw), dtype=torch.float32)
    y_train = torch.tensor(y_scaler.transform(y_train_raw), dtype=torch.float32).squeeze(-1)
    x_val   = torch.tensor(x_scaler.transform(x_val_raw),   dtype=torch.float32)

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(config.seed)
    model = WorkabilityMLP(X.shape[1], config.hidden_layers, config.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    no_improve    = 0
    history: list[dict] = []

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(x_train)

        # Validation (deterministic, dropout off)
        model.eval()
        with torch.no_grad():
            val_pred_sc = model(x_val).numpy()
        val_pred = y_scaler.inverse_transform(val_pred_sc[:, None]).ravel()
        val_loss = float(np.mean((y_val_raw.ravel() - val_pred) ** 2))

        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_mse": val_loss})

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
        if no_improve >= config.early_stopping_rounds:
            print(f"[workability] Early stop at epoch {epoch}")
            break

        if epoch % 50 == 0:
            val_metrics = _metrics_workability(y_val_raw.ravel(), val_pred)
            print(f"[workability] ep {epoch:4d}  train_loss={epoch_loss:.4f}  "
                  f"val_MAE={val_metrics['mae']:.2f}mm  val_R²={val_metrics['r2']:.3f}")

    model.load_state_dict(best_state)

    # ── Final evaluation (MC) ─────────────────────────────────────────────────
    mc_mean_sc, mc_std_sc = model.predict_mc(x_val, mc_samples=config.mc_samples)
    mc_mean = y_scaler.inverse_transform(mc_mean_sc[:, None]).ravel()
    mc_std  = mc_std_sc * y_scaler.scale.ravel()[0]   # approximate undo of scaling
    train_pred_sc, _ = model.predict_mc(x_train, mc_samples=config.mc_samples)
    train_pred = y_scaler.inverse_transform(train_pred_sc[:, None]).ravel()

    val_metrics   = _metrics_workability(y_val_raw.ravel(), mc_mean)
    train_metrics = _metrics_workability(y_train_raw.ravel(), train_pred)

    picp95 = float(np.mean(
        (y_val_raw.ravel() >= (mc_mean - 1.96 * mc_std)) &
        (y_val_raw.ravel() <= (mc_mean + 1.96 * mc_std))
    ))

    print(f"[workability] Final val  MAE={val_metrics['mae']:.2f}mm  "
          f"RMSE={val_metrics['rmse']:.2f}mm  MAPE={val_metrics['mape']:.2f}%  "
          f"R²={val_metrics['r2']:.3f}  PICP95={picp95:.3f}")
    print(f"[workability] Final train MAE={train_metrics['mae']:.2f}mm  R²={train_metrics['r2']:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    checkpoint = {
        "state_dict":     best_state,
        "x_scaler":       x_scaler.to_dict(),
        "y_scaler":       y_scaler.to_dict(),
        "feature_names":  feature_names,
        "config":         config.to_dict(),
    }
    torch.save(checkpoint, out_dir / "workability_model.pt")

    # Predictions CSV
    val_df = pd.DataFrame({
        "slump_true_mm":  y_val_raw.ravel(),
        "slump_pred_mm":  mc_mean,
        "slump_std_mm":   mc_std,
    })
    val_df.to_csv(out_dir / "val_predictions.csv", index=False)

    summary = {
        "stage": "train_workability",
        "artifacts_dir": str(out_dir),
        "n_train": int(len(train_idx)),
        "n_val":   int(len(val_idx)),
        "feature_names": feature_names,
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "validation_picp95": picp95,
            "validation_mean_uncertainty_mm": float(mc_std.mean()),
        },
        "training_history": history[-10:],
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary
