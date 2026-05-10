from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OptimizerConfig, _resolve_config_path
from .data import read_dataset_frame
from .neat_bnn import NeatBNNRegressor, build_regressor_from_genome
from .neat_optimizer import NEATOptimizer
from .scaler import StandardScaler
from .stage_common import write_json


@dataclass
class GanDatasetConfig:
    """Dataset block for conditional GAN training."""

    data_path: str
    components: list[str]
    strength_by_day: dict[str, str] = field(default_factory=dict)
    time_column: str | None = None
    strength_column: str | None = None
    component_aliases: dict[str, str] = field(default_factory=dict)
    min_time: float = 0.0
    max_time: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GanDatasetConfig":
        config = cls(
            data_path=str(payload.get("data_path", "")),
            components=[str(name) for name in payload.get("components", [])],
            strength_by_day={str(day): str(name) for day, name in payload.get("strength_by_day", {}).items()},
            time_column=None if payload.get("time_column") in (None, "") else str(payload.get("time_column")),
            strength_column=None if payload.get("strength_column") in (None, "") else str(payload.get("strength_column")),
            component_aliases={str(src): str(dst) for src, dst in payload.get("component_aliases", {}).items()},
            min_time=float(payload.get("min_time", 0.0)),
            max_time=None if payload.get("max_time") in (None, "") else float(payload.get("max_time")),
        )
        config.validate()
        return config

    def resolve_paths(self, base_dir: Path) -> None:
        resolved = _resolve_config_path(base_dir, self.data_path)
        self.data_path = "" if resolved is None else resolved

    def validate(self) -> None:
        if not self.data_path:
            raise ValueError("dataset.data_path must not be empty")
        if not self.components:
            raise ValueError("dataset.components must contain at least one column")
        if not self.strength_by_day and not (self.time_column and self.strength_column):
            raise ValueError(
                "dataset must define either strength_by_day or both time_column and strength_column"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "components": self.components,
            "strength_by_day": self.strength_by_day,
            "time_column": self.time_column,
            "strength_column": self.strength_column,
            "component_aliases": self.component_aliases,
            "min_time": self.min_time,
            "max_time": self.max_time,
        }


@dataclass
class GanPhysicsConfig:
    """Physics priors and regularization controls."""

    cement_feature: str = "cement"
    water_feature: str = "water"
    add_log_time_feature: bool = True
    add_abrams_features: bool = True
    add_extended_features: bool = False   # log(c/w), effective w/c, aggregate/binder, plasticizer dosage
    add_late_age_features: bool = False    # extra late-age ratios for 28-day emphasis
    monotonic_weight: float = 0.08
    abrams_weight: float = 0.06

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GanPhysicsConfig":
        return cls(
            cement_feature=str(payload.get("cement_feature", "cement")),
            water_feature=str(payload.get("water_feature", "water")),
            add_log_time_feature=bool(payload.get("add_log_time_feature", True)),
            add_abrams_features=bool(payload.get("add_abrams_features", True)),
            add_extended_features=bool(payload.get("add_extended_features", False)),
            add_late_age_features=bool(payload.get("add_late_age_features", False)),
            monotonic_weight=float(payload.get("monotonic_weight", 0.08)),
            abrams_weight=float(payload.get("abrams_weight", 0.06)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cement_feature": self.cement_feature,
            "water_feature": self.water_feature,
            "add_log_time_feature": self.add_log_time_feature,
            "add_abrams_features": self.add_abrams_features,
            "add_extended_features": self.add_extended_features,
            "add_late_age_features": self.add_late_age_features,
            "monotonic_weight": self.monotonic_weight,
            "abrams_weight": self.abrams_weight,
        }


@dataclass
class GanGeneratorConfig:
    """Generator model and optimization settings."""

    hidden_layers: list[int] = field(default_factory=lambda: [128, 96, 64])
    noise_dim: int = 12
    dropout: float = 0.08
    learning_rate: float = 0.0015
    weight_decay: float = 1e-4
    batch_size: int = 64
    warmup_epochs: int = 35
    epochs_per_round: int = 24
    early_stopping_rounds: int = 18
    supervised_weight: float = 1.0
    adversarial_weight: float = 0.25
    mape_weight: float = 0.0
    period_weights: list[float] = field(default_factory=list)   # per-period loss weights; uniform if empty
    max_grad_norm: float = 1.5
    use_uncertainty_head: bool = False  # if True, generator also outputs log-variance per period
    nll_weight: float = 0.0             # weight for heteroscedastic NLL loss term
    aux_day28_weight: float = 0.0       # optional auxiliary loss on the 28-day head
    use_day28_aux_head: bool = False    # if True, generator learns a direct day-28 shortcut head

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GanGeneratorConfig":
        return cls(
            hidden_layers=[int(v) for v in payload.get("hidden_layers", [128, 96, 64])],
            noise_dim=int(payload.get("noise_dim", 12)),
            dropout=float(payload.get("dropout", 0.08)),
            learning_rate=float(payload.get("learning_rate", 0.0015)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            batch_size=int(payload.get("batch_size", 64)),
            warmup_epochs=int(payload.get("warmup_epochs", 35)),
            epochs_per_round=int(payload.get("epochs_per_round", 24)),
            early_stopping_rounds=int(payload.get("early_stopping_rounds", 18)),
            supervised_weight=float(payload.get("supervised_weight", 1.0)),
            adversarial_weight=float(payload.get("adversarial_weight", 0.25)),
            mape_weight=float(payload.get("mape_weight", 0.0)),
            period_weights=[float(w) for w in payload.get("period_weights", [])],
            max_grad_norm=float(payload.get("max_grad_norm", 1.5)),
            use_uncertainty_head=bool(payload.get("use_uncertainty_head", False)),
            nll_weight=float(payload.get("nll_weight", 0.0)),
            aux_day28_weight=float(payload.get("aux_day28_weight", 0.0)),
            use_day28_aux_head=bool(payload.get("use_day28_aux_head", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": self.hidden_layers,
            "noise_dim": self.noise_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "warmup_epochs": self.warmup_epochs,
            "epochs_per_round": self.epochs_per_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "supervised_weight": self.supervised_weight,
            "adversarial_weight": self.adversarial_weight,
            "mape_weight": self.mape_weight,
            "period_weights": self.period_weights,
            "max_grad_norm": self.max_grad_norm,
            "use_uncertainty_head": self.use_uncertainty_head,
            "nll_weight": self.nll_weight,
            "aux_day28_weight": self.aux_day28_weight,
            "use_day28_aux_head": self.use_day28_aux_head,
        }


@dataclass
class GanDiscriminatorConfig:
    """Discriminator settings: NEAT topology search + BNN fitting."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    neat_config_path: str | None = None
    prior_std: float = 0.6
    posterior_scale_init: float = 0.08
    kl_weight: float = 0.02
    bnn_learning_rate: float = 0.006
    bnn_epochs: int = 80
    bnn_batch_size: int = 64
    bnn_validation_split: float = 0.2
    bnn_mc_samples: int = 30
    bnn_early_stopping_rounds: int = 18

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GanDiscriminatorConfig":
        return cls(
            optimizer=OptimizerConfig.from_dict(payload.get("optimizer", {})),
            neat_config_path=None if payload.get("neat_config_path") in (None, "") else str(payload.get("neat_config_path")),
            prior_std=float(payload.get("prior_std", 0.6)),
            posterior_scale_init=float(payload.get("posterior_scale_init", 0.08)),
            kl_weight=float(payload.get("kl_weight", 0.02)),
            bnn_learning_rate=float(payload.get("bnn_learning_rate", 0.006)),
            bnn_epochs=int(payload.get("bnn_epochs", 80)),
            bnn_batch_size=int(payload.get("bnn_batch_size", 64)),
            bnn_validation_split=float(payload.get("bnn_validation_split", 0.2)),
            bnn_mc_samples=int(payload.get("bnn_mc_samples", 30)),
            bnn_early_stopping_rounds=int(payload.get("bnn_early_stopping_rounds", 18)),
        )

    def resolve_paths(self, base_dir: Path) -> None:
        self.neat_config_path = _resolve_config_path(base_dir, self.neat_config_path)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "optimizer": self.optimizer.__dict__,
            "neat_config_path": self.neat_config_path,
            "prior_std": self.prior_std,
            "posterior_scale_init": self.posterior_scale_init,
            "kl_weight": self.kl_weight,
            "bnn_learning_rate": self.bnn_learning_rate,
            "bnn_epochs": self.bnn_epochs,
            "bnn_batch_size": self.bnn_batch_size,
            "bnn_validation_split": self.bnn_validation_split,
            "bnn_mc_samples": self.bnn_mc_samples,
            "bnn_early_stopping_rounds": self.bnn_early_stopping_rounds,
        }
        return payload


@dataclass
class GanTrainingConfig:
    """Global training loop controls."""

    rounds: int = 3
    validation_split: float = 0.2
    seed: int = 42
    generator_mc_samples: int = 36
    fake_jitter_std: float = 0.2

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GanTrainingConfig":
        return cls(
            rounds=int(payload.get("rounds", 3)),
            validation_split=float(payload.get("validation_split", 0.2)),
            seed=int(payload.get("seed", 42)),
            generator_mc_samples=int(payload.get("generator_mc_samples", 36)),
            fake_jitter_std=float(payload.get("fake_jitter_std", 0.2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "validation_split": self.validation_split,
            "seed": self.seed,
            "generator_mc_samples": self.generator_mc_samples,
            "fake_jitter_std": self.fake_jitter_std,
        }


@dataclass
class GanStageConfig:
    """Root config for train_gan stage."""

    dataset: GanDatasetConfig
    physics: GanPhysicsConfig = field(default_factory=GanPhysicsConfig)
    generator: GanGeneratorConfig = field(default_factory=GanGeneratorConfig)
    discriminator: GanDiscriminatorConfig = field(default_factory=GanDiscriminatorConfig)
    training: GanTrainingConfig = field(default_factory=GanTrainingConfig)
    gost_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GanStageConfig":
        dataset = GanDatasetConfig.from_dict(payload.get("dataset", payload.get("gan_input", {})))
        config = cls(
            dataset=dataset,
            physics=GanPhysicsConfig.from_dict(payload.get("physics", {})),
            generator=GanGeneratorConfig.from_dict(payload.get("generator", {})),
            discriminator=GanDiscriminatorConfig.from_dict(payload.get("discriminator", {})),
            training=GanTrainingConfig.from_dict(payload.get("training", {})),
            gost_path=None if payload.get("gost_path") in (None, "") else str(payload.get("gost_path")),
        )
        config.validate()
        return config

    def resolve_paths(self, base_dir: Path) -> None:
        self.dataset.resolve_paths(base_dir)
        self.discriminator.resolve_paths(base_dir)
        self.gost_path = _resolve_config_path(base_dir, self.gost_path)

    def validate(self) -> None:
        self.dataset.validate()
        if self.training.rounds < 1:
            raise ValueError("training.rounds must be at least 1")
        if self.generator.batch_size < 1:
            raise ValueError("generator.batch_size must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "physics": self.physics.to_dict(),
            "generator": self.generator.to_dict(),
            "discriminator": self.discriminator.to_dict(),
            "training": self.training.to_dict(),
            "gost_path": self.gost_path,
        }


@dataclass
class PreparedStrengthDataset:
    """Wide-format multi-period strength dataset (one row per mix)."""

    components: np.ndarray          # (N, C) raw component values
    strengths: np.ndarray           # (N, P) strength values per curing period
    target_weights: np.ndarray      # (N, P) reliability / supervision weights per target
    features: np.ndarray            # (N, F) feature matrix for generator input
    component_names: list[str]
    feature_names: list[str]
    period_days: list[float]        # sorted curing ages, e.g. [1.0, 3.0, 7.0, 28.0]
    component_bounds: dict[str, list[float]]
    source_labels: np.ndarray | None = None  # (N,) optional source labels for stratified splitting


class AutoregressiveGenerator(nn.Module):
    """Autoregressive generator predicting a full strength curve [y_1, y_3, y_7, y_28].

    Architecture:
        1. Encoder  : (composition_features ∥ z) → shared representation
        2. Base head: shared_repr → ŷ_1  (first curing period, unconstrained)
        3. Delta heads (one per subsequent period k):
               delta_k = DeltaHead_k(shared_repr ∥ ŷ_{k-1})
               ŷ_k     = ŷ_{k-1} + softplus(delta_k)
           Monotonicity ŷ_1 ≤ ŷ_3 ≤ … ≤ ŷ_P is guaranteed architecturally
           because softplus > 0 always — no penalty term required.

    Returns tensor of shape (N, n_periods) in the scaled target space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        noise_dim: int,
        dropout: float,
        n_periods: int = 4,
        use_uncertainty_head: bool = False,
        use_day28_aux_head: bool = False,
    ) -> None:
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.n_periods = int(n_periods)
        self.use_uncertainty_head = bool(use_uncertainty_head)
        self.use_day28_aux_head = bool(use_day28_aux_head)

        def _make_mlp(in_dim: int, widths: list[int], out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_dim
            for w in widths:
                layers.append(nn.Linear(prev, int(w)))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
                prev = int(w)
            layers.append(nn.Linear(prev, out_dim))
            return nn.Sequential(*layers)

        enc_in = input_dim + self.noise_dim
        self.encoder = _make_mlp(enc_in, hidden_layers, hidden_layers[-1])

        enc_out = hidden_layers[-1]
        head_hidden = max(32, enc_out // 4)

        # Predicts the first (unconstrained) strength value
        self.base_head = nn.Linear(enc_out, 1)

        # Each delta head takes (shared_repr ∥ ŷ_{prev}) and predicts a positive increment
        self.delta_heads = nn.ModuleList([
            _make_mlp(enc_out + 1, [head_hidden], 1)
            for _ in range(self.n_periods - 1)
        ])

        # Optional heteroscedastic log-variance heads (one per period)
        if self.use_uncertainty_head:
            self.log_var_base_head: nn.Linear | None = nn.Linear(enc_out, 1)
            self.log_var_delta_heads: nn.ModuleList | None = nn.ModuleList([
                nn.Linear(enc_out, 1) for _ in range(self.n_periods - 1)
            ])
        else:
            self.log_var_base_head = None
            self.log_var_delta_heads = None

        # Optional direct shortcut for late-age strength. This gives the model a
        # separate path to specialize on 28-day behavior without changing the
        # monotone autoregressive curve.
        self.day28_aux_head: nn.Linear | None = nn.Linear(enc_out, 1) if self.use_day28_aux_head else None

    def forward(
        self, features: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.noise_dim > 0:
            if noise is None:
                noise = torch.randn(features.shape[0], self.noise_dim, device=features.device, dtype=features.dtype)
            fused = torch.cat([features, noise], dim=1)
        else:
            fused = features

        shared = self.encoder(fused)        # (N, enc_out)
        y_prev = self.base_head(shared)     # (N, 1)
        outputs: list[torch.Tensor] = [y_prev]

        for delta_head in self.delta_heads:
            delta = F.softplus(delta_head(torch.cat([shared, y_prev], dim=1)))
            y_prev = y_prev + delta
            outputs.append(y_prev)

        mean_out = torch.cat(outputs, dim=1)    # (N, n_periods)
        day28_aux_out = self.day28_aux_head(shared) if self.day28_aux_head is not None else None

        if self.use_uncertainty_head and self.log_var_base_head is not None and self.log_var_delta_heads is not None:
            lv_base = self.log_var_base_head(shared)   # (N, 1)
            lv_outputs: list[torch.Tensor] = [lv_base]
            for lv_head in self.log_var_delta_heads:
                lv_outputs.append(lv_head(shared))     # (N, 1)
            log_var_out = torch.clamp(torch.cat(lv_outputs, dim=1), -6.0, 2.0)  # (N, n_periods)
            if day28_aux_out is not None:
                return mean_out, log_var_out, day28_aux_out
            return mean_out, log_var_out

        if day28_aux_out is not None:
            return mean_out, day28_aux_out

        return mean_out


class DeterministicBnnDiscriminator(nn.Module):
    """Deterministic discriminator extracted from NEAT-BNN posterior means."""

    def __init__(self, regressor: NeatBNNRegressor) -> None:
        super().__init__()
        if regressor.property_scaler is None:
            raise RuntimeError("Discriminator regressor must be fitted before export")

        store = pyro.get_param_store()
        self.n_layers = regressor.model.n_layers

        self.register_buffer("input_mean", torch.as_tensor(regressor.property_scaler.mean, dtype=torch.float32))
        self.register_buffer("input_scale", torch.as_tensor(regressor.property_scaler.scale, dtype=torch.float32))
        self.register_buffer("bounds_lower", regressor.model.bounds_lower.detach().clone().float())
        self.register_buffer("bounds_upper", regressor.model.bounds_upper.detach().clone().float())
        self.register_buffer("output_indices", regressor.model.output_indices.detach().clone().long())

        for layer_idx in range(self.n_layers):
            weight_loc_name = f"layer_{layer_idx}.weight_loc"
            bias_loc_name = f"layer_{layer_idx}.bias_loc"
            if weight_loc_name not in store or bias_loc_name not in store:
                raise RuntimeError("Expected variational parameters are missing in pyro param store")

            self.register_buffer(
                f"weight_loc_{layer_idx}",
                store[weight_loc_name].detach().clone().float(),
            )
            self.register_buffer(
                f"bias_loc_{layer_idx}",
                store[bias_loc_name].detach().clone().float(),
            )
            self.register_buffer(
                f"mask_{layer_idx}",
                getattr(regressor.model, f"mask_{layer_idx}").detach().clone().float(),
            )
            self.register_buffer(
                f"response_{layer_idx}",
                getattr(regressor.model, f"response_{layer_idx}").detach().clone().float(),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scaled_input = (values - self.input_mean) / self.input_scale
        activations = scaled_input

        for layer_idx in range(self.n_layers):
            weight = getattr(self, f"weight_loc_{layer_idx}") * getattr(self, f"mask_{layer_idx}")
            bias = getattr(self, f"bias_loc_{layer_idx}")
            response = getattr(self, f"response_{layer_idx}")
            pre_activation = bias + response * (activations @ weight.T)
            layer_out = torch.tanh(pre_activation)
            activations = torch.cat([activations, layer_out], dim=1)

        output_tanh = activations[:, self.output_indices]
        decoded = self.bounds_lower + ((output_tanh + 1.0) / 2.0) * (self.bounds_upper - self.bounds_lower)
        return torch.clamp(decoded, 0.0, 1.0)


def load_gan_config(path: str | Path) -> GanStageConfig:
    """Read and validate train_gan JSON config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = GanStageConfig.from_dict(payload)
    config.resolve_paths(config_path.resolve().parent)
    return config


def _as_numeric(series: pd.Series) -> pd.Series:
    stringified = series.astype(str)
    cleaned = (
        stringified
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _extract_component_frame(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"Component columns not found: {', '.join(missing)}")
    return pd.DataFrame({name: _as_numeric(frame[name]) for name in names})


def _prepare_strength_dataset(config: GanStageConfig) -> PreparedStrengthDataset:
    """Build a wide-format dataset: one row per mix, P columns of strength targets.

    Requires strength_by_day to be defined (multi-output mode only supports wide format).
    Rows where any curing-period column is missing or non-positive are dropped.
    log(t) is NOT added as a feature: the autoregressive generator predicts all periods
    simultaneously from composition features alone.
    """
    if not config.dataset.strength_by_day:
        raise ValueError(
            "Multi-output mode requires strength_by_day to list all curing periods."
        )

    frame = read_dataset_frame(config.dataset.data_path)
    if config.dataset.component_aliases:
        frame = frame.rename(columns=config.dataset.component_aliases)

    component_frame = _extract_component_frame(frame, config.dataset.components)
    component_values = component_frame.to_numpy(dtype=float)

    # Sort periods numerically so columns are ordered [y_1, y_3, y_7, y_28]
    period_days_raw = sorted(config.dataset.strength_by_day.keys(), key=float)
    period_days = [float(d) for d in period_days_raw]

    strength_columns: list[np.ndarray] = []
    strength_weight_columns: list[np.ndarray] = []
    for day_raw in period_days_raw:
        strength_col = config.dataset.strength_by_day[day_raw]
        if strength_col not in frame.columns:
            raise KeyError(f"Strength column not found: {strength_col}")
        strength_columns.append(_as_numeric(frame[strength_col]).to_numpy(dtype=float))
        weight_col = f"{strength_col}_weight"
        if weight_col in frame.columns:
            strength_weight_columns.append(_as_numeric(frame[weight_col]).to_numpy(dtype=float))
        else:
            strength_weight_columns.append(np.ones(len(frame), dtype=float))

    strengths_matrix = np.column_stack(strength_columns)  # (N_raw, P)
    target_weights_matrix = np.column_stack(strength_weight_columns)  # (N_raw, P)

    finite_components = np.isfinite(component_values).all(axis=1)
    finite_strengths = np.isfinite(strengths_matrix).all(axis=1)
    positive_strengths = (strengths_matrix > 0).all(axis=1)
    finite_weights = np.isfinite(target_weights_matrix).all(axis=1)
    positive_weight_sum = (target_weights_matrix.sum(axis=1) > 0)
    valid_mask = finite_components & finite_strengths & positive_strengths & finite_weights & positive_weight_sum

    if not valid_mask.any():
        raise ValueError(
            "No rows with all components and all curing periods present and positive."
        )

    components = component_values[valid_mask]
    strengths = strengths_matrix[valid_mask]   # (N, P)
    target_weights = np.clip(target_weights_matrix[valid_mask], 0.0, 1.0)
    source_labels = frame["source"].astype(str).to_numpy()[valid_mask] if "source" in frame.columns else None

    if components.shape[0] < 12:
        raise ValueError("Too few valid samples after preprocessing; need at least 12 rows")

    feature_blocks: list[np.ndarray] = [components]
    feature_names: list[str] = list(config.dataset.components)

    # add_log_time_feature is intentionally ignored: time is not an input in multi-output mode
    if config.physics.add_abrams_features:
        cement_index = (
            feature_names.index(config.physics.cement_feature)
            if config.physics.cement_feature in feature_names
            else None
        )
        water_index = (
            feature_names.index(config.physics.water_feature)
            if config.physics.water_feature in feature_names
            else None
        )
        if cement_index is not None and water_index is not None:
            cement = components[:, cement_index]
            water = components[:, water_index]
            wc_ratio = water / np.clip(cement, 1e-4, None)
            cw_ratio = cement / np.clip(water, 1e-4, None)
            feature_blocks.append(wc_ratio[:, None])
            feature_blocks.append(cw_ratio[:, None])
            feature_names.extend(["water_cement_ratio", "cement_water_ratio"])

            # Extended domain features (inductive bias from cement technology)
            if config.physics.add_extended_features:
                # log(c/w) — direct form of Abrams law: f28 ≈ A * log(c/w) + B
                log_cw = np.log(np.clip(cw_ratio, 1e-4, None))
                feature_blocks.append(log_cw[:, None])
                feature_names.append("log_cw_ratio")

                # Pozzolan-adjusted effective w/c: accounts for fly_ash and microsilica
                fa_index = feature_names.index("fly_ash") if "fly_ash" in feature_names else None
                ms_index = (
                    feature_names.index("microsilica_kg")
                    if "microsilica_kg" in feature_names
                    else None
                )
                pl_index = (
                    feature_names.index("plasticizer_kg")
                    if "plasticizer_kg" in feature_names
                    else None
                )
                sand_index = feature_names.index("sand") if "sand" in feature_names else None
                gravel_index = feature_names.index("gravel") if "gravel" in feature_names else None

                fly_ash = components[:, fa_index] if fa_index is not None else np.zeros(components.shape[0])
                microsilica = components[:, ms_index] if ms_index is not None else np.zeros(components.shape[0])
                binder = cement + 0.4 * fly_ash + microsilica  # effective binder, CEM-equivalent

                # Effective w/c (ACI/EN method for blended cements)
                eff_wc = water / np.clip(binder, 1e-4, None)
                feature_blocks.append(eff_wc[:, None])
                feature_names.append("effective_wc_ratio")

                # Aggregate-to-binder ratio — governs packing density and workability
                if sand_index is not None and gravel_index is not None:
                    sand_vals = components[:, sand_index]
                    gravel_vals = components[:, gravel_index]
                    agg_binder = (sand_vals + gravel_vals) / np.clip(binder, 1e-4, None)
                    feature_blocks.append(agg_binder[:, None])
                    feature_names.append("aggregate_binder_ratio")

                # Plasticizer dosage rate (% by mass of binder) — affects early strength gain
                if pl_index is not None:
                    pl_vals = components[:, pl_index]
                    pl_binder_pct = pl_vals / np.clip(binder, 1e-4, None) * 100.0
                    feature_blocks.append(pl_binder_pct[:, None])
                    feature_names.append("plasticizer_binder_pct")

                # Late-age specific descriptors: useful when Day-28 quality is the priority.
                if config.physics.add_late_age_features:
                    water_binder = water / np.clip(binder, 1e-4, None)
                    pozzolan_share = (fly_ash + microsilica) / np.clip(binder, 1e-4, None)
                    binder_efficiency = cement / np.clip(binder, 1e-4, None)
                    feature_blocks.append(water_binder[:, None])
                    feature_names.append("water_binder_ratio")
                    feature_blocks.append(pozzolan_share[:, None])
                    feature_names.append("pozzolan_share")
                    feature_blocks.append(binder_efficiency[:, None])
                    feature_names.append("binder_efficiency")

    total_mass = components.sum(axis=1, keepdims=True)
    feature_blocks.append(total_mass)
    feature_names.append("total_mass")

    features = np.hstack(feature_blocks)

    component_stats = pd.DataFrame(components, columns=config.dataset.components).agg(["min", "max"]).transpose()

    return PreparedStrengthDataset(
        components=components,
        strengths=strengths,
        target_weights=target_weights,
        features=features,
        component_names=list(config.dataset.components),
        feature_names=feature_names,
        period_days=period_days,
        component_bounds={
            name: [float(component_stats.loc[name, "min"]), float(component_stats.loc[name, "max"])]
            for name in config.dataset.components
        },
        source_labels=source_labels,
    )


def _split_indices(
    n_samples: int,
    validation_split: float,
    seed: int,
    *,
    strengths_28: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if strengths_28 is None and source_labels is None:
        indices = rng.permutation(n_samples)
        val_size = max(1, int(n_samples * validation_split))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        if train_idx.size == 0:
            train_idx = val_idx
        return train_idx, val_idx

    if strengths_28 is None:
        strengths_28 = np.zeros(n_samples, dtype=float)
    if source_labels is None:
        source_labels = np.array(["default"] * n_samples, dtype=object)

    strengths_28 = np.asarray(strengths_28, dtype=float)
    source_labels = np.asarray(source_labels, dtype=object)
    finite_mask = np.isfinite(strengths_28)
    finite_values = strengths_28[finite_mask]
    if finite_values.size >= 8:
        quantiles = np.quantile(finite_values, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(strengths_28, quantiles, right=True)
    else:
        bins = np.zeros(n_samples, dtype=int)

    val_parts: list[np.ndarray] = []
    all_indices = np.arange(n_samples)
    for src in np.unique(source_labels):
        src_mask = source_labels == src
        for b in np.unique(bins[src_mask]):
            group_idx = all_indices[src_mask & (bins == b)]
            if group_idx.size == 0:
                continue
            group_idx = rng.permutation(group_idx)
            group_val_size = max(1, int(round(group_idx.size * validation_split))) if group_idx.size > 3 else 1
            group_val_size = min(group_val_size, max(1, group_idx.size - 1)) if group_idx.size > 1 else 1
            val_parts.append(group_idx[:group_val_size])

    val_idx = np.unique(np.concatenate(val_parts)) if val_parts else rng.permutation(n_samples)[: max(1, int(n_samples * validation_split))]
    train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
    if train_idx.size == 0:
        train_idx = val_idx
    return train_idx, val_idx


def _make_fake_strength(strengths_scaled: np.ndarray, rng: np.random.Generator, jitter_std: float) -> np.ndarray:
    """Shuffle rows and add noise; works for both (N,) and (N, P) arrays."""
    shuffled = strengths_scaled[rng.permutation(strengths_scaled.shape[0])]
    jitter = rng.normal(0.0, jitter_std, size=strengths_scaled.shape)
    return shuffled + jitter


def _make_discriminator_dataset(
    feature_scaled: np.ndarray,
    strengths_scaled_real: np.ndarray,   # (N, P)
    strengths_scaled_fake: np.ndarray,   # (N, P)
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    real_input = np.hstack([feature_scaled, strengths_scaled_real])
    fake_input = np.hstack([feature_scaled, strengths_scaled_fake])

    x_values = np.vstack([real_input, fake_input])
    y_values = np.concatenate([
        np.ones(real_input.shape[0], dtype=float),
        np.zeros(fake_input.shape[0], dtype=float),
    ])

    perm = rng.permutation(x_values.shape[0])
    return x_values[perm], y_values[perm][:, None]


def _fit_discriminator_bnn(
    regressor: NeatBNNRegressor,
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: GanStageConfig,
    seed: int,
) -> dict[str, Any]:
    return regressor.fit(
        x_values,
        y_values,
        learning_rate=config.discriminator.bnn_learning_rate,
        epochs=config.discriminator.bnn_epochs,
        batch_size=config.discriminator.bnn_batch_size,
        validation_split=config.discriminator.bnn_validation_split,
        mc_samples=config.discriminator.bnn_mc_samples,
        early_stopping_rounds=config.discriminator.bnn_early_stopping_rounds,
        seed=seed,
    )


def _load_optional_neat_config(artifact: dict[str, str], metadata: dict[str, Any]):
    if metadata.get("algorithm") != "python-neat":
        return None
    config_path = artifact.get("config")
    if not config_path:
        return None
    config_file = Path(config_path)
    if not config_file.exists():
        return None
    import neat

    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_file),
    )


def _write_gan_discriminator_neat_override(
    destination: Path,
    optimizer_config: OptimizerConfig,
) -> Path:
    """Write an INI overlay so JSON optimizer values are honored consistently.

    NEATOptimizer merges defaults from neat.ini before applying stage settings.
    This overlay keeps the GAN stage deterministic and allows setting algorithm,
    population size, and generation limits from gan.json.
    """

    algorithm = optimizer_config.algorithm or "python-neat"
    payload = f"""
[General]
limit_generations = {int(optimizer_config.limit_generations)}

[NEATEST]
algorithm = {algorithm}
pop_size = {int(optimizer_config.pop_size)}
es_population = {int(optimizer_config.es_population)}
sigma = {float(optimizer_config.sigma)}
seed = {int(optimizer_config.seed) if optimizer_config.seed is not None else 42}
elite_rate = {float(optimizer_config.elite_rate)}
use_bias = {str(bool(optimizer_config.use_bias))}
optimizer_lr = {float(optimizer_config.optimizer_lr)}

[Topology]
node_mutation_rate = {float(optimizer_config.node_mutation_rate)}
connection_mutation_rate = {float(optimizer_config.connection_mutation_rate)}
disable_connection_mutation_rate = {float(optimizer_config.disable_connection_mutation_rate)}
dominant_gene_rate = {float(optimizer_config.dominant_gene_rate)}
dominant_gene_delta = {float(optimizer_config.dominant_gene_delta)}
hidden_activation = {optimizer_config.hidden_activation}
output_activation = {optimizer_config.output_activation}

[BNEATEST]
sigma_prior = {float(optimizer_config.sigma_prior)}
kl_weight = {float(optimizer_config.kl_weight)}
kl_warmup_steps = {int(optimizer_config.kl_warmup_steps)}
initial_rho = {float(optimizer_config.initial_rho)}
n_eval_samples = {int(optimizer_config.n_eval_samples)}
risk_aversion = {float(optimizer_config.risk_aversion)}

[NEAT]
pop_size = {int(optimizer_config.pop_size)}
seed = {int(optimizer_config.seed) if optimizer_config.seed is not None else 42}
""".strip()

    destination.write_text(payload + "\n", encoding="utf-8")
    return destination


def _build_discriminator(
    feature_names: list[str],
    strength_names: list[str],
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: GanStageConfig,
    output_dir: Path,
) -> tuple[NeatBNNRegressor, dict[str, Any], dict[str, Any]]:
    neat_dir = output_dir / "discriminator_neat"
    neat_dir.mkdir(parents=True, exist_ok=True)

    optimizer_config = copy.deepcopy(config.discriminator.optimizer)
    if optimizer_config.algorithm in (None, ""):
        optimizer_config.algorithm = "python-neat"

    if config.discriminator.neat_config_path:
        neat_config_path = config.discriminator.neat_config_path
    else:
        neat_config_path = str(
            _write_gan_discriminator_neat_override(
                neat_dir / "gan_neat_override.ini",
                optimizer_config,
            )
        )

    optimizer = NEATOptimizer(
        input_size=x_values.shape[1],
        output_size=1,
        config=optimizer_config,
        bounds_lower=np.asarray([0.0], dtype=float),
        bounds_upper=np.asarray([1.0], dtype=float),
        input_names=feature_names + strength_names,
        output_names=["realness"],
    )

    neat_result = optimizer.optimize(
        properties_scaled=x_values,
        target_components=y_values,
        top_k=1,
        artifacts_dir=str(neat_dir),
        neat_config_path=neat_config_path,
    )

    if not neat_result.get("network_artifacts"):
        raise RuntimeError("Discriminator topology search produced no network artifact")

    artifact = neat_result["network_artifacts"][0]
    metadata_path = Path(artifact.get("metadata", ""))
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    genome_path = Path(artifact["genome"])
    with genome_path.open("rb") as handle:
        import cloudpickle

        genome = cloudpickle.load(handle)

    neat_config = _load_optional_neat_config(artifact, metadata)
    regressor = build_regressor_from_genome(
        genome,
        bounds_lower=np.asarray([0.0], dtype=float),
        bounds_upper=np.asarray([1.0], dtype=float),
        input_names=feature_names + strength_names,
        output_names=["realness"],
        prior_std=config.discriminator.prior_std,
        posterior_scale_init=config.discriminator.posterior_scale_init,
        kl_weight=config.discriminator.kl_weight,
        seed=config.training.seed,
        neat_config=neat_config,
    )

    return regressor, neat_result, metadata


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    non_zero = np.abs(y_true) > 1e-8
    if np.any(non_zero):
        mape = float(np.mean(np.abs(residual[non_zero] / y_true[non_zero])) * 100.0)
    else:
        mape = float("nan")

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def _weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    clipped_weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total_weight = float(clipped_weights.sum())
    if total_weight <= 1e-12:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
        }

    residual = y_true - y_pred
    mae = float(np.sum(np.abs(residual) * clipped_weights) / total_weight)
    rmse = float(np.sqrt(np.sum((residual ** 2) * clipped_weights) / total_weight))

    non_zero = (np.abs(y_true) > 1e-8) & (clipped_weights > 0.0)
    if np.any(non_zero):
        mape = float(
            np.sum(np.abs(residual[non_zero] / y_true[non_zero]) * clipped_weights[non_zero])
            / np.sum(clipped_weights[non_zero])
            * 100.0
        )
    else:
        mape = float("nan")

    weighted_mean = float(np.sum(y_true * clipped_weights) / total_weight)
    ss_res = float(np.sum((residual ** 2) * clipped_weights))
    ss_tot = float(np.sum(((y_true - weighted_mean) ** 2) * clipped_weights))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def _metrics_by_period(
    y_true: np.ndarray,       # (N, P)
    y_pred: np.ndarray,       # (N, P)
    period_days: list[float],
) -> dict[str, dict[str, float]]:
    """Compute regression metrics per curing period (column-wise)."""
    return {
        f"day_{int(day)}": {**_metrics(y_true[:, i], y_pred[:, i]), "n_samples": int(y_true.shape[0])}
        for i, day in enumerate(period_days)
    }


def _weighted_metrics_by_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    period_days: list[float],
) -> dict[str, dict[str, float]]:
    return {
        f"day_{int(day)}": {
            **_weighted_metrics(y_true[:, i], y_pred[:, i], weights[:, i]),
            "n_samples": int(y_true.shape[0]),
            "weight_sum": float(np.clip(weights[:, i], 0.0, None).sum()),
        }
        for i, day in enumerate(period_days)
    }


def _mc_predict_generator(
    generator: AutoregressiveGenerator,
    x_values: torch.Tensor,
    target_scaler: StandardScaler,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC predict; returns (mean_raw, std_raw, mean_scaled) each of shape (N, P)."""
    samples_scaled: list[torch.Tensor] = []
    aleatoric_var_scaled: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(max(1, int(mc_samples))):
            out = generator(x_values)
            if isinstance(out, tuple):
                mean_t = out[0]
                log_var_t = out[1] if len(out) > 1 and isinstance(out[1], torch.Tensor) and out[1].shape == mean_t.shape else None
                samples_scaled.append(mean_t)           # (N, P)
                if log_var_t is not None:
                    aleatoric_var_scaled.append(log_var_t.exp())  # (N, P)
            else:
                samples_scaled.append(out)              # (N, P)

    stacked = torch.stack(samples_scaled, dim=0)          # (mc, N, P)
    mean_scaled_t = stacked.mean(dim=0)                   # (N, P), float32
    epistemic_var_t = stacked.var(dim=0)                  # (N, P)

    if aleatoric_var_scaled:
        mean_aleatoric_t = torch.stack(aleatoric_var_scaled, dim=0).mean(dim=0)  # (N, P)
        total_var_t = epistemic_var_t + mean_aleatoric_t
    else:
        total_var_t = epistemic_var_t
    std_scaled_t = total_var_t.sqrt()                     # (N, P)

    scale = float(target_scaler.scale[0])
    mean_offset = float(target_scaler.mean[0])

    # Convert via Python lists to avoid numpy dtype corruption issues when
    # mixing PyTorch-created arrays with certain numpy/torch version combos.
    mean_raw: np.ndarray = np.array(mean_scaled_t.tolist(), dtype=np.float64) * scale + mean_offset
    std_raw: np.ndarray = np.array(std_scaled_t.tolist(), dtype=np.float64) * scale
    mean_scaled: np.ndarray = np.array(mean_scaled_t.tolist(), dtype=np.float32)
    return mean_raw, std_raw, mean_scaled


def predict_loglinear_at_time(
    mean_raw: np.ndarray,
    std_raw: np.ndarray,
    period_days: list[float],
    t_new_days: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict strength at arbitrary time points using a log-linear fit.

    Concrete strength follows strength(t) ≈ a * log(t) + b (Abrams / RILEM).
    Given MC-mean and MC-std at the known *period_days*, fit per-sample
    log-linear regression and extrapolate to *t_new_days*.

    Parameters
    ----------
    mean_raw : (N, P)  MC mean in MPa at training periods
    std_raw  : (N, P)  MC std  in MPa at training periods
    period_days : list of P floats — training curing days
    t_new_days  : list of T floats — target curing days

    Returns
    -------
    mean_new : (N, T)  predicted mean strength in MPa
    std_new  : (N, T)  propagated std (first-order)
    """
    log_t = np.log(np.asarray(period_days, dtype=np.float64))     # (P,)
    log_t_new = np.log(np.asarray(t_new_days, dtype=np.float64))  # (T,)

    # Design matrix for log-linear fit: [log_t | 1]  shape (P, 2)
    A = np.column_stack([log_t, np.ones_like(log_t)])              # (P, 2)
    AtA_inv_At = np.linalg.lstsq(A, np.eye(len(log_t)), rcond=None)[0]  # (2, P)

    # Fit coefficients per sample: (N, 2) = (N, P) @ (P, 2)
    coeffs = mean_raw @ AtA_inv_At.T   # (N, 2)

    # Design matrix for new times: (T, 2)
    A_new = np.column_stack([log_t_new, np.ones_like(log_t_new)])  # (T, 2)

    # Predicted mean: (N, 2) @ (2, T) → (N, T)
    mean_new = coeffs @ A_new.T

    # Propagate std: var_new_i ≈ sum_p (dŷ_new/dy_p)^2 * std_p^2
    # dŷ_new/dy_p = (AtA⁻¹Aᵀ)_0p * log(t_new) + (AtA⁻¹Aᵀ)_1p
    # = A_new @ AtA_inv_At[:, p]   — outer product approach
    H = A_new @ AtA_inv_At         # (T, P)  — "hat" row for each new time
    var_new = (std_raw**2) @ H.T**2   # (N, T)
    std_new = np.sqrt(np.clip(var_new, 0.0, None))

    return mean_new, std_new


def _train_generator_epochs(
    generator: AutoregressiveGenerator,
    optimizer: torch.optim.Optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,           # (N_train, P)
    target_weights_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val_raw: np.ndarray,           # (N_val, P) in original MPa units
    target_scaler: StandardScaler,
    *,
    batch_size: int,
    epochs: int,
    early_stopping_rounds: int,
    mc_samples: int,
    supervised_weight: float,
    adversarial_weight: float,
    mape_weight: float,
    period_weights: list[float],
    abrams_weight: float,
    nll_weight: float,
    aux_day28_weight: float,
    wc_feature_index: int | None,
    discriminator: DeterministicBnnDiscriminator | None,
    max_grad_norm: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    train_loss_history: list[float] = []
    val_rmse_history: list[float] = []
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Any] | None = None
    best_rmse = float("inf")
    epochs_without_improvement = 0

    for _ in range(epochs):
        permutation = torch.randperm(x_train.shape[0])
        total_epoch_loss = 0.0
        total_batches = 0

        for start in range(0, x_train.shape[0], batch_size):
            stop = start + batch_size
            batch_idx = permutation[start:stop]
            batch_x = x_train[batch_idx].clone()
            batch_y = y_train[batch_idx]            # (B, P)
            batch_w = target_weights_train[batch_idx]  # (B, P)
            batch_x.requires_grad_(True)

            gen_out = generator(batch_x)
            if isinstance(gen_out, tuple):
                prediction = gen_out[0]  # (B, P)
                log_var_pred = gen_out[1] if len(gen_out) > 1 and isinstance(gen_out[1], torch.Tensor) and gen_out[1].shape == prediction.shape else None
                aux_day28_pred = gen_out[-1] if len(gen_out) > 2 and isinstance(gen_out[-1], torch.Tensor) and gen_out[-1].shape == prediction[:, -1:].shape else None
            else:
                prediction = gen_out  # (B, P)
                log_var_pred = None
                aux_day28_pred = None

            # Per-period weighted SmoothL1 loss
            n_periods = prediction.shape[1]
            if period_weights and len(period_weights) == n_periods:
                pw = torch.tensor(period_weights, dtype=prediction.dtype, device=prediction.device)
                pw = pw / pw.sum() * n_periods          # normalise so sum = n_periods
                per_period = torch.stack([
                    ((F.smooth_l1_loss(prediction[:, k], batch_y[:, k], reduction="none") * batch_w[:, k]).sum()
                     / torch.clamp(batch_w[:, k].sum(), min=1e-6))
                    for k in range(n_periods)
                ])
                supervised_loss = (pw * per_period).mean()
            else:
                supervised_loss = ((F.smooth_l1_loss(prediction, batch_y, reduction="none") * batch_w).sum()
                                   / torch.clamp(batch_w.sum(), min=1e-6))

            adversarial_loss = torch.tensor(0.0, dtype=prediction.dtype, device=prediction.device)
            if discriminator is not None and adversarial_weight > 0.0:
                disc_input = torch.cat([batch_x, prediction], dim=1)  # (B, F+P)
                disc_score = discriminator(disc_input)
                target = torch.full_like(disc_score, 0.9)
                adversarial_loss = F.binary_cross_entropy(torch.clamp(disc_score, 1e-4, 1.0 - 1e-4), target)

            # Smooth MAPE loss in original MPa space (floor +1 MPa prevents div/0)
            smooth_mape_loss = torch.tensor(0.0, dtype=prediction.dtype, device=prediction.device)
            if mape_weight > 0.0:
                t_mean = float(target_scaler.mean[0])
                t_scale = float(target_scaler.scale[0])
                pred_mpa = prediction * t_scale + t_mean
                y_mpa = batch_y * t_scale + t_mean
                per_period_mape = torch.abs(pred_mpa - y_mpa) / (torch.abs(y_mpa) + 1.0)  # (B, P)
                if period_weights and len(period_weights) == prediction.shape[1]:
                    pw = torch.tensor(period_weights, dtype=prediction.dtype, device=prediction.device)
                    pw = pw / pw.sum() * prediction.shape[1]
                    weighted_period_mape = (per_period_mape * batch_w).sum(dim=0) / torch.clamp(batch_w.sum(dim=0), min=1e-6)
                    smooth_mape_loss = (pw * weighted_period_mape).mean()
                else:
                    smooth_mape_loss = (per_period_mape * batch_w).sum() / torch.clamp(batch_w.sum(), min=1e-6)

            # Abrams law: d(y_28)/d(w/c) <= 0
            # Monotonicity across periods is guaranteed architecturally via softplus.
            physics_penalty = torch.tensor(0.0, dtype=prediction.dtype, device=prediction.device)
            if wc_feature_index is not None and abrams_weight > 0.0:
                gradients = torch.autograd.grad(
                    prediction[:, -1].sum(), batch_x, create_graph=True, retain_graph=True,
                )[0]
                abrams_penalty = torch.relu(gradients[:, wc_feature_index]).mean()
                physics_penalty = physics_penalty + abrams_weight * abrams_penalty

            # Heteroscedastic NLL loss (optional)
            nll_loss = torch.tensor(0.0, dtype=prediction.dtype, device=prediction.device)
            if nll_weight > 0.0 and log_var_pred is not None:
                nll_elementwise = 0.5 * (batch_y - prediction) ** 2 / log_var_pred.exp() + 0.5 * log_var_pred
                nll_loss = (nll_elementwise * batch_w).sum() / torch.clamp(batch_w.sum(), min=1e-6)

            aux_day28_loss = torch.tensor(0.0, dtype=prediction.dtype, device=prediction.device)
            if aux_day28_weight > 0.0 and aux_day28_pred is not None:
                aux_l = F.smooth_l1_loss(aux_day28_pred.squeeze(1), batch_y[:, -1], reduction="none")
                aux_day28_loss = (aux_l * batch_w[:, -1]).sum() / torch.clamp(batch_w[:, -1].sum(), min=1e-6)

            loss = (supervised_weight * supervised_loss
                    + adversarial_weight * adversarial_loss
                    + mape_weight * smooth_mape_loss
                    + physics_penalty
                    + nll_weight * nll_loss
                    + aux_day28_weight * aux_day28_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
            optimizer.step()

            total_epoch_loss += float(loss.item())
            total_batches += 1

        train_loss_history.append(total_epoch_loss / max(1, total_batches))

        val_mean_raw, _val_std_raw, _ = _mc_predict_generator(generator, x_val, target_scaler, mc_samples=mc_samples)
        # Early stopping on overall RMSE across all periods
        overall_rmse = float(np.sqrt(np.mean((y_val_raw - val_mean_raw) ** 2)))
        val_rmse_history.append(overall_rmse)

        if overall_rmse + 1e-8 < best_rmse:
            best_rmse = overall_rmse
            best_metrics = _metrics(y_val_raw.ravel(), val_mean_raw.ravel())
            best_state = copy.deepcopy(generator.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_rounds:
                break

    if best_state is not None:
        generator.load_state_dict(best_state)

    return {
        "train_loss_history": train_loss_history,
        "val_rmse_history": val_rmse_history,
        "best_metrics": best_metrics or {},
        "epochs_run": len(train_loss_history),
    }


def _load_gost_strength_range(path: str | Path | None) -> tuple[float, float] | None:
    if not path:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None

    frame = read_dataset_frame(csv_path)
    has_minmax_columns = any("min" in str(col).lower() for col in frame.columns)
    if frame.shape[1] <= 2 or not has_minmax_columns:
        # Some GOST files include descriptive rows before the tabular header.
        for skip_rows in (1, 2, 3, 4):
            try:
                candidate = pd.read_csv(
                    csv_path,
                    sep=";",
                    engine="python",
                    decimal=",",
                    skipinitialspace=True,
                    skiprows=skip_rows,
                )
            except Exception:
                continue
            if candidate.shape[1] > 4 and any("min" in str(col).lower() for col in candidate.columns):
                frame = candidate
                break

    if frame.empty:
        return None

    numeric = pd.DataFrame({col: _as_numeric(frame[col]) for col in frame.columns})

    lower_col = None
    upper_col = None
    for col in frame.columns:
        col_l = str(col).lower()
        if lower_col is None and ("min" in col_l and ("r" in col_l or "сж" in col_l)):
            lower_col = col
        if upper_col is None and ("max" in col_l and ("r" in col_l or "сж" in col_l)):
            upper_col = col
    if lower_col is None:
        min_candidates = [col for col in frame.columns if "min" in str(col).lower()]
        if min_candidates:
            lower_col = min_candidates[0]
    if upper_col is None:
        max_candidates = [col for col in frame.columns if "max" in str(col).lower()]
        if max_candidates:
            upper_col = max_candidates[0]

    if lower_col is None or upper_col is None:
        return None

    lower_values = numeric[lower_col].dropna().to_numpy(dtype=float)
    upper_values = numeric[upper_col].dropna().to_numpy(dtype=float)
    if lower_values.size == 0 or upper_values.size == 0:
        return None

    return float(np.nanmin(lower_values)), float(np.nanmax(upper_values))


def _gost_violations_at_28_days(
    predictions_28d: np.ndarray,
    gost_range: tuple[float, float] | None,
) -> dict[str, Any]:
    if gost_range is None:
        return {"enabled": False, "violations": 0, "samples_28d": 0}

    lower, upper = gost_range
    violations = int(np.sum((predictions_28d < lower) | (predictions_28d > upper)))
    return {
        "enabled": True,
        "range": [lower, upper],
        "violations": violations,
        "samples_28d": int(predictions_28d.shape[0]),
    }


def _save_generator_checkpoint(
    path: Path,
    generator: AutoregressiveGenerator,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    feature_names: list[str],
    period_days: list[float],
    config: GanStageConfig,
) -> None:
    payload = {
        "state_dict": generator.state_dict(),
        "feature_scaler": feature_scaler.to_dict(),
        "target_scaler": target_scaler.to_dict(),
        "feature_names": feature_names,
        "period_days": period_days,
        "n_periods": len(period_days),
        "generator_config": config.generator.to_dict(),
    }
    torch.save(payload, path)


def finetune_generator(
    *,
    pretrained_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    epochs: int = 60,
    learning_rate: float = 2e-4,
    batch_size: int = 16,
    freeze_encoder: bool = True,
    mc_samples: int = 64,
    seed: int = 42,
    validation_split: float = 0.20,
) -> dict[str, Any]:
    """Fine-tune a pre-trained generator on a narrow lab dataset.

    Encoder weights are frozen by default (transfer learning regime).
    Only base_head and delta_heads are updated — this prevents overfitting
    when the fine-tune set is very small (< 100 rows).

    Returns a summary dict with train/val metrics before and after fine-tuning.
    """
    import copy as _copy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pre-trained checkpoint ──────────────────────────────────────────
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    gen_cfg = GanGeneratorConfig.from_dict(ckpt["generator_config"])
    period_days: list[float] = ckpt["period_days"]
    n_periods = ckpt["n_periods"]

    # ── Load lab data ─────────────────────────────────────────────────────────
    # Re-use the same feature names as the pretrained model.
    pretrained_features: list[str] = ckpt["feature_names"]

    # Load original resolved config from the checkpoint directory to know
    # which components / strength columns to read.  Fall back to searching
    # alongside pretrained_path.
    pretrained_path = Path(pretrained_path)
    resolved_cfg_path = pretrained_path.parent / "gan_config.resolved.json"
    if not resolved_cfg_path.exists():
        raise FileNotFoundError(
            f"Cannot find gan_config.resolved.json next to {pretrained_path}. "
            "Pass the artifacts dir produced by train_gan."
        )
    orig_config = load_gan_config(resolved_cfg_path)
    orig_config.dataset.data_path = str(data_path)  # override with lab data
    tmp_physics = orig_config.physics
    tmp_physics.add_extended_features = len(pretrained_features) > 10
    tmp_physics.add_late_age_features = len(pretrained_features) > 14
    orig_config.physics = tmp_physics

    prepared = _prepare_strength_dataset(orig_config)
    n = prepared.features.shape[0]
    print(f"[finetune] Lab dataset: {n} rows, {prepared.features.shape[1]} features")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * validation_split))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    # Scale using pretrained scalers (domain adaptation — DO NOT refit)
    feature_scaler = StandardScaler.from_dict(ckpt["feature_scaler"])
    target_scaler  = StandardScaler.from_dict(ckpt["target_scaler"])

    x_all  = feature_scaler.transform(prepared.features.astype(float))
    y_all  = target_scaler.transform(
        prepared.strengths.reshape(-1, 1)
    ).reshape(prepared.strengths.shape)

    x_train = torch.tensor(x_all[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_idx], dtype=torch.float32)
    x_val   = torch.tensor(x_all[val_idx],   dtype=torch.float32)
    y_val_raw = prepared.strengths[val_idx]  # (n_val, P) in MPa

    # ── Rebuild model ─────────────────────────────────────────────────────────
    generator = AutoregressiveGenerator(
        input_dim=len(pretrained_features),
        hidden_layers=gen_cfg.hidden_layers,
        noise_dim=gen_cfg.noise_dim,
        dropout=gen_cfg.dropout,
        n_periods=n_periods,
        use_uncertainty_head=gen_cfg.use_uncertainty_head,
        use_day28_aux_head=gen_cfg.use_day28_aux_head,
    )
    generator.load_state_dict(ckpt["state_dict"])

    # ── Metrics before fine-tuning ────────────────────────────────────────────
    val_pred_pre, val_std_pre, _ = _mc_predict_generator(
        generator, x_val, target_scaler, mc_samples=mc_samples
    )
    metrics_before = _metrics(y_val_raw.ravel(), val_pred_pre.ravel())
    metrics_before_by_period = _metrics_by_period(y_val_raw, val_pred_pre, period_days)
    print(f"[finetune] BEFORE — val MAE={metrics_before['mae']:.3f}  R²={metrics_before['r2']:.3f}")

    # ── Fine-tune: optionally freeze encoder ──────────────────────────────────
    if freeze_encoder:
        for p in generator.encoder.parameters():
            p.requires_grad = False
        print("[finetune] Encoder frozen — updating only base_head + delta_heads")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.HuberLoss(delta=1.0)

    best_state = _copy.deepcopy(generator.state_dict())
    best_val_loss = float("inf")
    no_improve = 0
    history: list[dict] = []
    torch.manual_seed(seed)

    ds = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        generator.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            gen_out = generator(xb)
            pred = gen_out[0] if isinstance(gen_out, tuple) else gen_out  # (B, P)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimizer.step()

        generator.eval()
        with torch.no_grad():
            val_gen_out = generator(x_val)
            val_pred_sc = (val_gen_out[0] if isinstance(val_gen_out, tuple) else val_gen_out).numpy()
        val_pred = target_scaler.inverse_transform(val_pred_sc[:, :1]).ravel()  # rough
        val_loss = float(np.mean((y_val_raw[:, 0] - val_pred) ** 2))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = _copy.deepcopy(generator.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 15:
            print(f"[finetune] Early stop at epoch {epoch}")
            break

        history.append({"epoch": epoch, "val_mse": val_loss})

    generator.load_state_dict(best_state)

    # ── Metrics after fine-tuning ─────────────────────────────────────────────
    # Unfreeze for MC inference
    for p in generator.parameters():
        p.requires_grad_(True)
    val_pred_post, val_std_post, _ = _mc_predict_generator(
        generator, x_val, target_scaler, mc_samples=mc_samples
    )
    metrics_after = _metrics(y_val_raw.ravel(), val_pred_post.ravel())
    metrics_after_by_period = _metrics_by_period(y_val_raw, val_pred_post, period_days)
    print(f"[finetune] AFTER  — val MAE={metrics_after['mae']:.3f}  R²={metrics_after['r2']:.3f}")

    # ── Save fine-tuned checkpoint ────────────────────────────────────────────
    ft_ckpt = dict(ckpt)
    ft_ckpt["state_dict"] = best_state
    torch.save(ft_ckpt, output_dir / "generator_finetuned.pt")

    # Predictions CSV
    pd.DataFrame({
        **{f"strength_true_day{int(d)}_mpa": y_val_raw[:, i] for i, d in enumerate(period_days)},
        **{f"strength_pred_before_day{int(d)}_mpa": val_pred_pre[:, i] for i, d in enumerate(period_days)},
        **{f"strength_pred_after_day{int(d)}_mpa": val_pred_post[:, i] for i, d in enumerate(period_days)},
    }).to_csv(output_dir / "finetune_predictions.csv", index=False)

    summary = {
        "stage": "finetune_generator",
        "n_train": int(len(train_idx)),
        "n_val":   int(len(val_idx)),
        "feature_names": pretrained_features,
        "metrics_before_finetune": {
            "validation": metrics_before,
            "validation_by_period": metrics_before_by_period,
        },
        "metrics_after_finetune": {
            "validation": metrics_after,
            "validation_by_period": metrics_after_by_period,
        },
        "improvement": {
            "mae_delta": metrics_after["mae"] - metrics_before["mae"],
            "r2_delta":  metrics_after["r2"]  - metrics_before["r2"],
        },
        "history": history[-10:],
    }
    (output_dir / "finetune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def _conformal_calibrate(
    generator: AutoregressiveGenerator,
    x_val: torch.Tensor,
    y_val_raw: np.ndarray,
    target_scaler: "StandardScaler",
    mc_samples: int,
    target_coverage: float = 0.95,
) -> float:
    """Compute conformal calibration quantile q_hat for the given coverage target.

    Returns q_hat such that mean ± q_hat * std covers target_coverage of val samples.
    Normalized residuals = |y - mean| / (std + 1e-8) are used as non-conformity scores.
    """
    mean_raw, std_raw, _ = _mc_predict_generator(generator, x_val, target_scaler, mc_samples)
    n = mean_raw.size
    residuals = np.abs(y_val_raw - mean_raw) / (std_raw + 1e-8)   # (N, P) → flatten
    scores = residuals.ravel()
    level = min(float(np.ceil((n + 1) * target_coverage) / n), 1.0)
    q_hat = float(np.quantile(scores, level))
    return q_hat


def run_train_gan(
    *,
    config_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    gan_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train autoregressive conditional GAN for concrete strength curve prediction.

    Generator predicts the full strength curve [y_1, y_3, y_7, y_28] from mix composition.
    Monotonicity y_1 ≤ y_3 ≤ y_7 ≤ y_28 is guaranteed architecturally via softplus increments.
    Discriminator (NEAT+BNN) scores realism of (features, full strength curve).
    """

    config = load_gan_config(config_path)

    root_dir = Path(artifacts_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(gan_dir) if gan_dir is not None else root_dir / "train_gan"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_strength_dataset(config)

    train_idx, val_idx = _split_indices(
        n_samples=prepared.features.shape[0],
        validation_split=config.training.validation_split,
        seed=config.training.seed,
        strengths_28=prepared.strengths[:, -1],
        source_labels=prepared.source_labels,
    )

    x_train_raw = prepared.features[train_idx]
    y_train_raw = prepared.strengths[train_idx]   # (N_train, P)
    w_train_raw = prepared.target_weights[train_idx]  # (N_train, P)
    y_val_raw = prepared.strengths[val_idx]        # (N_val, P)

    feature_scaler = StandardScaler.fit(x_train_raw)
    # Pool all training strength values across periods for a single shared scaler
    target_scaler = StandardScaler.fit(y_train_raw.ravel()[:, None])

    x_scaled = feature_scaler.transform(prepared.features)
    y_scaled = target_scaler.transform(
        prepared.strengths.reshape(-1, 1)
    ).reshape(prepared.strengths.shape)            # (N, P)

    x_train = torch.as_tensor(x_scaled[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(y_scaled[train_idx], dtype=torch.float32)   # (N_train, P)
    w_train = torch.as_tensor(w_train_raw, dtype=torch.float32)
    x_val = torch.as_tensor(x_scaled[val_idx], dtype=torch.float32)

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    generator = AutoregressiveGenerator(
        input_dim=x_train.shape[1],
        hidden_layers=config.generator.hidden_layers,
        noise_dim=config.generator.noise_dim,
        dropout=config.generator.dropout,
        n_periods=len(prepared.period_days),
        use_uncertainty_head=config.generator.use_uncertainty_head,
        use_day28_aux_head=config.generator.use_day28_aux_head,
    )

    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.generator.learning_rate,
        weight_decay=config.generator.weight_decay,
    )

    feature_index = {name: idx for idx, name in enumerate(prepared.feature_names)}
    wc_feature_index = feature_index.get("water_cement_ratio")

    warmup_log = _train_generator_epochs(
        generator,
        generator_optimizer,
        x_train,
        y_train,
        w_train,
        x_val,
        y_val_raw,
        target_scaler,
        batch_size=config.generator.batch_size,
        epochs=config.generator.warmup_epochs,
        early_stopping_rounds=config.generator.early_stopping_rounds,
        mc_samples=config.training.generator_mc_samples,
        supervised_weight=config.generator.supervised_weight,
        adversarial_weight=0.0,
        mape_weight=config.generator.mape_weight,
        period_weights=config.generator.period_weights,
        abrams_weight=config.physics.abrams_weight,
        nll_weight=0.0,
        aux_day28_weight=0.0,
        wc_feature_index=wc_feature_index,
        discriminator=None,
        max_grad_norm=config.generator.max_grad_norm,
        seed=config.training.seed,
    )

    rng = np.random.default_rng(config.training.seed)
    strength_names = [f"strength_day{int(d)}_scaled" for d in prepared.period_days]
    reliable_disc_mask = (w_train_raw.mean(axis=1) >= 0.6)
    disc_train_idx = train_idx[reliable_disc_mask] if np.any(reliable_disc_mask) else train_idx
    initial_fake = _make_fake_strength(y_scaled[disc_train_idx], rng, config.training.fake_jitter_std)
    initial_disc_x, initial_disc_y = _make_discriminator_dataset(
        x_scaled[disc_train_idx], y_scaled[disc_train_idx], initial_fake, rng,
    )

    discriminator_regressor, neat_result, neat_metadata = _build_discriminator(
        feature_names=prepared.feature_names,
        strength_names=strength_names,
        x_values=initial_disc_x,
        y_values=initial_disc_y,
        config=config,
        output_dir=output_dir,
    )

    round_logs: list[dict[str, Any]] = []
    best_global_state = copy.deepcopy(generator.state_dict())
    best_global_rmse = float("inf")

    for round_index in range(config.training.rounds):
        pred_train_raw, _pred_train_std, pred_train_scaled = _mc_predict_generator(
            generator,
            x_train,
            target_scaler,
            mc_samples=max(8, config.training.generator_mc_samples // 2),
        )
        del pred_train_raw

        fake_strength = pred_train_scaled + rng.normal(
            0.0,
            config.training.fake_jitter_std,
            size=pred_train_scaled.shape,
        )

        pred_train_scaled_disc = pred_train_scaled[reliable_disc_mask] if np.any(reliable_disc_mask) else pred_train_scaled
        fake_strength_disc = fake_strength[reliable_disc_mask] if np.any(reliable_disc_mask) else fake_strength

        disc_x, disc_y = _make_discriminator_dataset(
            x_scaled[disc_train_idx],
            y_scaled[disc_train_idx],
            fake_strength_disc,
            rng,
        )

        disc_fit = _fit_discriminator_bnn(
            discriminator_regressor,
            disc_x,
            disc_y,
            config,
            seed=config.training.seed + 31 * round_index,
        )

        discriminator_torch = DeterministicBnnDiscriminator(discriminator_regressor)
        discriminator_torch.eval()

        gen_log = _train_generator_epochs(
            generator,
            generator_optimizer,
            x_train,
            y_train,
            w_train,
            x_val,
            y_val_raw,
            target_scaler,
            batch_size=config.generator.batch_size,
            epochs=config.generator.epochs_per_round,
            early_stopping_rounds=config.generator.early_stopping_rounds,
            mc_samples=config.training.generator_mc_samples,
            supervised_weight=config.generator.supervised_weight,
            adversarial_weight=config.generator.adversarial_weight,
            mape_weight=config.generator.mape_weight,
            period_weights=config.generator.period_weights,
            abrams_weight=config.physics.abrams_weight,
            nll_weight=config.generator.nll_weight,
            aux_day28_weight=config.generator.aux_day28_weight,
            wc_feature_index=wc_feature_index,
            discriminator=discriminator_torch,
            max_grad_norm=config.generator.max_grad_norm,
            seed=config.training.seed + 101 * (round_index + 1),
        )

        current_rmse = float(gen_log.get("best_metrics", {}).get("rmse", float("inf")))
        if current_rmse < best_global_rmse:
            best_global_rmse = current_rmse
            best_global_state = copy.deepcopy(generator.state_dict())

        round_logs.append(
            {
                "round": round_index + 1,
                "discriminator_fit": disc_fit,
                "generator_fit": gen_log,
            }
        )

    generator.load_state_dict(best_global_state)

    train_pred_raw, train_pred_std, train_pred_scaled = _mc_predict_generator(
        generator,
        x_train,
        target_scaler,
        mc_samples=config.training.generator_mc_samples,
    )
    val_pred_raw, val_pred_std, val_pred_scaled = _mc_predict_generator(
        generator,
        x_val,
        target_scaler,
        mc_samples=config.training.generator_mc_samples,
    )

    train_metrics = _metrics(y_train_raw.ravel(), train_pred_raw.ravel())
    val_metrics = _metrics(y_val_raw.ravel(), val_pred_raw.ravel())

    # Per-period breakdown: each curing age gets its own metrics dict
    train_metrics_by_period = _metrics_by_period(y_train_raw, train_pred_raw, prepared.period_days)
    val_metrics_by_period = _metrics_by_period(y_val_raw, val_pred_raw, prepared.period_days)
    train_weighted_metrics = _weighted_metrics(
        y_train_raw.reshape(-1),
        train_pred_raw.reshape(-1),
        prepared.target_weights[train_idx].reshape(-1),
    )
    val_weighted_metrics = _weighted_metrics(
        y_val_raw.reshape(-1),
        val_pred_raw.reshape(-1),
        prepared.target_weights[val_idx].reshape(-1),
    )
    train_weighted_metrics_by_period = _weighted_metrics_by_period(
        y_train_raw,
        train_pred_raw,
        prepared.target_weights[train_idx],
        prepared.period_days,
    )
    val_weighted_metrics_by_period = _weighted_metrics_by_period(
        y_val_raw,
        val_pred_raw,
        prepared.target_weights[val_idx],
        prepared.period_days,
    )

    val_picp95 = float(
        np.mean((y_val_raw >= (val_pred_raw - 1.96 * val_pred_std)) & (y_val_raw <= (val_pred_raw + 1.96 * val_pred_std)))
    )

    # Conformal calibration: compute q_hat from validation non-conformity scores
    conformal_q_hat = _conformal_calibrate(
        generator, x_val, y_val_raw, target_scaler,
        mc_samples=config.training.generator_mc_samples,
    )
    val_picp95_conformal = float(
        np.mean(
            (y_val_raw >= (val_pred_raw - conformal_q_hat * val_pred_std))
            & (y_val_raw <= (val_pred_raw + conformal_q_hat * val_pred_std))
        )
    )

    discriminator_path = output_dir / "discriminator_bnn.pt"
    discriminator_regressor.save(discriminator_path)

    generator_path = output_dir / "generator.pt"
    _save_generator_checkpoint(
        generator_path,
        generator,
        feature_scaler,
        target_scaler,
        prepared.feature_names,
        prepared.period_days,
        config,
    )

    val_disc_input = np.hstack([x_scaled[val_idx], val_pred_scaled])  # (N, F+P)
    disc_mean, disc_std = discriminator_regressor.predict_components(
        val_disc_input,
        mc_samples=config.discriminator.bnn_mc_samples,
    )

    predictions_frame = pd.DataFrame(
        prepared.components[val_idx],
        columns=prepared.component_names,
    )
    if prepared.source_labels is not None:
        predictions_frame["source"] = prepared.source_labels[val_idx]
    for i, day in enumerate(prepared.period_days):
        day_key = f"day{int(day)}"
        predictions_frame[f"strength_true_{day_key}_mpa"] = y_val_raw[:, i]
        predictions_frame[f"strength_pred_{day_key}_mpa"] = val_pred_raw[:, i]
        predictions_frame[f"strength_pred_{day_key}_std_mpa"] = val_pred_std[:, i]
        predictions_frame[f"strength_true_{day_key}_weight"] = prepared.target_weights[val_idx, i]
    predictions_frame["disc_realness_mean"] = disc_mean.ravel()
    predictions_frame["disc_realness_std"] = disc_std.ravel()

    predictions_path = output_dir / "validation_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8")

    gost_range = _load_gost_strength_range(config.gost_path)
    day28_idx = next(
        (i for i, d in enumerate(prepared.period_days) if abs(d - 28.0) < 0.5),
        len(prepared.period_days) - 1,
    )
    gost_report = _gost_violations_at_28_days(val_pred_raw[:, day28_idx], gost_range)

    config_out = write_json(output_dir / "gan_config.resolved.json", config.to_dict())
    neat_summary_path = write_json(output_dir / "discriminator_neat_summary.json", neat_result)
    history_path = write_json(
        output_dir / "gan_training_history.json",
        {
            "warmup": warmup_log,
            "rounds": round_logs,
        },
    )

    summary = {
        "stage": "train_gan",
        "artifacts_dir": str(output_dir),
        "generator_model": str(generator_path),
        "discriminator_model": str(discriminator_path),
        "config_file": str(config_out),
        "discriminator_neat_summary": str(neat_summary_path),
        "training_history": str(history_path),
        "validation_predictions": str(predictions_path),
        "dataset": {
            "samples_total": int(prepared.features.shape[0]),
            "samples_train": int(train_idx.shape[0]),
            "samples_val": int(val_idx.shape[0]),
            "feature_names": prepared.feature_names,
            "component_bounds": prepared.component_bounds,
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "train_by_period": train_metrics_by_period,
            "validation_by_period": val_metrics_by_period,
            "validation_picp95": val_picp95,
            "conformal_q_hat": conformal_q_hat,
            "validation_picp95_conformal": val_picp95_conformal,
            "train_weighted": train_weighted_metrics,
            "validation_weighted": val_weighted_metrics,
            "train_weighted_by_period": train_weighted_metrics_by_period,
            "validation_weighted_by_period": val_weighted_metrics_by_period,
            "validation_mean_uncertainty_mpa": float(np.mean(val_pred_std)),
        },
        "gost_28d": gost_report,
        "discriminator": {
            "neat_algorithm": neat_metadata.get(
                "algorithm",
                config.discriminator.optimizer.algorithm or "python-neat",
            ),
            "validation_realness_mean": float(np.mean(disc_mean)),
            "validation_realness_std_mean": float(np.mean(disc_std)),
        },
    }

    write_json(output_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "GanStageConfig",
    "AutoregressiveGenerator",
    "load_gan_config",
    "run_train_gan",
]
