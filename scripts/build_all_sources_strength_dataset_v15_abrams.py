from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_COLUMNS = [
    "source_synthetic",
    "source_boxcrete",
    "source_normal_concrete",
]


def to_float_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def with_source_flags(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    out["source"] = source
    out["source_synthetic"] = 1.0 if source == "synthetic" else 0.0
    out["source_boxcrete"] = 1.0 if source == "boxcrete" else 0.0
    out["source_normal_concrete"] = 1.0 if source == "normal_concrete" else 0.0
    return out


def _clip_ratio(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def fit_abrams_ratio_models(reference: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Fit ratio(day)/strength_28 as linear function of log(c/w)."""
    cement = reference["cement"].to_numpy(dtype=float)
    water = reference["water"].to_numpy(dtype=float)
    cw = cement / np.clip(water, 1e-6, None)
    log_cw = np.log(np.clip(cw, 1e-6, None))
    s28 = reference["strength_28"].to_numpy(dtype=float)

    out: dict[int, tuple[float, float]] = {}
    bounds = {1: (0.18, 0.58), 3: (0.38, 0.82), 7: (0.55, 0.93)}

    for day in (1, 3, 7):
        sy = reference[f"strength_{day}"].to_numpy(dtype=float)
        ratio = sy / np.clip(s28, 1e-6, None)
        ratio = _clip_ratio(ratio, bounds[day][0], bounds[day][1])
        mask = np.isfinite(log_cw) & np.isfinite(ratio)
        if mask.sum() < 10:
            out[day] = (0.0, float(np.nanmedian(ratio)))
            continue
        x = log_cw[mask]
        y = ratio[mask]
        X = np.column_stack([x, np.ones_like(x)])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        out[day] = (float(coef[0]), float(coef[1]))
    return out


def predict_ratio(day: int, cement: float, water: float, models: dict[int, tuple[float, float]]) -> float:
    slope, intercept = models[day]
    log_cw = np.log(max(cement / max(water, 1e-6), 1e-6))
    value = slope * log_cw + intercept
    lo_hi = {1: (0.18, 0.58), 3: (0.38, 0.82), 7: (0.55, 0.93)}[day]
    return float(np.clip(value, lo_hi[0], lo_hi[1]))


def read_synthetic() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "synthetic_training_data.csv", sep=";", decimal=",", engine="python")
    out = pd.DataFrame(
        {
            "cement": to_float_series(df["cement"]),
            "sand": to_float_series(df["sand"]),
            "gravel": to_float_series(df["gravel"]),
            "water": to_float_series(df["water"]),
            "plasticizer_kg": to_float_series(df["plasticizer_kg"]),
            "fly_ash": to_float_series(df["fly_ash"]),
            "microsilica_kg": to_float_series(df["microsilica_kg"]),
            "strength_1": to_float_series(df["strength_1"]),
            "strength_3": to_float_series(df["strength_3"]),
            "strength_7": to_float_series(df["strength_7"]),
            "strength_28": to_float_series(df["strength_28"]),
        }
    )
    for column in ["strength_1", "strength_3", "strength_7", "strength_28"]:
        out[f"{column}_weight"] = 1.0
    out["imputed_early"] = 0
    return with_source_flags(out, "synthetic")


def read_boxcrete(models: dict[int, tuple[float, float]]) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "boxcrete_data.csv")
    numeric_columns = [
        "Cement (kg/m3)",
        "Fly Ash (kg/m3)",
        "Slag (kg/m3)",
        "Water (kg/m3)",
        "HRWR (kg/m3)",
        "Fine Aggregate (kg/m3)",
        "Coarse Aggregates (kg/m3)",
        "Time",
        "strength(mean) (MPa)",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    concrete_only = df[df["Mortar or Concrete"].astype(str).str.lower() == "concrete"].copy()
    frame = pd.DataFrame(
        {
            "mix_id": concrete_only["Mix Name"].astype(str),
            "time": concrete_only["Time"],
            "strength": concrete_only["strength(mean) (MPa)"],
            "cement": concrete_only["Cement (kg/m3)"],
            "sand": concrete_only["Fine Aggregate (kg/m3)"],
            "gravel": concrete_only["Coarse Aggregates (kg/m3)"].fillna(0.0),
            "water": concrete_only["Water (kg/m3)"],
            "plasticizer_kg": concrete_only["HRWR (kg/m3)"].fillna(0.0),
            "fly_ash": concrete_only["Fly Ash (kg/m3)"].fillna(0.0) + concrete_only["Slag (kg/m3)"].fillna(0.0),
            "microsilica_kg": 0.0,
        }
    ).dropna(subset=["time", "strength"])

    rows: list[dict[str, float | int | str]] = []
    for _, group in frame.groupby("mix_id"):
        group = group.sort_values("time")
        by_time = {
            int(round(time)): float(strength)
            for time, strength in zip(group["time"], group["strength"])
            if np.isfinite(time) and np.isfinite(strength) and strength > 0
        }
        s28 = by_time.get(28)
        if s28 is None or s28 <= 0:
            continue

        cement = float(group["cement"].median())
        water = float(group["water"].median())
        s1 = by_time.get(1)
        s3 = by_time.get(3)
        s7 = by_time.get(7)

        if s1 is None:
            s1 = s28 * predict_ratio(1, cement, water, models)
        if s3 is None:
            s3 = s28 * predict_ratio(3, cement, water, models)
        if s7 is None:
            s7 = s28 * predict_ratio(7, cement, water, models)

        row = {
            "cement": cement,
            "sand": float(group["sand"].median()),
            "gravel": float(group["gravel"].median()),
            "water": water,
            "plasticizer_kg": float(group["plasticizer_kg"].median()),
            "fly_ash": float(group["fly_ash"].median()),
            "microsilica_kg": float(group["microsilica_kg"].median()),
            "strength_1": float(s1),
            "strength_3": float(s3),
            "strength_7": float(s7),
            "strength_28": float(s28),
            "strength_1_weight": 1.0 if 1 in by_time else 0.06,
            "strength_3_weight": 1.0 if 3 in by_time else 0.22,
            "strength_7_weight": 1.0 if 7 in by_time else 0.28,
            "strength_28_weight": 1.0,
            "imputed_early": int(not (1 in by_time and 3 in by_time and 7 in by_time)),
        }
        rows.append(row)

    return with_source_flags(pd.DataFrame(rows), "boxcrete")


def read_normal_concrete(models: dict[int, tuple[float, float]]) -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "Normal_Concrete_DB.csv", sep=";", header=None, engine="python", dtype=str)
    header = raw.iloc[2].tolist()
    df = raw.iloc[3:].copy()
    df.columns = header
    df = df.rename(
        columns={
            "Цемент (кг/м³)": "cement",
            "Шлак BFS (кг/м³)": "slag",
            "Зола-унос (кг/м³)": "fly_ash",
            "Микрокремнезём (кг/м³)": "microsilica_kg",
            "Вода (кг/м³)": "water",
            "Суперпластификатор (кг/м³)": "plasticizer_kg",
            "ЩебеньКрупный заполнитель (кг/м³)": "gravel",
            "Песок Мелкий заполнитель (кг/м³)": "sand",
            "Возраст (дней)": "age",
            "Прочность CS_28d (МПа)  ← ТАРГЕТ 2": "strength",
        }
    )
    numeric_columns = [
        "cement",
        "slag",
        "fly_ash",
        "microsilica_kg",
        "water",
        "plasticizer_kg",
        "gravel",
        "sand",
        "age",
        "strength",
    ]
    for column in numeric_columns:
        df[column] = to_float_series(df[column])
    df["fly_ash"] = df["fly_ash"].fillna(0.0) + df["slag"].fillna(0.0)

    group_columns = ["cement", "sand", "gravel", "water", "plasticizer_kg", "fly_ash", "microsilica_kg"]
    rows: list[dict[str, float | int | str]] = []
    for _, group in df.groupby(group_columns, dropna=False):
        group = group.dropna(subset=["age", "strength"])
        group = group[group["strength"] > 0]
        if group.empty:
            continue
        by_age = group.groupby(group["age"].round().astype(int))["strength"].mean().to_dict()
        s28 = by_age.get(28)
        if s28 is None or s28 <= 0:
            continue

        cement = float(group["cement"].iloc[0])
        water = float(group["water"].iloc[0])

        s1 = by_age.get(1)
        s3 = by_age.get(3)
        s7 = by_age.get(7)
        if s1 is None:
            s1 = s28 * predict_ratio(1, cement, water, models)
        if s3 is None:
            s3 = s28 * predict_ratio(3, cement, water, models)
        if s7 is None:
            s7 = s28 * predict_ratio(7, cement, water, models)

        has_any_early = any(day in by_age for day in (1, 3, 7))
        row = {column: float(group[column].iloc[0]) for column in group_columns}
        row.update(
            {
                "strength_1": float(s1),
                "strength_3": float(s3),
                "strength_7": float(s7),
                "strength_28": float(s28),
                "strength_1_weight": 1.0 if 1 in by_age else (0.0 if has_any_early else 0.0),
                "strength_3_weight": 1.0 if 3 in by_age else (0.28 if has_any_early else 0.10),
                "strength_7_weight": 1.0 if 7 in by_age else (0.30 if has_any_early else 0.12),
                "strength_28_weight": 1.0,
                "imputed_early": int(not (1 in by_age and 3 in by_age and 7 in by_age)),
            }
        )
        rows.append(row)

    return with_source_flags(pd.DataFrame(rows), "normal_concrete")


def main() -> None:
    synthetic = read_synthetic()
    abrams_models = fit_abrams_ratio_models(synthetic)
    boxcrete = read_boxcrete(abrams_models)
    normal_concrete = read_normal_concrete(abrams_models)
    full = pd.concat([synthetic, boxcrete, normal_concrete], ignore_index=True)

    ordered_columns = [
        "cement",
        "sand",
        "gravel",
        "water",
        "plasticizer_kg",
        "fly_ash",
        "microsilica_kg",
        *SOURCE_COLUMNS,
        "strength_1",
        "strength_3",
        "strength_7",
        "strength_28",
        "strength_1_weight",
        "strength_3_weight",
        "strength_7_weight",
        "strength_28_weight",
        "source",
        "imputed_early",
    ]
    full = full[ordered_columns]
    numeric_columns = [column for column in ordered_columns if column != "source"]
    for column in numeric_columns:
        full[column] = pd.to_numeric(full[column], errors="coerce")

    full = full.dropna(
        subset=[
            "cement",
            "sand",
            "gravel",
            "water",
            "plasticizer_kg",
            "fly_ash",
            "microsilica_kg",
            *SOURCE_COLUMNS,
            "strength_1",
            "strength_3",
            "strength_7",
            "strength_28",
        ]
    )
    full = full[(full[["strength_1", "strength_3", "strength_7", "strength_28"]] > 0).all(axis=1)]
    full = full[(full[["strength_1_weight", "strength_3_weight", "strength_7_weight", "strength_28_weight"]].sum(axis=1) > 0)]

    out_path = OUT_DIR / "all_sources_strength_v15_abrams.csv"
    full.to_csv(out_path, index=False)

    print("Abrams ratio models (ratio = a*log(c/w)+b):", abrams_models)
    print("Saved:", out_path)
    print("Shape:", full.shape)
    print(
        full.groupby("source").agg(
            rows=("source", "size"),
            day1_weight=("strength_1_weight", "mean"),
            day3_weight=("strength_3_weight", "mean"),
            day7_weight=("strength_7_weight", "mean"),
            day28_weight=("strength_28_weight", "mean"),
            imputed_share=("imputed_early", "mean"),
            mean_day28=("strength_28", "mean"),
        ).to_string()
    )


if __name__ == "__main__":
    main()
