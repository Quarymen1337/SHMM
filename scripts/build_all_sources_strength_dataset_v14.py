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


def fit_log_curve(points: dict[int, float]) -> tuple[float, float] | None:
    valid_points = sorted((day, value) for day, value in points.items() if day > 0 and np.isfinite(value) and value > 0)
    if len(valid_points) < 2:
        return None
    xs = np.log(np.asarray([day for day, _ in valid_points], dtype=float))
    ys = np.asarray([value for _, value in valid_points], dtype=float)
    design = np.vstack([xs, np.ones(len(xs))]).T
    coef, _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
    return float(coef[0]), float(coef[1])


def predict_from_curve(curve: tuple[float, float] | None, day: int, fallback_28: float, factor: float) -> float:
    if curve is not None:
        value = curve[0] * np.log(float(day)) + curve[1]
        if np.isfinite(value) and value > 0:
            return float(value)
    return float(fallback_28 * factor)


def read_boxcrete() -> pd.DataFrame:
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
        if not any(day in by_time for day in (3, 5, 7)):
            continue

        curve = fit_log_curve(by_time)
        s1 = by_time.get(1)
        s3 = by_time.get(3)
        s7 = by_time.get(7)
        if s7 is None and 5 in by_time and curve is not None:
            s7 = predict_from_curve(curve, 7, s28, 0.82)
        if s1 is None:
            s1 = predict_from_curve(curve, 1, s28, 0.38)
        if s3 is None:
            s3 = predict_from_curve(curve, 3, s28, 0.64)
        if s7 is None:
            s7 = predict_from_curve(curve, 7, s28, 0.82)

        weights = {
            "strength_1_weight": 1.0 if 1 in by_time else 0.05,
            "strength_3_weight": 1.0 if 3 in by_time else 0.35,
            "strength_7_weight": 1.0 if 7 in by_time else (0.6 if 5 in by_time else 0.2),
            "strength_28_weight": 1.0,
        }

        row = {
            "cement": float(group["cement"].median()),
            "sand": float(group["sand"].median()),
            "gravel": float(group["gravel"].median()),
            "water": float(group["water"].median()),
            "plasticizer_kg": float(group["plasticizer_kg"].median()),
            "fly_ash": float(group["fly_ash"].median()),
            "microsilica_kg": float(group["microsilica_kg"].median()),
            "strength_1": float(s1),
            "strength_3": float(s3),
            "strength_7": float(s7),
            "strength_28": float(s28),
            "imputed_early": int(not (1 in by_time and 3 in by_time and 7 in by_time)),
        }
        row.update(weights)
        rows.append(row)

    return with_source_flags(pd.DataFrame(rows), "boxcrete")


def read_normal_concrete() -> pd.DataFrame:
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
        if not (3 in by_age or 7 in by_age):
            continue

        curve_points = {age: strength for age, strength in by_age.items() if age in {3, 7, 28}}
        curve = fit_log_curve(curve_points)
        s1 = by_age.get(1)
        s3 = by_age.get(3)
        s7 = by_age.get(7)
        if s1 is None:
            s1 = predict_from_curve(curve, 1, s28, 0.34)
        if s3 is None:
            s3 = predict_from_curve(curve, 3, s28, 0.58)
        if s7 is None:
            s7 = predict_from_curve(curve, 7, s28, 0.79)

        row = {column: float(group[column].iloc[0]) for column in group_columns}
        row.update(
            {
                "strength_1": float(s1),
                "strength_3": float(s3),
                "strength_7": float(s7),
                "strength_28": float(s28),
                "strength_1_weight": 1.0 if 1 in by_age else 0.0,
                "strength_3_weight": 1.0 if 3 in by_age else 0.3,
                "strength_7_weight": 1.0 if 7 in by_age else 0.35,
                "strength_28_weight": 1.0,
                "imputed_early": int(not (1 in by_age and 3 in by_age and 7 in by_age)),
            }
        )
        rows.append(row)

    return with_source_flags(pd.DataFrame(rows), "normal_concrete")


def main() -> None:
    synthetic = read_synthetic()
    boxcrete = read_boxcrete()
    normal_concrete = read_normal_concrete()
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

    out_path = OUT_DIR / "all_sources_strength_v14.csv"
    full.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print("Shape:", full.shape)
    print(
        full.groupby("source").agg(
            rows=("source", "size"),
            early_weight=("strength_1_weight", "mean"),
            day3_weight=("strength_3_weight", "mean"),
            day7_weight=("strength_7_weight", "mean"),
            day28_weight=("strength_28_weight", "mean"),
            imputed_share=("imputed_early", "mean"),
        ).to_string()
    )


if __name__ == "__main__":
    main()
