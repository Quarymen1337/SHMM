from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DAYS = [1, 3, 7, 28]


def _to_float_series(s: pd.Series) -> pd.Series:
    # Convert values like '1 013,00' or '1\xa0013,00' safely to float
    return (
        s.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def _read_synthetic() -> pd.DataFrame:
    p = DATA_DIR / "synthetic_training_data.csv"
    df = pd.read_csv(p, sep=";", decimal=",", engine="python")
    out = pd.DataFrame({
        "cement": _to_float_series(df["cement"]),
        "sand": _to_float_series(df["sand"]),
        "gravel": _to_float_series(df["gravel"]),
        "water": _to_float_series(df["water"]),
        "plasticizer_kg": _to_float_series(df["plasticizer_kg"]),
        "fly_ash": _to_float_series(df["fly_ash"]),
        "microsilica_kg": _to_float_series(df["microsilica_kg"]),
        "strength_1": _to_float_series(df["strength_1"]),
        "strength_3": _to_float_series(df["strength_3"]),
        "strength_7": _to_float_series(df["strength_7"]),
        "strength_28": _to_float_series(df["strength_28"]),
    })
    out["source"] = "synthetic"
    out["imputed_early"] = 0
    return out


def _read_boxcrete() -> pd.DataFrame:
    p = DATA_DIR / "boxcrete_data.csv"
    df = pd.read_csv(p)

    # Numeric conversion
    num_cols = [
        "Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)", "Water (kg/m3)",
        "HRWR (kg/m3)", "Fine Aggregate (kg/m3)", "Coarse Aggregates (kg/m3)",
        "Time", "strength(mean) (MPa)",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build canonical features
    df2 = pd.DataFrame({
        "mix_id": df["Mix Name"].astype(str),
        "time": df["Time"],
        "strength": df["strength(mean) (MPa)"],
        "cement": df["Cement (kg/m3)"],
        "sand": df["Fine Aggregate (kg/m3)"],
        "gravel": df["Coarse Aggregates (kg/m3)"].fillna(0.0),
        "water": df["Water (kg/m3)"],
        "plasticizer_kg": df["HRWR (kg/m3)"].fillna(0.0),
        "fly_ash": df["Fly Ash (kg/m3)"].fillna(0.0) + df["Slag (kg/m3)"].fillna(0.0),
        "microsilica_kg": 0.0,
    })
    df2 = df2.dropna(subset=["time", "strength"])

    rows: list[dict[str, float | int | str]] = []
    for mix_id, g in df2.groupby("mix_id"):
        g = g.sort_values("time")
        rec: dict[str, float | int | str] = {
            "cement": float(g["cement"].median()),
            "sand": float(g["sand"].median()),
            "gravel": float(g["gravel"].median()),
            "water": float(g["water"].median()),
            "plasticizer_kg": float(g["plasticizer_kg"].median()),
            "fly_ash": float(g["fly_ash"].median()),
            "microsilica_kg": float(g["microsilica_kg"].median()),
            "source": "boxcrete",
            "imputed_early": 0,
        }

        # exact observed points
        by_t = {int(round(t)): float(s) for t, s in zip(g["time"], g["strength"]) if np.isfinite(t) and np.isfinite(s)}
        s1 = by_t.get(1)
        s3 = by_t.get(3)
        s5 = by_t.get(5)
        s7 = by_t.get(7)
        s28 = by_t.get(28)

        # If no day-28, skip (can't align with objective)
        if s28 is None or s28 <= 0:
            continue

        # estimate day 7 from day 5 + day 28 by log interpolation when possible
        if s7 is None and s5 is not None:
            t_a, y_a = 5.0, s5
            t_b, y_b = 28.0, s28
            a = (y_b - y_a) / (np.log(t_b) - np.log(t_a))
            b = y_a - a * np.log(t_a)
            s7 = float(a * np.log(7.0) + b)

        # fallback factors from day-28
        imputed = 0
        if s1 is None:
            s1 = 0.40 * s28
            imputed = 1
        if s3 is None:
            s3 = 0.65 * s28
            imputed = 1
        if s7 is None:
            s7 = 0.82 * s28
            imputed = 1

        rec.update({
            "strength_1": float(s1),
            "strength_3": float(s3),
            "strength_7": float(s7),
            "strength_28": float(s28),
            "imputed_early": imputed,
        })
        rows.append(rec)

    return pd.DataFrame(rows)


def _read_normal_concrete() -> pd.DataFrame:
    p = DATA_DIR / "Normal_Concrete_DB.csv"
    # first 2 rows are textual headers; row 3 contains actual column names
    raw = pd.read_csv(p, sep=";", header=None, engine="python", dtype=str)
    header = raw.iloc[2].tolist()
    df = raw.iloc[3:].copy()
    df.columns = header

    # rename Russian columns to canonical names
    colmap = {
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
        "Источник": "src_name",
    }
    df = df.rename(columns=colmap)

    needed = ["cement", "slag", "fly_ash", "microsilica_kg", "water", "plasticizer_kg", "gravel", "sand", "age", "strength"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = _to_float_series(df[c])

    # combine SCMs so final feature set matches model inputs
    df["fly_ash"] = df["fly_ash"].fillna(0.0) + df["slag"].fillna(0.0)

    # group by composition signature and map strength at ages to 1/3/7/28
    group_cols = ["cement", "sand", "gravel", "water", "plasticizer_kg", "fly_ash", "microsilica_kg"]
    rows: list[dict[str, float | int | str]] = []

    for _, g in df.groupby(group_cols, dropna=False):
        g = g.dropna(subset=["age", "strength"])
        if g.empty:
            continue
        g = g[g["strength"] > 0]
        if g.empty:
            continue

        rec: dict[str, float | int | str] = {c: float(g[c].iloc[0]) for c in group_cols}
        rec["source"] = "normal_concrete"
        rec["imputed_early"] = 0

        # aggregate by rounded age
        by_age = g.groupby(g["age"].round().astype(int))["strength"].mean().to_dict()
        s1 = by_age.get(1)
        s3 = by_age.get(3)
        s7 = by_age.get(7)
        s28 = by_age.get(28)

        # if 28 is missing but we have other ages, extrapolate/interpolate by log-linear fit
        if s28 is None:
            xs = []
            ys = []
            for a, s in sorted(by_age.items()):
                if a > 0 and np.isfinite(s):
                    xs.append(np.log(float(a)))
                    ys.append(float(s))
            if len(xs) >= 2:
                A = np.vstack([xs, np.ones(len(xs))]).T
                coef, _, _, _ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
                s28 = float(coef[0] * np.log(28.0) + coef[1])

        if s28 is None or not np.isfinite(s28) or s28 <= 0:
            continue

        # fill missing early ages using age-curve fit if available else fixed factors
        xs = []
        ys = []
        for a, s in sorted(by_age.items()):
            if a > 0 and np.isfinite(s):
                xs.append(np.log(float(a)))
                ys.append(float(s))

        def pred_day(day: int) -> float:
            if len(xs) >= 2:
                A = np.vstack([xs, np.ones(len(xs))]).T
                coef, _, _, _ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
                return float(coef[0] * np.log(float(day)) + coef[1])
            factors = {1: 0.40, 3: 0.65, 7: 0.82}
            return float(factors[day] * s28)

        imputed = 0
        if s1 is None:
            s1 = pred_day(1)
            imputed = 1
        if s3 is None:
            s3 = pred_day(3)
            imputed = 1
        if s7 is None:
            s7 = pred_day(7)
            imputed = 1

        rec.update({
            "strength_1": float(s1),
            "strength_3": float(s3),
            "strength_7": float(s7),
            "strength_28": float(s28),
            "imputed_early": imputed,
        })
        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    syn = _read_synthetic()
    box = _read_boxcrete()
    ncd = _read_normal_concrete()

    all_df = pd.concat([syn, box, ncd], ignore_index=True)

    cols = [
        "cement", "sand", "gravel", "water", "plasticizer_kg", "fly_ash", "microsilica_kg",
        "strength_1", "strength_3", "strength_7", "strength_28", "source", "imputed_early"
    ]
    all_df = all_df[cols]

    # keep only valid rows for training
    for c in ["cement", "sand", "gravel", "water", "plasticizer_kg", "fly_ash", "microsilica_kg",
              "strength_1", "strength_3", "strength_7", "strength_28"]:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    all_df = all_df.dropna()
    all_df = all_df[(all_df[["strength_1", "strength_3", "strength_7", "strength_28"]] > 0).all(axis=1)]
    all_df = all_df.reset_index(drop=True)

    out_csv = OUT_DIR / "all_sources_strength_v12.csv"
    all_df.to_csv(out_csv, index=False)

    summary = all_df.groupby("source").agg(
        rows=("source", "size"),
        mean_s28=("strength_28", "mean"),
        imputed_share=("imputed_early", "mean"),
    )

    print("Saved:", out_csv)
    print("Shape:", all_df.shape)
    print(summary)


if __name__ == "__main__":
    main()
