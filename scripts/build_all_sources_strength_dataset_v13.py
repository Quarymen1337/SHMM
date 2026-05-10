from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_float_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def read_synthetic() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "synthetic_training_data.csv", sep=";", decimal=",", engine="python")
    out = pd.DataFrame({
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
    })
    for c in ["strength_1", "strength_3", "strength_7", "strength_28"]:
        out[f"{c}_weight"] = 1.0
    out["source"] = "synthetic"
    out["imputed_early"] = 0
    return out


def read_boxcrete() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "boxcrete_data.csv")
    for c in [
        "Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)", "Water (kg/m3)",
        "HRWR (kg/m3)", "Fine Aggregate (kg/m3)", "Coarse Aggregates (kg/m3)",
        "Time", "strength(mean) (MPa)",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df2 = pd.DataFrame({
        "mix_id": df["Mix Name"].astype(str),
        "kind": df["Mortar or Concrete"].astype(str),
        "time": df["Time"],
        "strength": df["strength(mean) (MPa)"],
        "cement": df["Cement (kg/m3)"],
        "sand": df["Fine Aggregate (kg/m3)"],
        "gravel": df["Coarse Aggregates (kg/m3)"].fillna(0.0),
        "water": df["Water (kg/m3)"],
        "plasticizer_kg": df["HRWR (kg/m3)"].fillna(0.0),
        "fly_ash": df["Fly Ash (kg/m3)"].fillna(0.0) + df["Slag (kg/m3)"].fillna(0.0),
        "microsilica_kg": 0.0,
    }).dropna(subset=["time", "strength"])

    rows = []
    for mix_id, g in df2.groupby("mix_id"):
        g = g.sort_values("time")
        kind = str(g["kind"].mode().iloc[0]) if not g["kind"].mode().empty else "Concrete"
        weight_multiplier = 0.65 if kind.lower().startswith("mortar") else 1.0
        rec = {
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
        by_t = {int(round(t)): float(s) for t, s in zip(g["time"], g["strength"]) if np.isfinite(t) and np.isfinite(s)}
        s1 = by_t.get(1)
        s3 = by_t.get(3)
        s5 = by_t.get(5)
        s7 = by_t.get(7)
        s28 = by_t.get(28)
        if s28 is None or s28 <= 0:
            continue

        w1 = 1.0 if s1 is not None else 0.15
        w3 = 1.0 if s3 is not None else 0.20
        w7 = 1.0 if s7 is not None else 0.0
        w28 = 1.0

        if s7 is None and s5 is not None:
            a = (s28 - s5) / (np.log(28.0) - np.log(5.0))
            b = s5 - a * np.log(5.0)
            s7 = float(a * np.log(7.0) + b)
            w7 = 0.65

        if s1 is None:
            s1 = 0.40 * s28
            rec["imputed_early"] = 1
        if s3 is None:
            s3 = 0.65 * s28
            rec["imputed_early"] = 1
        if s7 is None:
            s7 = 0.82 * s28
            w7 = 0.20
            rec["imputed_early"] = 1

        rec.update({
            "strength_1": float(s1),
            "strength_3": float(s3),
            "strength_7": float(s7),
            "strength_28": float(s28),
            "strength_1_weight": float(w1 * weight_multiplier),
            "strength_3_weight": float(w3 * weight_multiplier),
            "strength_7_weight": float(w7 * weight_multiplier),
            "strength_28_weight": float(w28 * weight_multiplier),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def read_normal_concrete() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "Normal_Concrete_DB.csv", sep=";", header=None, engine="python", dtype=str)
    header = raw.iloc[2].tolist()
    df = raw.iloc[3:].copy()
    df.columns = header
    df = df.rename(columns={
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
    })
    for c in ["cement", "slag", "fly_ash", "microsilica_kg", "water", "plasticizer_kg", "gravel", "sand", "age", "strength"]:
        df[c] = to_float_series(df[c])
    df["fly_ash"] = df["fly_ash"].fillna(0.0) + df["slag"].fillna(0.0)

    group_cols = ["cement", "sand", "gravel", "water", "plasticizer_kg", "fly_ash", "microsilica_kg"]
    rows = []
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.dropna(subset=["age", "strength"])
        g = g[g["strength"] > 0]
        if g.empty:
            continue
        rec = {c: float(g[c].iloc[0]) for c in group_cols}
        rec["source"] = "normal_concrete"
        rec["imputed_early"] = 0
        by_age = g.groupby(g["age"].round().astype(int))["strength"].mean().to_dict()
        s1 = by_age.get(1)
        s3 = by_age.get(3)
        s7 = by_age.get(7)
        s28 = by_age.get(28)
        exact28 = s28 is not None

        xs = [np.log(float(a)) for a, s in sorted(by_age.items()) if a > 0 and np.isfinite(s)]
        ys = [float(s) for a, s in sorted(by_age.items()) if a > 0 and np.isfinite(s)]
        coef = None
        if len(xs) >= 2:
            A = np.vstack([xs, np.ones(len(xs))]).T
            coef, _, _, _ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)

        def pred_day(day: int) -> float:
            if coef is not None:
                return float(coef[0] * np.log(float(day)) + coef[1])
            factors = {1: 0.35, 3: 0.58, 7: 0.78}
            return float(factors[day] * s28)

        if s28 is None and coef is not None:
            s28 = pred_day(28)
        if s28 is None or not np.isfinite(s28) or s28 <= 0:
            continue

        w1 = 1.0 if by_age.get(1) is not None else (0.20 if coef is not None else 0.02)
        w3 = 1.0 if by_age.get(3) is not None else (0.35 if coef is not None else 0.03)
        w7 = 1.0 if by_age.get(7) is not None else (0.55 if coef is not None else 0.05)
        w28 = 1.0 if exact28 else 0.70

        if s1 is None:
            s1 = pred_day(1)
            rec["imputed_early"] = 1
        if s3 is None:
            s3 = pred_day(3)
            rec["imputed_early"] = 1
        if s7 is None:
            s7 = pred_day(7)
            rec["imputed_early"] = 1

        rec.update({
            "strength_1": float(s1),
            "strength_3": float(s3),
            "strength_7": float(s7),
            "strength_28": float(s28),
            "strength_1_weight": float(w1),
            "strength_3_weight": float(w3),
            "strength_7_weight": float(w7),
            "strength_28_weight": float(w28),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    syn = read_synthetic()
    box = read_boxcrete()
    ncd = read_normal_concrete()
    all_df = pd.concat([syn, box, ncd], ignore_index=True)
    cols = [
        "cement", "sand", "gravel", "water", "plasticizer_kg", "fly_ash", "microsilica_kg",
        "strength_1", "strength_3", "strength_7", "strength_28",
        "strength_1_weight", "strength_3_weight", "strength_7_weight", "strength_28_weight",
        "source", "imputed_early"
    ]
    all_df = all_df[cols]
    numeric_cols = [c for c in cols if c not in {"source"}]
    for c in numeric_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    all_df = all_df.dropna(subset=[
        "cement", "sand", "gravel", "water", "plasticizer_kg", "fly_ash", "microsilica_kg",
        "strength_1", "strength_3", "strength_7", "strength_28"
    ])
    all_df = all_df[(all_df[["strength_1", "strength_3", "strength_7", "strength_28"]] > 0).all(axis=1)]
    all_df = all_df[(all_df[["strength_1_weight", "strength_3_weight", "strength_7_weight", "strength_28_weight"]].sum(axis=1) > 0)]
    out = OUT_DIR / "all_sources_strength_v13.csv"
    all_df.to_csv(out, index=False)
    print('Saved:', out)
    print('Shape:', all_df.shape)
    print(all_df.groupby('source').agg(rows=('source','size'), early_weight=('strength_1_weight','mean'), day28_weight=('strength_28_weight','mean'), imputed_share=('imputed_early','mean')).to_string())


if __name__ == '__main__':
    main()
