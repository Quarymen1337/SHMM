from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQ_COLS = [
    "cement",
    "sand",
    "gravel",
    "water",
    "plasticizer_kg",
    "fly_ash",
    "microsilica_kg",
    "strength_1",
    "strength_3",
    "strength_7",
    "strength_28",
]


def read_flexible(path: Path) -> pd.DataFrame:
    for sep, dec in ((";", ","), (",", "."), (";", ".")):
        try:
            df = pd.read_csv(path, sep=sep, decimal=dec, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


def keep_and_clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    for col in REQ_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out = out[REQ_COLS]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=REQ_COLS)
    for t in ["strength_1", "strength_3", "strength_7", "strength_28"]:
        out = out[out[t] > 0]
    out = out.reset_index(drop=True)
    out.insert(0, "source_dataset", source)
    return out


def prepare_synthetic() -> pd.DataFrame:
    df = read_flexible(DATA_DIR / "synthetic_training_data.csv")
    return keep_and_clean(df, "synthetic")


def prepare_lab_narrow() -> pd.DataFrame:
    df = read_flexible(DATA_DIR / "lab_narrow_data.csv")
    return keep_and_clean(df, "lab_narrow")


def prepare_workability() -> pd.DataFrame:
    df = read_flexible(DATA_DIR / "workability_data.csv")

    # Map to v7 schema. Assumption: slag acts as additional SCM and is merged into fly_ash.
    if "slag_kg" in df.columns:
        df["fly_ash"] = pd.to_numeric(df.get("fly_ash", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(
            df["slag_kg"], errors="coerce"
        ).fillna(0.0)

    return keep_and_clean(df, "workability")


def main() -> None:
    outputs: dict[str, dict[str, int]] = {}

    synth = prepare_synthetic()
    synth_path = OUT_DIR / "strength_synthetic_v7schema.csv"
    synth.to_csv(synth_path, index=False)
    outputs[synth_path.name] = {"rows": int(len(synth)), "cols": int(synth.shape[1])}

    lab = prepare_lab_narrow()
    lab_path = OUT_DIR / "strength_lab_narrow_v7schema.csv"
    lab.to_csv(lab_path, index=False)
    outputs[lab_path.name] = {"rows": int(len(lab)), "cols": int(lab.shape[1])}

    work = prepare_workability()
    work_path = OUT_DIR / "strength_workability_v7schema.csv"
    work.to_csv(work_path, index=False)
    outputs[work_path.name] = {"rows": int(len(work)), "cols": int(work.shape[1])}

    merged = pd.concat([synth, lab, work], ignore_index=True)
    merged_path = OUT_DIR / "strength_merged_v7schema.csv"
    merged.to_csv(merged_path, index=False)
    outputs[merged_path.name] = {"rows": int(len(merged)), "cols": int(merged.shape[1])}

    info_path = OUT_DIR / "strength_compare_manifest.json"
    info_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    print("Prepared datasets:")
    for name, shape in outputs.items():
        print(f"- {name}: {shape['rows']} rows x {shape['cols']} cols")


if __name__ == "__main__":
    main()
