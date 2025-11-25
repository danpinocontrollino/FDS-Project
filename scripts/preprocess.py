"""Preprocess daily + weekly data into ML-ready tabular datasets.

Steps
-----
1. Expect `scripts/create_burnout_labels.py` to have produced daily/weekly parquet files.
2. Aggregate daily behavior to weekly statistics.
3. Merge categorical attributes + intervention flags.
4. One-hot encode categoricals and scale numeric columns.
5. Persist processed parquet + scaler for reuse.

Usage:
    python scripts/preprocess.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DAILY_WITH_TARGET = PROCESSED_DIR / "daily_with_burnout.parquet"
WEEKLY_WITH_TARGET = PROCESSED_DIR / "weekly_with_burnout.parquet"
OUTPUT_TABULAR = PROCESSED_DIR / "tabular_ml_ready.parquet"
SCALER_PATH = PROCESSED_DIR / "feature_scaler.joblib"
FEATURE_META = PROCESSED_DIR / "feature_columns.json"

NUMERIC_SIGNAL_COLS = [
    "sleep_hours",
    "sleep_quality",
    "work_hours",
    "meetings_count",
    "tasks_completed",
    "exercise_minutes",
    "steps_count",
    "caffeine_mg",
    "alcohol_units",
    "screen_time_hours",
    "social_interactions",
    "outdoor_time_minutes",
    "diet_quality",
    "calories_intake",
    "stress_level",
    "mood_score",
    "energy_level",
    "focus_score",
    "work_pressure",
]

CAT_COLS = [
    "profession",
    "work_mode",
    "chronotype",
    "sex",
    "mental_health_history",
    "exercise_habit",
]

INTERVENTION_FLAGS = [
    "intervention_diet_coaching",
    "intervention_exercise_plan",
    "intervention_meditation",
    "intervention_sick_leave",
    "intervention_therapy",
    "intervention_vacation",
    "intervention_workload_cap",
]


def ensure_inputs_exist() -> None:
    missing: List[Path] = [p for p in [DAILY_WITH_TARGET, WEEKLY_WITH_TARGET] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed parquet files. Run scripts/create_burnout_labels.py first: "
            + ", ".join(str(p) for p in missing)
        )


def aggregate_daily_signals(daily: pd.DataFrame) -> pd.DataFrame:
    agg_map = {col: ["mean", "std"] for col in NUMERIC_SIGNAL_COLS}
    for col in NUMERIC_SIGNAL_COLS:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    grouped = daily.groupby(["user_id", "week"], as_index=False).agg(agg_map)
    grouped.columns = [
        "_".join(filter(None, map(str, col))).rstrip("_") for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(columns={"user_id_": "user_id", "week_": "week"})
    return grouped


def aggregate_daily_all(daily_all: pd.DataFrame) -> pd.DataFrame:
    daily_all = daily_all.copy()
    daily_all["week"] = pd.to_datetime(daily_all["week_start"]).dt.isocalendar().week.astype(int)
    agg = {col: "first" for col in CAT_COLS}
    flag_agg = {flag: "max" for flag in ["has_intervention"] + INTERVENTION_FLAGS}
    agg.update(flag_agg)
    weekly_cat = daily_all.groupby(["user_id", "week"], as_index=False).agg(agg)
    return weekly_cat


def attach_interventions(weekly: pd.DataFrame, interventions: pd.DataFrame) -> pd.DataFrame:
    interventions = interventions.copy()
    interventions["start_date"] = pd.to_datetime(interventions["start_date"])
    interventions["end_date"] = pd.to_datetime(interventions["end_date"])
    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["week_end"] = weekly["week_start"] + pd.to_timedelta(6, unit="D")

    def has_overlap(row):
        mask = (
            (interventions["user_id"] == row["user_id"])
            & (interventions["start_date"] <= row["week_end"])
            & (interventions["end_date"] >= row["week_start"])
        )
        return bool(mask.any())

    weekly["intervention_active"] = weekly.apply(has_overlap, axis=1)
    return weekly


def build_feature_table() -> pd.DataFrame:
    daily = pd.read_parquet(DAILY_WITH_TARGET)
    weekly = pd.read_parquet(WEEKLY_WITH_TARGET)
    daily_all = pd.read_csv(RAW_DIR / "daily_all.csv", parse_dates=["date", "week_start"])
    interventions = pd.read_csv(RAW_DIR / "interventions.csv")

    weekly = attach_interventions(weekly, interventions)
    daily_features = aggregate_daily_signals(daily)
    cat_features = aggregate_daily_all(daily_all)

    merged = weekly.merge(daily_features, on=["user_id", "week"], how="left")
    merged = merged.merge(cat_features, on=["user_id", "week"], how="left")

    merged = merged.dropna(subset=["burnout_score", "burnout_level"])
    merged = merged.ffill().bfill()
    return merged


def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["burnout_level", "burnout_score"]
    feature_df = df.drop(columns=["week_start", "week_end"], errors="ignore")
    dummies = pd.get_dummies(feature_df, columns=CAT_COLS + ["has_intervention"], drop_first=True)
    feature_cols = [c for c in dummies.columns if c not in target_cols]

    numeric_cols = [
        col for col in feature_cols if dummies[col].dtype.kind in {"f", "i"}
    ]
    zero_var = [col for col in numeric_cols if dummies[col].nunique(dropna=True) <= 1]
    if zero_var:
        dummies = dummies.drop(columns=zero_var)
        feature_cols = [col for col in feature_cols if col not in zero_var]
        numeric_cols = [col for col in numeric_cols if col not in zero_var]
    scaler = StandardScaler()
    dummies[numeric_cols] = scaler.fit_transform(dummies[numeric_cols].values)

    joblib.dump(scaler, SCALER_PATH)
    FEATURE_META.write_text(json.dumps({"feature_cols": feature_cols, "numeric_cols": numeric_cols}, indent=2))

    ordered_cols = feature_cols + target_cols
    return dummies[ordered_cols]


def main() -> None:
    ensure_inputs_exist()
    features = build_feature_table()
    processed = encode_and_scale(features)
    processed.to_parquet(OUTPUT_TABULAR, index=False)
    print("Wrote", OUTPUT_TABULAR)


if __name__ == "__main__":
    main()
