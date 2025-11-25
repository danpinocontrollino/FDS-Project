"""Create burnout_score and burnout_level targets, then merge them into daily logs.

Usage:
    python scripts/create_burnout_labels.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SCORE_COLS = [
    "perceived_stress_scale",
    "anxiety_score",
    "depression_score",
    "sleep_debt_hours",
    "job_satisfaction",
]

def load_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = pd.read_csv(RAW_DIR / "weekly_summaries.csv", parse_dates=["week_start"])
    daily = pd.read_csv(RAW_DIR / "daily_logs.csv", parse_dates=["date"])
    return weekly, daily

def compute_burnout_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    weekly_z = scaler.fit_transform(weekly[SCORE_COLS])
    weekly_z = pd.DataFrame(weekly_z, columns=SCORE_COLS, index=weekly.index)
    weekly = weekly.copy()
    weekly["burnout_score"] = (
        weekly_z["perceived_stress_scale"]
        + weekly_z["anxiety_score"]
        + weekly_z["depression_score"]
        + weekly_z["sleep_debt_hours"]
        - weekly_z["job_satisfaction"]
    ) / len(SCORE_COLS)

    low_thr = weekly["burnout_score"].quantile(0.33)
    high_thr = weekly["burnout_score"].quantile(0.66)

    def classify(score: float) -> int:
        if score < low_thr:
            return 0
        if score < high_thr:
            return 1
        return 2

    weekly["burnout_level"] = weekly["burnout_score"].apply(classify)
    weekly["week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    return weekly

def merge_daily_with_weekly(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    merged = daily.merge(
        weekly[["user_id", "week", "burnout_score", "burnout_level"]],
        on=["user_id", "week"],
        how="left",
    )
    return merged

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weekly, daily = load_raw_frames()
    weekly_targets = compute_burnout_targets(weekly)
    merged_daily = merge_daily_with_weekly(daily, weekly_targets)

    weekly_targets.to_parquet(PROCESSED_DIR / "weekly_with_burnout.parquet", index=False)
    merged_daily.to_parquet(PROCESSED_DIR / "daily_with_burnout.parquet", index=False)

    print("Saved:")
    print("  -", PROCESSED_DIR / "weekly_with_burnout.parquet")
    print("  -", PROCESSED_DIR / "daily_with_burnout.parquet")

if __name__ == "__main__":
    main()
