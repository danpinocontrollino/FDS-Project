"""
================================================================================
PREPROCESSING SCRIPT
================================================================================
This script transforms raw daily behavioral data into ML-ready tabular features.

The preprocessing pipeline:
1. Load daily and weekly data with burnout targets (from create_burnout_labels.py)
2. Aggregate daily signals to weekly statistics (mean, std)
3. Merge with categorical user attributes (profession, work_mode, etc.)
4. Attach intervention flags (therapy, vacation, etc.)
5. One-hot encode categorical variables
6. Standardize numeric features (z-score normalization)
7. Save the final feature matrix for training

This produces a single tabular dataset where each row is one user-week,
suitable for training traditional ML models (Random Forest, XGBoost, MLP).

Usage:
    python scripts/preprocess.py

Input:  data/processed/daily_with_burnout.parquet (from create_burnout_labels.py)
        data/processed/weekly_with_burnout.parquet
        data/raw/daily_all.csv
        data/raw/interventions.csv
        
Output: data/processed/tabular_ml_ready.parquet
        data/processed/feature_scaler.joblib
        data/processed/feature_columns.json

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Input files (created by create_burnout_labels.py)
DAILY_WITH_TARGET = PROCESSED_DIR / "daily_with_burnout.parquet"
WEEKLY_WITH_TARGET = PROCESSED_DIR / "weekly_with_burnout.parquet"

# Output files
OUTPUT_TABULAR = PROCESSED_DIR / "tabular_ml_ready.parquet"
SCALER_PATH = PROCESSED_DIR / "feature_scaler.joblib"
FEATURE_META = PROCESSED_DIR / "feature_columns.json"

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Numeric behavioral signals from daily logs
# These will be aggregated to weekly mean and std
NUMERIC_SIGNAL_COLS = [
    "sleep_hours",           # Hours of sleep per night
    "sleep_quality",         # Self-reported sleep quality (1-10)
    "work_hours",            # Hours worked per day
    "meetings_count",        # Number of meetings attended
    "tasks_completed",       # Number of tasks completed
    "exercise_minutes",      # Minutes of exercise
    "steps_count",           # Daily step count
    "caffeine_mg",           # Caffeine intake in mg
    "alcohol_units",         # Alcohol units consumed
    "screen_time_hours",     # Hours of screen time
    "social_interactions",   # Number of social interactions
    "outdoor_time_minutes",  # Time spent outdoors
    "diet_quality",          # Self-reported diet quality (1-10)
    "calories_intake",       # Daily calorie intake
    "stress_level",          # Self-reported stress (1-10)
    "mood_score",            # Self-reported mood (1-10)
    "energy_level",          # Self-reported energy (1-10)
    "focus_score",           # Self-reported focus (1-10)
    # === NEW V2 FEATURES ===
    "social_quality",        # Quality of social interactions (1-10)
    "environment_distractions",  # How distracting is work environment (1-10)
    "recovery_ability",      # How well you recover after stress (1-10)
]

# Categorical user attributes (will be one-hot encoded)
CAT_COLS = [
    "profession",            # Job type (engineer, teacher, etc.)
    "work_mode",             # Remote, hybrid, or on-site
    "chronotype",            # Morning person vs night owl
    "sex",                   # Male/Female
    "mental_health_history", # Previous mental health issues
    "exercise_habit",        # Regular exerciser or not
    "work_pressure",         # Low/medium/high work pressure
    # === NEW V2 FEATURES ===
    "work_arrangement",      # Remote, hybrid, office, field
    "workspace_type",        # Private office, open plan, home office, etc.
    "job_type",              # Knowledge work, creative, healthcare, etc.
    "loneliness_level",      # Never, rarely, sometimes, often, always
    "break_flexibility",     # How flexible are work breaks
    "work_life_boundary",    # Clear separation or blurred
    "after_hours_checking",  # Frequency of checking work outside hours
]

# New: Job types that inherently require screen time
# Used to contextualize screen_time_hours
SCREEN_INTENSIVE_JOBS = {
    "knowledge_work",        # Software, research, writing
    "creative_work",         # Design, marketing, content
    "finance",               # Finance, accounting
    "customer_service",      # Often computer-based
}

# Binary intervention flags (was user receiving any intervention that week?)
INTERVENTION_FLAGS = [
    "intervention_diet_coaching",
    "intervention_exercise_plan",
    "intervention_meditation",
    "intervention_sick_leave",
    "intervention_therapy",
    "intervention_vacation",
    "intervention_workload_cap",
]

# === NEW V2: Feature for contextualizing screen time ===
def adjust_screen_time_for_job(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an adjusted screen time metric that accounts for job requirements.
    
    For jobs that inherently require screen time (developers, designers, etc.),
    we create a 'leisure_screen_time' estimate and weight it differently.
    
    Args:
        df: DataFrame with screen_time_hours and job_type columns
        
    Returns:
        DataFrame with additional adjusted screen time column
    """
    df = df.copy()
    
    if "job_type" not in df.columns or "screen_time_hours" not in df.columns:
        return df
    
    # Estimate work-required screen time based on job type
    job_screen_requirements = {
        "knowledge_work": 6,
        "creative_work": 5,
        "finance": 6,
        "customer_service": 5,
        "management": 4,
        "sales": 3,
        "healthcare": 2,
        "education": 3,
        "manual_labor": 1,
        "other": 3,
    }
    
    # Calculate excess screen time (leisure portion)
    df["job_screen_requirement"] = df["job_type"].map(job_screen_requirements).fillna(3)
    df["leisure_screen_hours"] = (df["screen_time_hours"] - df["job_screen_requirement"]).clip(lower=0)
    
    return df


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def ensure_inputs_exist() -> None:
    """
    Check that required input files exist before processing.
    Raises FileNotFoundError with helpful message if files are missing.
    """
    missing: List[Path] = [p for p in [DAILY_WITH_TARGET, WEEKLY_WITH_TARGET] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed parquet files. Run scripts/create_burnout_labels.py first: "
            + ", ".join(str(p) for p in missing)
        )


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def aggregate_daily_signals(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily behavioral data to weekly statistics.
    
    For each numeric signal, we compute:
    - mean: Average value over the week (central tendency)
    - std: Standard deviation (variability/consistency)
    
    This captures both the typical behavior and how consistent it was.
    For example, someone with high mean sleep but high std might have
    irregular sleep patterns despite good average hours.
    
    Args:
        daily: DataFrame with daily behavioral logs
        
    Returns:
        DataFrame with one row per (user_id, week), containing mean/std features
    """
    # Define aggregation: mean and std for each numeric column
    agg_map = {col: ["mean", "std"] for col in NUMERIC_SIGNAL_COLS}
    
    # Ensure numeric type (some columns might be strings)
    for col in NUMERIC_SIGNAL_COLS:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    
    # Group by user and week, then aggregate
    grouped = daily.groupby(["user_id", "week"], as_index=False).agg(agg_map)
    
    # Flatten multi-level column names: (sleep_hours, mean) -> sleep_hours_mean
    grouped.columns = [
        "_".join(filter(None, map(str, col))).rstrip("_") 
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(columns={"user_id_": "user_id", "week_": "week"})
    
    return grouped


def aggregate_daily_all(daily_all: pd.DataFrame) -> pd.DataFrame:
    """
    Extract categorical user attributes from the extended daily dataset.
    
    These are user-level attributes that don't change day-to-day,
    so we just take the first value for each user-week combination.
    
    Args:
        daily_all: Extended daily dataset with categorical columns
        
    Returns:
        DataFrame with categorical features per (user_id, week)
    """
    daily_all = daily_all.copy()
    
    # Extract week number for grouping
    daily_all["week"] = pd.to_datetime(daily_all["week_start"]).dt.isocalendar().week.astype(int)
    
    # For categorical columns, just take first value (they're constant per user)
    agg = {col: "first" for col in CAT_COLS}
    
    # For intervention flags, take max (1 if any day had intervention)
    flag_agg = {flag: "max" for flag in ["has_intervention"] + INTERVENTION_FLAGS}
    agg.update(flag_agg)
    
    weekly_cat = daily_all.groupby(["user_id", "week"], as_index=False).agg(agg)
    return weekly_cat


def attach_interventions(weekly: pd.DataFrame, interventions: pd.DataFrame) -> pd.DataFrame:
    """
    Add a flag indicating whether any intervention was active during each week.
    
    This checks if the intervention's date range overlaps with the week's range.
    Having an active intervention might affect burnout levels (hopefully reducing them).
    
    Args:
        weekly: DataFrame with weekly records
        interventions: DataFrame with intervention date ranges per user
        
    Returns:
        Weekly DataFrame with added intervention_active boolean column
    """
    interventions = interventions.copy()
    interventions["start_date"] = pd.to_datetime(interventions["start_date"])
    interventions["end_date"] = pd.to_datetime(interventions["end_date"])
    
    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["week_end"] = weekly["week_start"] + pd.to_timedelta(6, unit="D")

    def has_overlap(row):
        """Check if any intervention overlaps with this week."""
        mask = (
            (interventions["user_id"] == row["user_id"])
            & (interventions["start_date"] <= row["week_end"])
            & (interventions["end_date"] >= row["week_start"])
        )
        return bool(mask.any())

    weekly["intervention_active"] = weekly.apply(has_overlap, axis=1)
    return weekly


def build_feature_table() -> pd.DataFrame:
    """
    Build the complete feature table by merging all data sources.
    
    This combines:
    - Weekly burnout targets
    - Aggregated daily behavioral statistics (mean/std)
    - Categorical user attributes
    - Intervention flags
    
    Returns:
        Complete feature DataFrame ready for encoding and scaling
    """
    # Load preprocessed data
    daily = pd.read_parquet(DAILY_WITH_TARGET)
    weekly = pd.read_parquet(WEEKLY_WITH_TARGET)
    daily_all = pd.read_csv(RAW_DIR / "daily_all.csv", parse_dates=["date", "week_start"])
    interventions = pd.read_csv(RAW_DIR / "interventions.csv")

    # Attach intervention flags to weekly data
    weekly = attach_interventions(weekly, interventions)
    
    # Aggregate daily signals to weekly statistics
    daily_features = aggregate_daily_signals(daily)
    
    # Extract categorical features
    cat_features = aggregate_daily_all(daily_all)

    # Merge everything together on (user_id, week)
    merged = weekly.merge(daily_features, on=["user_id", "week"], how="left")
    merged = merged.merge(cat_features, on=["user_id", "week"], how="left")

    # Drop rows with missing targets (shouldn't happen, but safety check)
    merged = merged.dropna(subset=["burnout_score", "burnout_level"])
    
    # Fill any remaining NaN values (forward fill, then backward fill)
    merged = merged.ffill().bfill()
    
    return merged


# ============================================================================
# ENCODING AND SCALING
# ============================================================================

def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables and standardize numeric features.
    
    Steps:
    1. One-hot encode categorical columns (creates binary columns)
    2. Remove zero-variance columns (constant values don't help ML)
    3. Standardize numeric columns to z-scores (mean=0, std=1)
    4. Save scaler for later use in inference
    
    Args:
        df: Raw feature DataFrame
        
    Returns:
        Encoded and scaled DataFrame ready for ML training
    """
    target_cols = ["burnout_level", "burnout_score"]
    
    # Remove date columns that shouldn't be features
    feature_df = df.drop(columns=["week_start", "week_end"], errors="ignore")
    
    # One-hot encode categorical columns
    # drop_first=True avoids multicollinearity (dummy variable trap)
    dummies = pd.get_dummies(feature_df, columns=CAT_COLS + ["has_intervention"], drop_first=True)
    
    # Identify feature columns (everything except targets)
    feature_cols = [c for c in dummies.columns if c not in target_cols]

    # Find numeric columns for scaling
    numeric_cols = [
        col for col in feature_cols 
        if dummies[col].dtype.kind in {"f", "i"}  # float or int
    ]
    
    # Remove zero-variance columns (they provide no information)
    zero_var = [col for col in numeric_cols if dummies[col].nunique(dropna=True) <= 1]
    if zero_var:
        dummies = dummies.drop(columns=zero_var)
        feature_cols = [col for col in feature_cols if col not in zero_var]
        numeric_cols = [col for col in numeric_cols if col not in zero_var]
    
    # Standardize numeric features (important for neural networks and regularized models)
    scaler = StandardScaler()
    dummies[numeric_cols] = scaler.fit_transform(dummies[numeric_cols].values)

    # Save scaler for inference (we need to apply same transformation to new data)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save feature metadata for reference
    FEATURE_META.write_text(json.dumps({
        "feature_cols": feature_cols, 
        "numeric_cols": numeric_cols
    }, indent=2))

    # Reorder columns: features first, then targets
    ordered_cols = feature_cols + target_cols
    return dummies[ordered_cols]


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> None:
    """
    Main execution: run the complete preprocessing pipeline.
    """
    # Validate inputs exist
    ensure_inputs_exist()
    
    # Build and process feature table
    features = build_feature_table()
    processed = encode_and_scale(features)
    
    # Save final dataset
    processed.to_parquet(OUTPUT_TABULAR, index=False)
    print("Wrote", OUTPUT_TABULAR)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
