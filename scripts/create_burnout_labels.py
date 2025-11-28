"""
================================================================================
CREATE BURNOUT LABELS SCRIPT
================================================================================
This script creates the target variables (burnout_score and burnout_level) that
we will use to train our machine learning models.

The burnout score is computed as a composite index from multiple mental health
indicators, following the approach of combining standardized (z-score) values:
    burnout_score = (stress + anxiety + depression + sleep_debt - job_satisfaction) / 5

We then discretize into 3 classes using quantile thresholds:
    - Level 0 (Low):    bottom 33% of burnout scores
    - Level 1 (Medium): middle 33% of burnout scores  
    - Level 2 (High):   top 33% of burnout scores

This creates a balanced 3-class classification problem.

Usage:
    python scripts/create_burnout_labels.py

Input:  data/raw/weekly_summaries.csv, data/raw/daily_logs.csv
Output: data/processed/weekly_with_burnout.parquet
        data/processed/daily_with_burnout.parquet

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Columns used to compute the burnout composite score
# These represent key mental health and wellbeing indicators from the weekly surveys
SCORE_COLS = [
    "perceived_stress_scale",  # Higher = more stressed (contributes positively to burnout)
    "anxiety_score",           # Higher = more anxious (contributes positively to burnout)
    "depression_score",        # Higher = more depressed (contributes positively to burnout)
    "sleep_debt_hours",        # Higher = more sleep deficit (contributes positively to burnout)
    "job_satisfaction",        # Higher = more satisfied (contributes NEGATIVELY to burnout)
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV files from Kaggle dataset.
    
    Returns:
        weekly: DataFrame with weekly mental health summaries per user
        daily: DataFrame with daily behavioral logs per user
    """
    weekly = pd.read_csv(RAW_DIR / "weekly_summaries.csv", parse_dates=["week_start"])
    daily = pd.read_csv(RAW_DIR / "daily_logs.csv", parse_dates=["date"])
    return weekly, daily


# ============================================================================
# TARGET VARIABLE CREATION
# ============================================================================

def compute_burnout_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute burnout_score (continuous) and burnout_level (categorical) from weekly data.
    
    Methodology:
    1. Standardize all mental health indicators to z-scores (mean=0, std=1)
       This ensures each variable contributes equally regardless of its original scale.
    
    2. Combine into composite score:
       burnout = (stress_z + anxiety_z + depression_z + sleep_debt_z - satisfaction_z) / 5
       Note: job_satisfaction is SUBTRACTED because higher satisfaction = lower burnout
    
    3. Discretize into 3 classes using 33rd and 66th percentiles
       This creates balanced classes for classification.
    
    Args:
        weekly: DataFrame with weekly mental health scores
        
    Returns:
        DataFrame with added burnout_score, burnout_level, and week columns
    """
    # Step 1: Standardize the mental health indicators using z-scores
    # This is important because the raw scales may differ (e.g., 1-10 vs 0-100)
    scaler = StandardScaler()
    weekly_z = scaler.fit_transform(weekly[SCORE_COLS])
    weekly_z = pd.DataFrame(weekly_z, columns=SCORE_COLS, index=weekly.index)
    
    # Step 2: Compute composite burnout score
    # Positive contributors: stress, anxiety, depression, sleep debt
    # Negative contributor: job satisfaction (higher satisfaction = lower burnout)
    weekly = weekly.copy()
    weekly["burnout_score"] = (
        weekly_z["perceived_stress_scale"]
        + weekly_z["anxiety_score"]
        + weekly_z["depression_score"]
        + weekly_z["sleep_debt_hours"]
        - weekly_z["job_satisfaction"]  # Note the MINUS sign!
    ) / len(SCORE_COLS)

    # Step 3: Discretize into 3 classes using quantile thresholds
    # This creates approximately balanced classes (33% each)
    low_thr = weekly["burnout_score"].quantile(0.33)   # Bottom third threshold
    high_thr = weekly["burnout_score"].quantile(0.66)  # Top third threshold

    def classify(score: float) -> int:
        """Map continuous score to discrete class."""
        if score < low_thr:
            return 0  # Low burnout risk
        if score < high_thr:
            return 1  # Medium burnout risk
        return 2      # High burnout risk

    weekly["burnout_level"] = weekly["burnout_score"].apply(classify)
    
    # Extract ISO week number for merging with daily data
    weekly["week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    
    return weekly


# ============================================================================
# DATA MERGING
# ============================================================================

def merge_daily_with_weekly(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weekly burnout targets into daily logs.
    
    Each daily record gets the burnout_score and burnout_level from its corresponding
    week. This allows us to train sequence models on daily behavior to predict
    weekly burnout outcomes.
    
    Args:
        daily: DataFrame with daily behavioral logs
        weekly: DataFrame with weekly data including burnout targets
        
    Returns:
        Daily DataFrame with burnout_score and burnout_level columns added
    """
    daily = daily.copy()
    
    # Extract ISO week number from date for matching
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    
    # Merge on user_id and week to assign weekly targets to daily records
    merged = daily.merge(
        weekly[["user_id", "week", "burnout_score", "burnout_level"]],
        on=["user_id", "week"],
        how="left",  # Keep all daily records, even if no weekly match
    )
    
    return merged


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> None:
    """
    Main execution: load data, compute targets, merge, and save.
    
    This creates two output files:
    1. weekly_with_burnout.parquet: Weekly summaries with burnout targets
    2. daily_with_burnout.parquet: Daily logs with burnout targets (from their week)
    """
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    weekly, daily = load_raw_frames()
    
    # Compute burnout targets from weekly mental health data
    weekly_targets = compute_burnout_targets(weekly)
    
    # Merge targets into daily data (each day gets its week's burnout level)
    merged_daily = merge_daily_with_weekly(daily, weekly_targets)

    # Save as Parquet (efficient columnar format for ML workflows)
    weekly_targets.to_parquet(PROCESSED_DIR / "weekly_with_burnout.parquet", index=False)
    merged_daily.to_parquet(PROCESSED_DIR / "daily_with_burnout.parquet", index=False)

    print("Saved:")
    print("  -", PROCESSED_DIR / "weekly_with_burnout.parquet")
    print("  -", PROCESSED_DIR / "daily_with_burnout.parquet")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
