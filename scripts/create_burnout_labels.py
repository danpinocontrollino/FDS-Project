"""
================================================================================
CREATE BURNOUT & FOCUS LABELS SCRIPT
================================================================================
This script creates the target variables for our machine learning models:

1. BURNOUT PREDICTION (from weekly mental health surveys):
   - burnout_score: Composite index from stress, anxiety, depression, sleep debt, job satisfaction
   - burnout_level: 3-class (Low/Medium/High) using 33rd/66th percentile thresholds

2. FOCUS/DEEP WORK PREDICTION (from daily logs):
   - focus_level: 3-class (Low/Medium/High) based on focus_score
   - Thresholds: Low (1-4), Medium (5-7), High (8-10)

Usage:
    python scripts/create_burnout_labels.py

Input:  data/raw/weekly_summaries.csv, data/raw/daily_logs.csv
Output: data/processed/weekly_with_burnout.parquet
        data/processed/daily_with_burnout.parquet (includes both burnout_level and focus_level)

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
    
    # Step 2: Compute composite burnout score using PCA-derived weights
    # These weights were computed via PCA on the mental health indicators,
    # capturing the natural variance structure of the data.
    # PCA analysis showed: PC1 explains 49.2% variance, better class separation (+11%)
    #
    # PCA-derived weights (normalized to sum=1):
    #   - perceived_stress_scale: 0.278 (highest - core burnout indicator)
    #   - anxiety_score:          0.278 (highly correlated with stress, r=0.91)
    #   - depression_score:       0.193 (moderate contributor)
    #   - sleep_debt_hours:       0.076 (lower than expected - less variance)
    #   - job_satisfaction:       0.174 (protective factor, subtracted)
    #
    # Positive contributors: stress, anxiety, depression, sleep debt
    # Negative contributor: job satisfaction (higher satisfaction = lower burnout)
    weekly = weekly.copy()
    weekly["burnout_score"] = (
        0.278 * weekly_z["perceived_stress_scale"]
        + 0.278 * weekly_z["anxiety_score"]
        + 0.193 * weekly_z["depression_score"]
        + 0.076 * weekly_z["sleep_debt_hours"]
        - 0.174 * weekly_z["job_satisfaction"]  # Note the MINUS sign!
    )  # Weights sum to ~1, no additional division needed

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
    
    # Step 4: Create binary burnout classification (Healthy vs At-Risk)
    # This merges Medium+High into "At Risk" for better prediction accuracy
    # Rationale: Medium class is hardest to predict (38% accuracy), 
    # and from a prevention standpoint, both Medium and High need intervention
    weekly["burnout_binary"] = (weekly["burnout_level"] >= 1).astype(int)
    # 0 = Healthy (was Low), 1 = At Risk (was Medium or High)
    
    # Extract ISO week number for merging with daily data
    weekly["week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    
    return weekly


# ============================================================================
# DATA MERGING
# ============================================================================

def compute_focus_labels(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute focus_level (categorical) from daily focus_score.
    
    Focus/Deep Work prediction is valuable because:
    - Strong correlates: meetings (-0.31), stress (-0.20), sleep (+0.16)
    - Daily granularity allows actionable predictions ("optimize tomorrow")
    - Complements burnout prediction (burnout = long-term, focus = short-term)
    
    Thresholds (based on 1-10 scale distribution analysis):
    - Level 0 (Low):    focus_score 1-4 (23% of data - struggling to concentrate)
    - Level 1 (Medium): focus_score 5-7 (64% of data - normal productivity)
    - Level 2 (High):   focus_score 8-10 (14% of data - deep work state)
    
    Args:
        daily: DataFrame with daily behavioral logs including focus_score
        
    Returns:
        DataFrame with added focus_level column
    """
    daily = daily.copy()
    
    # Create focus_level using fixed thresholds (not quantiles)
    # This makes the classes more interpretable and stable
    def classify_focus(score: float) -> int:
        """Map focus_score (1-10) to discrete class."""
        if score <= 4:
            return 0  # Low focus - difficulty concentrating
        if score <= 7:
            return 1  # Medium focus - normal productivity
        return 2      # High focus - deep work state
    
    daily["focus_level"] = daily["focus_score"].apply(classify_focus)
    
    return daily


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
        weekly[["user_id", "week", "burnout_score", "burnout_level", "burnout_binary"]],
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
    2. daily_with_burnout.parquet: Daily logs with burnout AND focus targets
    """
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    weekly, daily = load_raw_frames()
    
    # Compute burnout targets from weekly mental health data
    weekly_targets = compute_burnout_targets(weekly)
    
    # Compute focus targets from daily data (before merging)
    daily_with_focus = compute_focus_labels(daily)
    
    # Merge burnout targets into daily data (each day gets its week's burnout level)
    merged_daily = merge_daily_with_weekly(daily_with_focus, weekly_targets)

    # Save as Parquet (efficient columnar format for ML workflows)
    weekly_targets.to_parquet(PROCESSED_DIR / "weekly_with_burnout.parquet", index=False)
    merged_daily.to_parquet(PROCESSED_DIR / "daily_with_burnout.parquet", index=False)

    # Print summary statistics
    print("=" * 60)
    print("TARGET LABELS CREATED")
    print("=" * 60)
    print()
    print("📁 Saved files:")
    print(f"   - {PROCESSED_DIR / 'weekly_with_burnout.parquet'}")
    print(f"   - {PROCESSED_DIR / 'daily_with_burnout.parquet'}")
    print()
    print("📊 Burnout Level Distribution (weekly, 3-class):")
    burnout_dist = weekly_targets["burnout_level"].value_counts(normalize=True).sort_index()
    for level, pct in burnout_dist.items():
        label = ["Low", "Medium", "High"][level]
        print(f"   {label}: {pct:.1%}")
    print()
    print("🏥 Burnout Binary Distribution (weekly, 2-class):")
    binary_dist = weekly_targets["burnout_binary"].value_counts(normalize=True).sort_index()
    for level, pct in binary_dist.items():
        label = ["Healthy", "At Risk"][level]
        print(f"   {label}: {pct:.1%}")
    print()
    print("🎯 Focus Level Distribution (daily):")
    focus_dist = merged_daily["focus_level"].value_counts(normalize=True).sort_index()
    for level, pct in focus_dist.items():
        label = ["Low", "Medium", "High"][level]
        print(f"   {label}: {pct:.1%}")
    print()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
