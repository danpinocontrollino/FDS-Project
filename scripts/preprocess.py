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
6. Label encode categorical variables for embeddings (job_type, chronotype, etc.)
7. Standardize numeric features (z-score normalization)
8. Save the final feature matrix for training

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
        data/processed/label_encoders.joblib  (NEW: for categorical embeddings)

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
LABEL_ENCODERS_PATH = PROCESSED_DIR / "label_encoders.joblib"  # NEW: for embeddings

# ============================================================================
# V2 GOOGLE FORM COLUMN MAPPING
# ============================================================================
# Maps verbose Google Form question headers to clean variable names

GOOGLE_FORM_COLUMN_MAPPING = {
    # Timestamp and identity
    "informazioni cronologiche": "_timestamp",
    "name / email": "_name",
    "your name": "_name",
    
    # === JOB & WORK CONTEXT ===
    "what best describes your job type?": "job_type",
    "what best describes your job": "job_type",
    "job type": "job_type",
    "primary work arrangement": "work_arrangement",
    "work arrangement": "work_arrangement",
    "natural sleep pattern (chronotype)": "chronotype",
    "natural sleep pattern": "chronotype",
    "chronotype": "chronotype",
    
    # === SOCIAL & ISOLATION ===
    "quality of social interactions today": "social_quality",
    "quality of social interaction": "social_quality",
    "how often do you feel lonely or isolated?": "loneliness_level",
    "lonely or isolated": "loneliness_level",
    "do you have someone to talk to about personal problems?": "social_support",
    "someone to talk to": "social_support",
    "number of meaningful social interactions today": "social_interactions",
    "meaningful social interaction": "social_interactions",
    
    # === WORK ENVIRONMENT ===
    "how distracting was your work environment?": "environment_distractions",
    "distracting": "environment_distractions",
    "work-life boundary clarity": "work_life_boundary",
    "boundary clarity": "work_life_boundary",
    "checking emails outside work hours?": "after_hours_checking",
    "checking emails outside": "after_hours_checking",
    
    # === RECOVERY & WELLBEING ===
    "recovery: how well do you recover after stress?": "recovery_ability",
    "how well do you recover": "recovery_ability",
    "recover after stress": "recovery_ability",
    
    # === CORE METRICS ===
    "stress level": "stress_level",
    "mood score": "mood_score",
    "energy level": "energy_level",
    "focus / productivity": "focus_score",
    "focus": "focus_score",
    "sleep quality": "sleep_quality",
    "diet quality": "diet_quality",
    "job satisfaction (weekly)": "job_satisfaction",
    "job satisfaction": "job_satisfaction",
    
    # === QUANTITATIVE METRICS ===
    "sleep hours (last night)": "sleep_hours",
    "sleep hours": "sleep_hours",
    "work hours (today)": "work_hours",
    "work hours": "work_hours",
    "commute time (minutes)": "commute_minutes",
    "commute time": "commute_minutes",
    "total screen time (hours)": "screen_time_hours",
    "screen time": "screen_time_hours",
    "does your job require extended screen time?": "job_requires_screen",
    "job require extended screen": "job_requires_screen",
    "exercise / workout (minutes)": "exercise_minutes",
    "exercise": "exercise_minutes",
    "workout": "exercise_minutes",
    "time spent outdoors today": "outdoor_time_minutes",
    "time outdoors": "outdoor_time_minutes",
    "outdoor": "outdoor_time_minutes",
    "caffeine intake (mg)": "caffeine_mg",
    "caffeine": "caffeine_mg",
    "alcohol units": "alcohol_units",
    "alcohol": "alcohol_units",
    "steps count": "steps_count",
    "steps": "steps_count",
    "number of meetings": "meetings_count",
    "meetings": "meetings_count",
    "emails received (approx)": "emails_received",
    "emails received": "emails_received",
    "perceived work pressure": "work_pressure",
    "work pressure": "work_pressure",
}

# ============================================================================
# CATEGORICAL VALUE MAPPINGS
# ============================================================================
# Maps text responses to standardized categories for label encoding

JOB_TYPE_MAPPING = {
    # Knowledge/Office work
    "knowledge work (software, research)": "knowledge_work",
    "knowledge work": "knowledge_work",
    "software": "knowledge_work",
    "research": "knowledge_work",
    "it": "knowledge_work",
    "engineer": "knowledge_work",
    "developer": "knowledge_work",
    "analyst": "knowledge_work",
    
    # Creative work
    "creative work (design, writing, art)": "creative_work",
    "creative work": "creative_work",
    "design": "creative_work",
    "writing": "creative_work",
    "marketing": "creative_work",
    "content": "creative_work",
    
    # Healthcare
    "healthcare / medical": "healthcare",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "nurse": "healthcare",
    "doctor": "healthcare",
    
    # Education
    "education / teaching": "education",
    "education": "education",
    "teaching": "education",
    "teacher": "education",
    "professor": "education",
    
    # Service industry
    "customer service / retail": "service",
    "customer service": "service",
    "retail": "service",
    "hospitality": "service",
    "sales": "service",
    
    # Manual/Physical
    "manual / physical labor": "manual_labor",
    "manual labor": "manual_labor",
    "physical labor": "manual_labor",
    "construction": "manual_labor",
    "manufacturing": "manual_labor",
    
    # Management
    "management / executive": "management",
    "management": "management",
    "executive": "management",
    "manager": "management",
    
    # Finance
    "finance / accounting": "finance",
    "finance": "finance",
    "accounting": "finance",
    
    # Other
    "other": "other",
    "student": "other",
    "unemployed": "other",
    "retired": "other",
}

WORK_ARRANGEMENT_MAPPING = {
    "fully remote": "remote",
    "remote": "remote",
    "work from home": "remote",
    "wfh": "remote",
    
    "hybrid": "hybrid",
    "hybrid (mix of remote and office)": "hybrid",
    "mix": "hybrid",
    
    "fully on-site / office": "onsite",
    "on-site": "onsite",
    "office": "onsite",
    "in-person": "onsite",
    
    "field work (travel, client sites)": "field",
    "field work": "field",
    "field": "field",
    "travel": "field",
}

CHRONOTYPE_MAPPING = {
    "strong morning person (early bird)": "morning",
    "morning person": "morning",
    "early bird": "morning",
    "morning": "morning",
    
    "intermediate (flexible)": "intermediate",
    "intermediate": "intermediate",
    "flexible": "intermediate",
    "neither": "intermediate",
    
    "evening person (night owl)": "evening",
    "night owl": "evening",
    "evening": "evening",
    "night": "evening",
}

LONELINESS_MAPPING = {
    "never": 0,
    "rarely": 1,
    "sometimes": 2,
    "often": 3,
    "always": 4,
}

WORK_LIFE_BOUNDARY_MAPPING = {
    "very clear separation": 4,
    "clear": 4,
    "mostly clear": 3,
    "somewhat blurred": 2,
    "blurred": 2,
    "very blurred": 1,
    "no separation": 0,
}

AFTER_HOURS_MAPPING = {
    "never": 0,
    "rarely": 1,
    "sometimes": 2,
    "often": 3,
    "always": 4,
    "constantly": 4,
}

WORK_PRESSURE_MAPPING = {
    "low": 0,
    "medium": 1,
    "moderate": 1,
    "high": 2,
    "very high": 2,
}

JOB_REQUIRES_SCREEN_MAPPING = {
    "no (mostly physical/hands-on work)": 0,
    "no": 0,
    "minimal, less than 2 hours": 1,
    "minimal": 1,
    "moderate, 2-4 hours": 2,
    "moderate": 2,
    "yes (6+ hours essential)": 3,
    "yes": 3,
    "essential": 3,
}

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
    # === V2 FEATURES ===
    "social_quality",        # Quality of social interactions (1-10)
    "environment_distractions",  # How distracting is work environment (1-10)
    "recovery_ability",      # How well you recover after stress (1-10)
    "job_satisfaction",      # Job satisfaction (1-10)
    "commute_minutes",       # Daily commute time
    "emails_received",       # Number of emails received
]

# V2 Categorical columns for LABEL ENCODING (used for embeddings in neural nets)
# These get converted to integer IDs and can be embedded
EMBEDDING_CAT_COLS = [
    "job_type",              # Knowledge work, creative, healthcare, etc.
    "work_arrangement",      # Remote, hybrid, onsite, field
    "chronotype",            # Morning, intermediate, evening
]

# V2 Ordinal columns (categorical but with natural ordering)
# These are encoded as integers directly
ORDINAL_COLS = [
    "loneliness_level",      # 0-4 scale
    "work_life_boundary",    # 0-4 scale  
    "after_hours_checking",  # 0-4 scale
    "work_pressure",         # 0-2 scale
    "job_requires_screen",   # 0-3 scale
]

# Original categorical columns (will be one-hot encoded for backward compatibility)
CAT_COLS = [
    "profession",            # Job type (engineer, teacher, etc.)
    "work_mode",             # Remote, hybrid, or on-site
    "sex",                   # Male/Female
    "mental_health_history", # Previous mental health issues
    "exercise_habit",        # Regular exerciser or not
]

# Combined V2 categorical columns for new models
V2_CAT_COLS = EMBEDDING_CAT_COLS + ORDINAL_COLS

# New: Job types that inherently require screen time
# Used to contextualize screen_time_hours
SCREEN_INTENSIVE_JOBS = {
    "knowledge_work",        # Software, research, writing
    "creative_work",         # Design, marketing, content
    "finance",               # Finance, accounting
    "service",               # Often computer-based customer service
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


# ============================================================================
# V2: GOOGLE FORM CSV PARSING
# ============================================================================

def parse_google_form_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse a Google Form CSV export and map columns to clean variable names.
    
    This function:
    1. Maps verbose column headers to standardized names
    2. Cleans numeric values (handles ranges like "6-10", units like "15-30 min")
    3. Maps categorical text to standardized categories
    4. Returns a cleaned DataFrame ready for processing
    
    Args:
        csv_path: Path to the Google Form CSV export
        
    Returns:
        DataFrame with cleaned column names and values
    """
    df = pd.read_csv(csv_path)
    
    # Map column names
    mapped_df = pd.DataFrame()
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Check each mapping pattern
        for pattern, target in GOOGLE_FORM_COLUMN_MAPPING.items():
            if pattern in col_lower:
                mapped_df[target] = df[col]
                break
    
    # Clean numeric columns
    def clean_numeric(value, default=5):
        """Extract numeric value from messy input."""
        if pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        
        import re
        s = str(value).strip().lower()
        
        # Remove units and extract first number
        s_cleaned = re.sub(r'[a-zA-Z%°]+', '', s).strip()
        numbers = re.findall(r'\d+', s_cleaned)
        if numbers:
            return float(numbers[0])
        return default
    
    # Apply cleaning to numeric columns
    numeric_cols = [
        "sleep_hours", "sleep_quality", "work_hours", "stress_level",
        "mood_score", "energy_level", "focus_score", "exercise_minutes",
        "caffeine_mg", "alcohol_units", "steps_count", "meetings_count",
        "emails_received", "commute_minutes", "screen_time_hours",
        "social_quality", "social_interactions", "outdoor_time_minutes",
        "diet_quality", "job_satisfaction", "recovery_ability",
        "environment_distractions"
    ]
    
    for col in numeric_cols:
        if col in mapped_df.columns:
            mapped_df[col] = mapped_df[col].apply(lambda x: clean_numeric(x, 5))
    
    # Map categorical columns
    if "job_type" in mapped_df.columns:
        mapped_df["job_type"] = mapped_df["job_type"].str.lower().map(
            lambda x: JOB_TYPE_MAPPING.get(x, "other") if pd.notna(x) else "other"
        )
    
    if "work_arrangement" in mapped_df.columns:
        mapped_df["work_arrangement"] = mapped_df["work_arrangement"].str.lower().map(
            lambda x: WORK_ARRANGEMENT_MAPPING.get(x, "hybrid") if pd.notna(x) else "hybrid"
        )
    
    if "chronotype" in mapped_df.columns:
        mapped_df["chronotype"] = mapped_df["chronotype"].str.lower().map(
            lambda x: CHRONOTYPE_MAPPING.get(x, "intermediate") if pd.notna(x) else "intermediate"
        )
    
    if "loneliness_level" in mapped_df.columns:
        mapped_df["loneliness_level"] = mapped_df["loneliness_level"].str.lower().map(
            lambda x: LONELINESS_MAPPING.get(x, 2) if pd.notna(x) else 2
        )
    
    if "work_life_boundary" in mapped_df.columns:
        mapped_df["work_life_boundary"] = mapped_df["work_life_boundary"].str.lower().map(
            lambda x: WORK_LIFE_BOUNDARY_MAPPING.get(x, 2) if pd.notna(x) else 2
        )
    
    if "after_hours_checking" in mapped_df.columns:
        mapped_df["after_hours_checking"] = mapped_df["after_hours_checking"].str.lower().map(
            lambda x: AFTER_HOURS_MAPPING.get(x, 2) if pd.notna(x) else 2
        )
    
    if "work_pressure" in mapped_df.columns:
        mapped_df["work_pressure"] = mapped_df["work_pressure"].str.lower().map(
            lambda x: WORK_PRESSURE_MAPPING.get(x, 1) if pd.notna(x) else 1
        )
    
    if "job_requires_screen" in mapped_df.columns:
        mapped_df["job_requires_screen"] = mapped_df["job_requires_screen"].str.lower().map(
            lambda x: JOB_REQUIRES_SCREEN_MAPPING.get(x, 2) if pd.notna(x) else 2
        )
    
    return mapped_df


def create_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """
    Create and fit LabelEncoders for categorical columns.
    
    This allows us to convert categorical text values to integers
    for use in neural network embeddings, and decode them back later.
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        Dictionary mapping column names to fitted LabelEncoder objects
    """
    encoders = {}
    
    for col in EMBEDDING_CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            # Get unique values, handling NaN
            unique_vals = df[col].dropna().unique().tolist()
            
            # Add 'unknown' category for unseen values during inference
            if "unknown" not in unique_vals:
                unique_vals.append("unknown")
            
            le.fit(unique_vals)
            encoders[col] = le
    
    return encoders


def apply_label_encoders(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Apply label encoders to convert categorical text to integers.
    
    Args:
        df: DataFrame with categorical columns
        encoders: Dictionary of fitted LabelEncoders
        
    Returns:
        DataFrame with encoded columns (suffixed with _encoded)
    """
    df = df.copy()
    
    for col, encoder in encoders.items():
        if col in df.columns:
            # Handle unseen values by mapping to 'unknown'
            def safe_transform(x):
                if pd.isna(x) or x not in encoder.classes_:
                    return encoder.transform(["unknown"])[0]
                return encoder.transform([x])[0]
            
            df[f"{col}_encoded"] = df[col].apply(safe_transform)
    
    return df

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
    # Only use columns that actually exist in the data
    available_cols = [col for col in NUMERIC_SIGNAL_COLS if col in daily.columns]
    missing_cols = [col for col in NUMERIC_SIGNAL_COLS if col not in daily.columns]
    if missing_cols:
        print(f"[INFO] Skipping missing numeric columns: {missing_cols}")
    
    # Define aggregation: mean and std for each numeric column
    agg_map = {col: ["mean", "std"] for col in available_cols}
    
    # Ensure numeric type (some columns might be strings)
    for col in available_cols:
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
    # Only use columns that exist in the data
    available_cat_cols = [col for col in CAT_COLS if col in daily_all.columns]
    missing_cat_cols = [col for col in CAT_COLS if col not in daily_all.columns]
    if missing_cat_cols:
        print(f"[INFO] Skipping missing categorical columns: {missing_cat_cols}")
    
    agg = {col: "first" for col in available_cat_cols}
    
    # For intervention flags, take max (1 if any day had intervention)
    available_flags = [flag for flag in ["has_intervention"] + INTERVENTION_FLAGS if flag in daily_all.columns]
    flag_agg = {flag: "max" for flag in available_flags}
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
    1. Create and save label encoders for embedding columns
    2. Apply label encoding for embedding columns (job_type, chronotype, etc.)
    3. One-hot encode remaining categorical columns (backward compatibility)
    4. Remove zero-variance columns (constant values don't help ML)
    5. Standardize numeric columns to z-scores (mean=0, std=1)
    6. Save scaler and encoders for later use in inference
    
    Args:
        df: Raw feature DataFrame
        
    Returns:
        Encoded and scaled DataFrame ready for ML training
    """
    # Target columns - burnout (weekly) and focus (daily)
    target_cols = ["burnout_level", "burnout_score", "focus_level"]
    
    # Remove date columns that shouldn't be features
    feature_df = df.drop(columns=["week_start", "week_end"], errors="ignore")
    
    # === Label encode categorical columns for embeddings ===
    # Create and fit label encoders
    label_encoders = create_label_encoders(feature_df)
    
    # Apply label encoding (creates new columns like job_type_encoded)
    feature_df = apply_label_encoders(feature_df, label_encoders)
    
    # Save label encoders for inference
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    
    # === One-hot encode remaining categorical columns ===
    # Only one-hot encode columns that aren't being embedded
    onehot_cols = [c for c in CAT_COLS if c in feature_df.columns and c not in EMBEDDING_CAT_COLS]
    
    # Also one-hot encode has_intervention if present
    if "has_intervention" in feature_df.columns:
        onehot_cols.append("has_intervention")
    
    dummies = pd.get_dummies(feature_df, columns=onehot_cols, drop_first=True)
    
    # Identify feature columns (everything except targets and original cat cols)
    exclude_cols = set(target_cols) | set(EMBEDDING_CAT_COLS)  # Keep encoded versions, drop originals
    feature_cols = [c for c in dummies.columns if c not in exclude_cols]

    # Find numeric columns for scaling
    numeric_cols = [
        col for col in feature_cols 
        if dummies[col].dtype.kind in {"f", "i"}  # float or int
        and not col.endswith("_encoded")  # Don't scale encoded columns (they're IDs)
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
    encoded_cols = [f"{c}_encoded" for c in EMBEDDING_CAT_COLS if f"{c}_encoded" in dummies.columns]
    FEATURE_META.write_text(json.dumps({
        "feature_cols": feature_cols, 
        "numeric_cols": numeric_cols,
        "embedding_cols": encoded_cols,
        "embedding_vocab_sizes": {
            col: len(label_encoders[col].classes_) 
            for col in EMBEDDING_CAT_COLS 
            if col in label_encoders
        },
    }, indent=2))

    # Reorder columns: features first, then targets
    # Drop original categorical columns (keep encoded versions)
    final_cols = [c for c in feature_cols if c not in EMBEDDING_CAT_COLS] + target_cols
    return dummies[[c for c in final_cols if c in dummies.columns]]


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
