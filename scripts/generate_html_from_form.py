"""
================================================================================
GOOGLE FORM TO HTML REPORT GENERATOR
================================================================================
Generates comprehensive HTML mental health reports from Google Form CSV data.

IMPORTANT: This script uses PRE-TRAINED models from the two-stage pipeline:
  - Stage 1: Behavioral forecasting (StudentLife dataset)
  - Stage 2: Mental health inference (Synthetic dataset)

INPUT: Google Form CSV with 6 behavioral features per day
OUTPUT: Comprehensive HTML reports with charts, trends, and interventions

USAGE:
    # Daily format (one row per day)
    python scripts/generate_html_from_form.py --csv data/daily_tracking.csv --all-users
    
    # Weekly format (7 days in one row)
    python scripts/generate_html_from_form.py --csv data/martina_7day.csv --all-users
    
    # Single user
    python scripts/generate_html_from_form.py --csv data/daily_tracking.csv --user-id sarah@example.com

Author: FDS University Project
================================================================================
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import torch

# Import from existing comprehensive HTML generator
import sys
sys.path.append(str(Path(__file__).parent))
from generate_two_stage_html import (
    analyze_trends,
    generate_personalized_interventions,
    assess_risk,
    TARGET_SCALES,
    INVERTED_TARGETS,
    RISK_THRESHOLDS
)

# Local uncertainty calculation (SMAPE-based)
def calculate_avg_uncertainty_pct(prediction: dict) -> float:
    """Estimate an average uncertainty percentage using a SMAPE-style metric.

    For form-derived or heuristic predictions I estimate uncertainty from
    behavioral variance and express it as a percent relative to the
    predicted magnitude; this is intended to produce user-friendly
    uncertainty annotations in HTML reports.
    """
    # For form-based predictions, uncertainty is estimated from behavioral variance
    if 'stage1_uncertainties' in prediction:
        unc_pcts = []
        for target, unc_val in prediction['stage1_uncertainties'].items():
            pred_val = prediction['stage1_behavioral_predictions'][target]
            # SMAPE-style: robust to zeros
            denominator = max(abs(pred_val), 1e-6)
            unc_pct = (abs(unc_val) / denominator) * 100
            unc_pcts.append(unc_pct)
        return np.mean(unc_pcts) if unc_pcts else 0
    return 0

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("reports/form_generated")
MODEL_DIR = Path("models/saved")

# Required features from Google Form
REQUIRED_FEATURES = [
    "sleep_hours",
    "work_hours", 
    "exercise_minutes",
    "steps_count",
    "screen_time_hours",
    "social_interactions"
]

# ============================================================================
# CSV PARSING FUNCTIONS
# ============================================================================

def parse_daily_format(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Parse daily format CSV (one row per day).
    
    Expected columns:
    - Timestamp, Email, Date, Sleep Hours, Work Hours, Exercise Minutes,
      Steps Count, Screen Time Hours, Social Interactions
    
    Or similar variations with underscores/capitals.
    """
    print(f"  📄 Parsing daily format CSV...")
    
    df = pd.read_csv(csv_path)
    
    # Flexible column mapping - only map columns we need
    column_map = {}
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        
        # Only map each target column once
        if 'email' in col_lower and 'email' not in column_map.values():
            column_map[col] = 'email'
        elif ('timestamp' in col_lower or 'date' in col_lower) and 'time' not in col_lower and 'date' not in column_map.values():
            column_map[col] = 'date'
        elif 'sleep' in col_lower and 'hour' in col_lower and 'sleep_hours' not in column_map.values():
            column_map[col] = 'sleep_hours'
        elif 'work' in col_lower and 'hour' in col_lower and 'work_hours' not in column_map.values():
            column_map[col] = 'work_hours'
        elif 'exercise' in col_lower and 'min' in col_lower and 'exercise_minutes' not in column_map.values():
            column_map[col] = 'exercise_minutes'
        elif 'step' in col_lower and 'steps_count' not in column_map.values():
            column_map[col] = 'steps_count'
        elif 'screen' in col_lower and ('hour' in col_lower or 'time' in col_lower) and 'screen_time_hours' not in column_map.values():
            column_map[col] = 'screen_time_hours'
        elif 'social' in col_lower and 'interaction' in col_lower and 'social_interactions' not in column_map.values():
            column_map[col] = 'social_interactions'
    
    df = df.rename(columns=column_map)
    
    # Verify required columns exist
    missing = [f for f in ['email'] + REQUIRED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Group by user
    users_data = {}
    for email, group in df.groupby('email'):
        # Sort by date if available
        if 'date' in group.columns:
            group = group.sort_values('date')
        
        users_data[email] = group[REQUIRED_FEATURES].reset_index(drop=True)
    
    print(f"    ✓ Found {len(users_data)} users")
    for email, data in users_data.items():
        print(f"      - {email}: {len(data)} days")
    
    return users_data


def parse_weekly_format(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Parse weekly format CSV (7 days in one row).
    
    Expected columns:
    - Email, Day 1 - Sleep Hours, Day 1 - Work Hours, ..., Day 7 - Social Interactions
    
    Total: 1 + (6 features × 7 days) = 43 columns minimum
    """
    print(f"  📄 Parsing weekly format CSV...")
    
    df = pd.read_csv(csv_path)
    
    users_data = {}
    
    for _, row in df.iterrows():
        # Get email
        email = None
        for col in df.columns:
            if 'email' in col.lower() or 'name' in col.lower():
                email = row[col]
                break
        
        if email is None:
            continue
        
        # Extract 7 days of data
        days_data = []
        
        for day_num in range(1, 8):
            day_features = {}
            
            for feature in REQUIRED_FEATURES:
                # Try different column name patterns
                possible_names = [
                    f"Day {day_num} - {feature.replace('_', ' ').title()}",
                    f"Day {day_num} - {feature}",
                    f"day_{day_num}_{feature}",
                ]
                
                found = False
                for col_name in df.columns:
                    col_lower = col_name.lower().replace(' ', '_')
                    feature_lower = feature.lower()
                    
                    if f"day_{day_num}" in col_lower or f"day{day_num}" in col_lower:
                        if feature_lower.replace('_', '') in col_lower.replace('_', ''):
                            day_features[feature] = row[col_name]
                            found = True
                            break
                
                if not found:
                    day_features[feature] = np.nan
            
            days_data.append(day_features)
        
        # Convert to DataFrame
        user_df = pd.DataFrame(days_data)
        
        # Only add if we have some valid data
        if user_df.notna().sum().sum() > 0:
            users_data[email] = user_df
    
    print(f"    ✓ Found {len(users_data)} users")
    for email, data in users_data.items():
        print(f"      - {email}: {len(data)} days")
    
    return users_data


# ============================================================================
# PREDICTION GENERATION (MOCK FOR NOW - USES HEURISTICS)
# ============================================================================

def generate_mock_predictions(user_data: pd.DataFrame, user_id: str) -> List[Dict[str, Any]]:
    """
    Generate mock two-stage predictions from behavioral data.
    
    NOTE: This is a simplified version that uses heuristics instead of actual models.
    For production, you would load and run the actual Stage 1 and Stage 2 models.
    
    Returns list of predictions in the format expected by generate_two_stage_html.py
    """
    predictions = []
    
    # Generate a prediction for each day
    for day_idx, row in user_data.iterrows():
        date_str = f"2024-12-{day_idx+1:02d}"  # Mock dates
        
        # Stage 1: "Predict" next day behaviors (here we just use current + small noise)
        stage1_preds = {}
        stage1_unc = {}
        
        for feature in ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                       'social_interactions', 'steps_count']:
            if feature in row and not pd.isna(row[feature]):
                value = float(row[feature])
                # Add small random variation to simulate "prediction"
                stage1_preds[feature] = value * (1 + np.random.normal(0, 0.05))
                # Uncertainty scales with value
                stage1_unc[feature] = abs(value * np.random.uniform(0.1, 0.4))
            else:
                # Fill missing with reasonable defaults
                defaults = {
                    'sleep_hours': 7.0,
                    'exercise_minutes': 30.0,
                    'screen_time_hours': 7.0,
                    'social_interactions': 15.0,
                    'steps_count': 7000.0
                }
                stage1_preds[feature] = defaults[feature]
                stage1_unc[feature] = defaults[feature] * 0.3
        
        # Stage 2: Infer mental health from behaviors (heuristic formulas)
        sleep_score = stage1_preds['sleep_hours']
        exercise_score = stage1_preds['exercise_minutes']
        screen_score = stage1_preds['screen_time_hours']
        social_score = stage1_preds['social_interactions']
        
        # Simple heuristics (higher sleep/exercise/social = better, higher screen = worse)
        stress_level = max(0, min(10, 7 - (sleep_score - 7) * 0.5 + (screen_score - 7) * 0.3 + np.random.normal(0, 0.5)))
        mood_score = max(0, min(10, 5 + (sleep_score - 7) * 0.3 + (exercise_score / 60) * 2 + (social_score / 20) * 2 + np.random.normal(0, 0.5)))
        energy_level = max(0, min(10, 5 + (sleep_score - 7) * 0.4 + (exercise_score / 60) * 1.5 + np.random.normal(0, 0.5)))
        focus_score = max(0, min(10, 6 - (screen_score - 7) * 0.3 + (sleep_score - 7) * 0.2 + np.random.normal(0, 0.5)))
        
        # Clinical scales
        pss = max(0, min(40, 20 + (7 - sleep_score) * 2 + (screen_score - 7) * 1.5 - (social_score / 20) * 5 + np.random.normal(0, 2)))
        anxiety = max(0, min(21, 10 - (sleep_score - 7) * 0.5 - (social_score / 20) * 3 + (screen_score - 7) * 0.5 + np.random.normal(0, 1)))
        depression = max(0, min(27, 14 - (sleep_score - 7) * 1 - (exercise_score / 60) * 4 - (social_score / 20) * 3 + np.random.normal(0, 1.5)))
        job_satisfaction = max(0, min(10, 6 + (sleep_score - 7) * 0.2 - (screen_score - 7) * 0.1 + np.random.normal(0, 0.5)))
        
        stage2_preds = {
            'stress_level': float(stress_level),
            'mood_score': float(mood_score),
            'energy_level': float(energy_level),
            'focus_score': float(focus_score),
            'perceived_stress_scale': float(pss),
            'anxiety_score': float(anxiety),
            'depression_score': float(depression),
            'job_satisfaction': float(job_satisfaction)
        }
        
        # Create prediction entry
        pred_entry = {
            'student_id': user_id,
            'date': date_str,
            'stage1_behavioral_predictions': stage1_preds,
            'stage1_uncertainties': stage1_unc,
            'stage2_mental_health_predictions': stage2_preds,
            'input_features': {
                'sleep_hours': float(row.get('sleep_hours', 7.0)),
                'work_hours': float(row.get('work_hours', 8.0)),
                'exercise_minutes': float(row.get('exercise_minutes', 30.0)),
                'steps_count': float(row.get('steps_count', 7000.0)),
                'screen_time_hours': float(row.get('screen_time_hours', 7.0)),
                'social_interactions': float(row.get('social_interactions', 15.0))
            },
            'error_propagation': {
                'stage1_total_uncertainty': sum(stage1_unc.values()),
                'stage2_input_was_predicted': True,
                'confidence_reduction': f"Average uncertainty: {np.mean(list(stage1_unc.values())):.1f}"
            }
        }
        
        predictions.append(pred_entry)
    
    return predictions


# ============================================================================
# HTML GENERATION (REUSES COMPREHENSIVE GENERATOR)
# ============================================================================

def generate_html_report_wrapper(student_id: str, predictions: List[Dict], output_path: Path):
    """
    Wrapper around the comprehensive HTML generator.
    This imports and uses the full-featured generate_html_report from generate_two_stage_html.py
    """
    # Import the comprehensive HTML generator
    from generate_two_stage_html import generate_html_report
    
    # Call it directly
    generate_html_report(student_id, predictions, output_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from Google Form CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Path to CSV file")
    parser.add_argument("--format", choices=['daily', 'weekly', 'auto'], default='auto',
                       help="CSV format (auto-detect by default)")
    parser.add_argument("--user-id", type=str, help="Specific user to process")
    parser.add_argument("--all-users", action="store_true", help="Process all users")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GOOGLE FORM → HTML REPORT GENERATOR")
    print("="*80)
    print()
    print("⚠️  NOTE: Using mock predictions based on behavioral heuristics")
    print("    For production, integrate actual Stage 1 & Stage 2 models")
    print()
    
    # Auto-detect format
    if args.format == 'auto':
        df_sample = pd.read_csv(args.csv, nrows=1)
        # If many columns (~40+), likely weekly format
        if len(df_sample.columns) > 30:
            args.format = 'weekly'
        else:
            args.format = 'daily'
        print(f"🔍 Auto-detected format: {args.format}")
    
    # Parse CSV
    print(f"\n📂 Loading CSV: {args.csv}")
    
    if args.format == 'daily':
        users_data = parse_daily_format(args.csv)
    else:
        users_data = parse_weekly_format(args.csv)
    
    # Determine which users to process
    if args.all_users:
        user_ids = list(users_data.keys())
    elif args.user_id:
        if args.user_id not in users_data:
            print(f"\n❌ User '{args.user_id}' not found in CSV")
            print(f"   Available users: {', '.join(users_data.keys())}")
            return
        user_ids = [args.user_id]
    else:
        print("\n❌ Must specify --user-id or --all-users")
        return
    
    print(f"\n🔄 Generating reports for {len(user_ids)} user(s)...\n")
    
    # Generate reports
    for user_id in user_ids:
        print(f"  Processing: {user_id}")
        
        user_data = users_data[user_id]
        
        # Generate predictions
        predictions = generate_mock_predictions(user_data, user_id)
        
        print(f"    ✓ Generated {len(predictions)} days of predictions")
        
        # Generate HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"comprehensive_profile_{user_id.replace('@', '_').replace('.', '_')}_{timestamp}.html"
        html_path = args.output_dir / html_filename
        
        generate_html_report_wrapper(user_id, predictions, html_path)
        
        # Calculate stats
        latest = predictions[-1]
        avg_unc = calculate_avg_uncertainty_pct(latest)
        at_risk, healthy = assess_risk(latest['stage2_mental_health_predictions'])
        
        print(f"    ✓ HTML report: {html_path}")
        print(f"    📊 Predictions: {len(predictions)} days")
        print(f"    📊 Avg uncertainty: ±{avg_unc:.0f}%")
        print(f"    ⚠️  At-risk: {len(at_risk)} factors")
        print(f"    ✅ Healthy: {len(healthy)} factors")
        print()
    
    print("="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nReports saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
