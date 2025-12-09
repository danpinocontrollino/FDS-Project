"""
StudentLife Dataset Analysis for Kaggle
========================================
This script processes the StudentLife dataset to validate the mental health
prediction system against real behavioral data from students.

Run this on Kaggle with the StudentLife dataset loaded.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime, timedelta

# ============================================================================
# STEP 1: EXPLORE DATASET STRUCTURE
# ============================================================================

def explore_dataset_structure(base_path='/kaggle/input/student-life/dataset'):
    """
    Explore all folders and files in the StudentLife dataset
    """
    print("=" * 80)
    print("STUDENTLIFE DATASET STRUCTURE")
    print("=" * 80)
    
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')
    
    return base_path


def load_and_inspect_ema(base_path):
    """
    Load EMA (Ecological Momentary Assessment) data
    This contains self-reported mood, stress, etc.
    """
    print("\n" + "=" * 80)
    print("EMA DATA (Self-Reported Assessments)")
    print("=" * 80)
    
    ema_path = os.path.join(base_path, 'EMA')
    
    # Check for response files
    response_path = os.path.join(ema_path, 'response')
    if os.path.exists(response_path):
        files = [f for f in os.listdir(response_path) if f.endswith('.json')]
        print(f"\nFound {len(files)} EMA response files")
        
        # Load first file as sample
        if files:
            sample_file = os.path.join(response_path, files[0])
            print(f"\nSample file: {files[0]}")
            
            # Try different loading methods
            try:
                with open(sample_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        print("\nFirst entry structure:")
                        print(json.dumps(data[0], indent=2))
                        return data
            except:
                try:
                    # Try line-by-line JSON
                    with open(sample_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            first_entry = json.loads(lines[0])
                            print("\nFirst entry structure:")
                            print(json.dumps(first_entry, indent=2))
                            return [json.loads(line) for line in lines]
                except Exception as e:
                    print(f"Error loading EMA data: {e}")
    
    return None


def load_and_inspect_survey(base_path):
    """
    Load survey data - should contain PHQ-9, GAD-7, PSS scales
    """
    print("\n" + "=" * 80)
    print("SURVEY DATA (Clinical Scales)")
    print("=" * 80)
    
    survey_path = os.path.join(base_path, 'survey')
    
    if os.path.exists(survey_path):
        files = [f for f in os.listdir(survey_path) if f.endswith(('.csv', '.json'))]
        print(f"\nFound {len(files)} survey files")
        
        for file in files[:3]:  # Check first 3 files
            full_path = os.path.join(survey_path, file)
            print(f"\n--- {file} ---")
            
            try:
                if file.endswith('.csv'):
                    df = pd.read_csv(full_path)
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {df.columns.tolist()}")
                    print(f"\nFirst row:\n{df.head(1).to_dict('records')[0]}")
                elif file.endswith('.json'):
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        print(f"Type: {type(data)}")
                        if isinstance(data, list) and len(data) > 0:
                            print(f"First entry: {data[0]}")
                        elif isinstance(data, dict):
                            print(f"Keys: {data.keys()}")
            except Exception as e:
                print(f"Error: {e}")
    
    return None


def load_and_inspect_sensing(base_path):
    """
    Load sensing data - activity, location, screen usage
    """
    print("\n" + "=" * 80)
    print("SENSING DATA (Phone Sensors)")
    print("=" * 80)
    
    sensing_path = os.path.join(base_path, 'sensing')
    
    if os.path.exists(sensing_path):
        files = [f for f in os.listdir(sensing_path) if f.endswith(('.csv', '.json'))]
        print(f"\nFound {len(files)} sensing files")
        
        # Look for specific sensor types
        sensor_types = ['activity', 'audio', 'conversation', 'dark', 'phonecharge', 
                       'phonelock', 'wifi_location']
        
        for sensor in sensor_types:
            sensor_files = [f for f in files if sensor in f.lower()]
            if sensor_files:
                print(f"\n{sensor.upper()}: {len(sensor_files)} files")
                
                # Load sample
                sample = os.path.join(sensing_path, sensor_files[0])
                try:
                    if sample.endswith('.csv'):
                        df = pd.read_csv(sample, nrows=3)
                        print(f"  Columns: {df.columns.tolist()}")
                    elif sample.endswith('.json'):
                        with open(sample, 'r') as f:
                            lines = f.readlines()[:2]
                            if lines:
                                entry = json.loads(lines[0])
                                print(f"  Keys: {list(entry.keys())}")
                except Exception as e:
                    print(f"  Error: {e}")


def load_and_inspect_app_usage(base_path):
    """
    Load app usage data - screen time patterns
    """
    print("\n" + "=" * 80)
    print("APP USAGE DATA")
    print("=" * 80)
    
    app_path = os.path.join(base_path, 'app_usage')
    
    if os.path.exists(app_path):
        files = [f for f in os.listdir(app_path) if f.endswith(('.csv', '.json'))]
        print(f"\nFound {len(files)} app usage files")
        
        if files:
            sample = os.path.join(app_path, files[0])
            print(f"\nSample file: {files[0]}")
            try:
                with open(sample, 'r') as f:
                    lines = f.readlines()[:3]
                    for i, line in enumerate(lines):
                        entry = json.loads(line)
                        print(f"Entry {i}: {entry}")
            except Exception as e:
                print(f"Error: {e}")


# ============================================================================
# STEP 2: FEATURE EXTRACTION TO YOUR 17 FEATURES
# ============================================================================

TARGET_FEATURES = [
    'sleep_hours',          # From activity/phonelock/dark sensors
    'sleep_quality',        # Derived from wake episodes
    'work_hours',           # From app usage (educational apps)
    'meetings_count',       # From calendar data
    'tasks_completed',      # Estimated from productivity apps
    'emails_received',      # From email/communication apps
    'commute_minutes',      # From wifi_location transitions
    'exercise_minutes',     # From activity sensor
    'steps_count',          # From activity sensor
    'caffeine_mg',          # Not directly available - use proxy
    'alcohol_units',        # Not directly available
    'screen_time_hours',    # From phonelock duration
    'social_interactions',  # From conversation/call_log
    'outdoor_time_minutes', # From wifi_location (off-campus)
    'diet_quality',         # From dinning data
    'work_pressure',        # Derived from work_hours + deadlines
    'weather_mood_impact'   # Not available - use neutral value
]

TARGET_MENTAL_HEALTH = [
    'stress_level',         # From EMA responses
    'mood_score',           # From EMA responses  
    'energy_level',         # From EMA responses
    'focus_score',          # From EMA responses
    'perceived_stress_scale', # From survey (PSS)
    'anxiety_score',        # From survey (GAD-7)
    'depression_score',     # From survey (PHQ-9)
    'job_satisfaction'      # Proxy: academic satisfaction
]


def extract_sleep_features(base_path, student_ids):
    """
    Extract sleep hours and quality from sensing data
    """
    print("\n" + "=" * 80)
    print("EXTRACTING SLEEP FEATURES")
    print("=" * 80)
    
    sensing_path = os.path.join(base_path, 'sensing')
    sleep_data = []
    
    for student_id in student_ids[:5]:  # Test with first 5 students
        # Look for phonelock or dark sensor data
        phonelock_file = os.path.join(sensing_path, f'phonelock_{student_id}.json')
        
        if os.path.exists(phonelock_file):
            try:
                with open(phonelock_file, 'r') as f:
                    lines = f.readlines()
                    # Process phonelock events to estimate sleep
                    # Long continuous locked periods at night = sleep
                    print(f"Student {student_id}: {len(lines)} phonelock events")
            except Exception as e:
                print(f"Error processing {student_id}: {e}")
    
    return sleep_data


def build_daily_features(base_path, student_ids):
    """
    Build daily feature matrix for all students
    """
    print("\n" + "=" * 80)
    print("BUILDING DAILY FEATURE MATRIX")
    print("=" * 80)
    
    daily_records = []
    
    # This is a template - will need to be customized based on actual data structure
    for student_id in student_ids[:5]:
        print(f"\nProcessing student {student_id}...")
        
        # Placeholder for daily aggregation logic
        # You'll need to aggregate sensor data by day
        record = {
            'student_id': student_id,
            'date': datetime.now().date(),
            **{feat: 0.0 for feat in TARGET_FEATURES},
            **{target: 0.0 for target in TARGET_MENTAL_HEALTH}
        }
        daily_records.append(record)
    
    df = pd.DataFrame(daily_records)
    print(f"\nCreated dataframe: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


# ============================================================================
# STEP 3: CORRELATION VALIDATION
# ============================================================================

def validate_correlations(df):
    """
    Check if correlations match clinical expectations
    """
    print("\n" + "=" * 80)
    print("CORRELATION VALIDATION vs SYNTHETIC DATA")
    print("=" * 80)
    
    expected_correlations = [
        ('sleep_hours', 'stress_level', 'negative', -0.35, -0.45),
        ('exercise_minutes', 'mood_score', 'positive', 0.30, 0.40),
        ('exercise_minutes', 'stress_level', 'negative', -0.25, -0.35),
        ('work_hours', 'stress_level', 'positive', 0.30, 0.45),
        ('social_interactions', 'depression_score', 'negative', -0.25, -0.35),
        ('screen_time_hours', 'anxiety_score', 'positive', 0.15, 0.30),
    ]
    
    print("\n{:<25} {:<25} {:<10} {:<15} {:<10}".format(
        "Feature", "Target", "Expected", "Actual", "Status"))
    print("-" * 85)
    
    synthetic_comparison = {
        'sleep_hours vs stress': -0.0706,  # Your synthetic data
        'exercise vs mood': 0.0887,
        'exercise vs stress': -0.1689,
        'work_hours vs stress': 0.2035,
    }
    
    for feat, target, direction, min_val, max_val in expected_correlations:
        if feat in df.columns and target in df.columns:
            corr = df[[feat, target]].corr().iloc[0, 1]
            
            # Check if in expected range
            if direction == 'negative':
                is_correct = min_val <= corr <= max_val and corr < 0
            else:
                is_correct = min_val <= corr <= max_val and corr > 0
            
            status = "✓ CORRECT" if is_correct else "✗ WEAK"
            
            print("{:<25} {:<25} {:<10} {:<15.4f} {:<10}".format(
                feat[:24], target[:24], direction, corr, status))
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH SYNTHETIC KAGGLE DATA")
    print("=" * 80)
    print("Synthetic sleep-stress:    -0.0706  (7x too weak)")
    print("Synthetic exercise-mood:   +0.0887  (3x too weak)")
    print("Expected sleep-stress:     -0.35 to -0.45")
    print("Expected exercise-mood:    +0.30 to +0.40")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - run this on Kaggle
    """
    print("=" * 80)
    print("STUDENTLIFE DATASET ANALYSIS")
    print("Mental Health Prediction System - Real Data Validation")
    print("=" * 80)
    
    # Set base path (adjust for Kaggle)
    base_path = '/kaggle/input/student-life/dataset'
    
    # STEP 1: Explore structure
    print("\n[STEP 1] Exploring dataset structure...")
    explore_dataset_structure(base_path)
    
    # STEP 2: Inspect each data source
    print("\n[STEP 2] Inspecting data sources...")
    load_and_inspect_ema(base_path)
    load_and_inspect_survey(base_path)
    load_and_inspect_sensing(base_path)
    load_and_inspect_app_usage(base_path)
    
    # STEP 3: Extract student IDs
    print("\n[STEP 3] Extracting student IDs...")
    ema_response_path = os.path.join(base_path, 'EMA', 'response')
    if os.path.exists(ema_response_path):
        files = [f for f in os.listdir(ema_response_path) if f.endswith('.json')]
        student_ids = [f.replace('response_time_', '').replace('.json', '') for f in files]
        print(f"Found {len(student_ids)} students: {student_ids[:5]}...")
    else:
        student_ids = []
        print("Could not find student IDs - check dataset structure")
    
    # STEP 4: Feature extraction (placeholder)
    # print("\n[STEP 4] Extracting features...")
    # df = build_daily_features(base_path, student_ids)
    
    # STEP 5: Correlation validation
    # print("\n[STEP 5] Validating correlations...")
    # validate_correlations(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the data structure output above")
    print("2. Customize feature extraction functions for actual data format")
    print("3. Build daily aggregation logic")
    print("4. Train comparison model")


if __name__ == '__main__':
    main()
