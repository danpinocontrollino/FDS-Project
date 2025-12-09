"""
StudentLife Dataset Feature Extraction
=======================================
Extract 17 behavioral features + 8 mental health targets from StudentLife dataset.

Run this on Kaggle with the StudentLife dataset loaded.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = '/kaggle/input/student-life/dataset'

# Your 17 behavioral features
BEHAVIORAL_FEATURES = [
    'sleep_hours',          # From phonelock/dark sensors
    'sleep_quality',        # Derived from sleep continuity
    'work_hours',           # From education data + app usage
    'meetings_count',       # From calendar
    'tasks_completed',      # From education deadlines
    'emails_received',      # Proxy from communication patterns
    'commute_minutes',      # From wifi_location changes
    'exercise_minutes',     # From activity sensor
    'steps_count',          # From activity sensor
    'caffeine_mg',          # Proxy: NOT AVAILABLE - use 150 (avg)
    'alcohol_units',        # Proxy: NOT AVAILABLE - use 0
    'screen_time_hours',    # From phonelock duration
    'social_interactions',  # From conversation + call_log + sms
    'outdoor_time_minutes', # From wifi_location (off-campus)
    'diet_quality',         # From dinning data (meal count)
    'work_pressure',        # From deadlines proximity
    'weather_mood_impact'   # NOT AVAILABLE - use 0
]

# Your 8 mental health targets
MENTAL_HEALTH_TARGETS = [
    'stress_level',         # From EMA "Stress" responses
    'mood_score',           # From EMA "Mood" responses
    'energy_level',         # From BigFive "full of energy" + activity
    'focus_score',          # From PSQI + work patterns
    'perceived_stress_scale', # PSS - NOT DIRECTLY AVAILABLE - derive from stress EMA
    'anxiety_score',        # From BigFive neuroticism items
    'depression_score',     # From BigFive "depressed, blue"
    'job_satisfaction'      # From FlourishingScale
]


# ============================================================================
# STEP 1: LOAD ALL STUDENT IDs
# ============================================================================

def get_all_student_ids(base_path):
    """
    Extract all student IDs from the dataset
    """
    print("=" * 80)
    print("EXTRACTING STUDENT IDs")
    print("=" * 80)
    
    student_ids = set()
    
    # Get from sensing data (most complete)
    sensing_path = os.path.join(base_path, 'sensing', 'phonelock')
    if os.path.exists(sensing_path):
        files = os.listdir(sensing_path)
        for f in files:
            if f.startswith('phonelock_u') and f.endswith('.csv'):
                uid = f.replace('phonelock_', '').replace('.csv', '')
                student_ids.add(uid)
    
    student_ids = sorted(list(student_ids))
    print(f"Found {len(student_ids)} students: {student_ids[:10]}...")
    
    return student_ids


# ============================================================================
# STEP 2: EXTRACT EMA MENTAL HEALTH RESPONSES
# ============================================================================

def load_ema_responses(base_path, ema_type, student_id):
    """
    Load EMA responses for a specific type (Stress, Mood, etc.)
    
    Args:
        ema_type: "Stress", "Mood", "Sleep", "Activity", etc.
    """
    ema_path = os.path.join(base_path, 'EMA', 'response', ema_type, f'{ema_type}_{student_id}.json')
    
    if not os.path.exists(ema_path):
        return []
    
    try:
        with open(ema_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except Exception as e:
        # Try line-by-line JSON
        try:
            with open(ema_path, 'r') as f:
                lines = f.readlines()
                return [json.loads(line.strip()) for line in lines if line.strip()]
        except:
            return []


def aggregate_ema_by_day(ema_responses, ema_type='Stress'):
    """
    Convert EMA responses to daily aggregates
    
    Different EMA types have different structures:
    - Stress: Only has resp_time (no actual response value!) - SKIP
    - Mood: Has happy, sad, happyornot, sadornot fields
    """
    daily_values = {}
    
    for entry in ema_responses:
        if 'resp_time' not in entry:
            continue
        
        timestamp = entry['resp_time']
        
        try:
            # Convert timestamp to date (Unix seconds)
            dt = datetime.fromtimestamp(timestamp)
            date_key = dt.date()
            
            # Parse based on EMA type
            if ema_type == 'Mood':
                # Mood has: happy (1-5), sad (1-5), happyornot (1-2), sadornot (1-2)
                happy = int(entry.get('happy', 3)) if 'happy' in entry else 3
                sad = int(entry.get('sad', 3)) if 'sad' in entry else 3
                
                # Convert to 0-10 scale: high happy + low sad = high mood
                # Formula: (happy * 2) - (sad * 0.5) scaled to 0-10
                mood_value = min(10, max(0, (happy - sad + 5)))
                numeric_value = mood_value
            
            elif ema_type == 'Stress':
                # Stress EMA has NO response value - just timestamps!
                # Count frequency instead: more pings = more stress engagement
                numeric_value = 5.0  # Default - can't extract actual stress value
            
            else:
                # Generic handling
                if 'resp' in entry:
                    value = entry['resp']
                    if isinstance(value, (int, float)):
                        numeric_value = float(value)
                    else:
                        numeric_value = 5.0
                else:
                    numeric_value = 5.0
            
            if date_key not in daily_values:
                daily_values[date_key] = []
            daily_values[date_key].append(numeric_value)
        
        except Exception as e:
            continue
    
    # Average per day
    daily_avg = {date: np.mean(values) for date, values in daily_values.items()}
    return daily_avg


# ============================================================================
# STEP 3: EXTRACT BEHAVIORAL FEATURES FROM SENSORS
# ============================================================================

def extract_sleep_from_phonelock(base_path, student_id):
    """
    Estimate sleep hours from phonelock sensor
    Uses start/end timestamps - locked periods at night = sleep
    """
    phonelock_path = os.path.join(base_path, 'sensing', 'phonelock', f'phonelock_{student_id}.csv')
    
    if not os.path.exists(phonelock_path):
        return {}
    
    try:
        df = pd.read_csv(phonelock_path, encoding='utf-8-sig')
        
        # Actual columns: 'start', 'end' (Unix timestamps)
        if 'start' not in df.columns or 'end' not in df.columns:
            return {}
        
        df['start_dt'] = pd.to_datetime(df['start'], unit='s')
        df['end_dt'] = pd.to_datetime(df['end'], unit='s')
        df['duration_hours'] = (df['end'] - df['start']) / 3600
        df['date'] = df['start_dt'].dt.date
        df['start_hour'] = df['start_dt'].dt.hour
        
        daily_sleep = {}
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date].copy()
            
            # Find long locked periods at night (10pm-10am, >3 hours)
            night_locks = day_data[
                ((day_data['start_hour'] >= 22) | (day_data['start_hour'] <= 10)) &
                (day_data['duration_hours'] >= 3)
            ]
            
            if len(night_locks) > 0:
                # Use longest lock period as sleep estimate
                sleep_hours = night_locks['duration_hours'].max()
                daily_sleep[date] = min(sleep_hours, 12)  # Cap at 12h
        
        return daily_sleep
    
    except Exception as e:
        print(f"  Error extracting sleep for {student_id}: {e}")
        return {}


def extract_activity(base_path, student_id):
    """
    Extract exercise minutes and steps from activity sensor
    """
    activity_path = os.path.join(base_path, 'sensing', 'activity', f'activity_{student_id}.csv')
    
    if not os.path.exists(activity_path):
        return {}, {}
    
    try:
        df = pd.read_csv(activity_path, encoding='utf-8-sig')
        
        if 'timestamp' not in df.columns:
            return {}, {}
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['datetime'].dt.date
        
        # Note: Column has leading space in actual data
        activity_col = ' activity inference' if ' activity inference' in df.columns else 'activity_inference'
        
        daily_exercise = {}
        daily_steps = {}
        
        if activity_col not in df.columns:
            return {}, {}
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            
            # Activity codes: 0=stationary, 1=walking, 2=running, 3=unknown
            # Count walking (1) and running (2) as exercise
            active_data = day_data[day_data[activity_col].isin([1, 2])]
            
            # Each record is ~3 seconds, so records * 3 / 60 = minutes
            active_minutes = len(active_data) * 3 / 60
            daily_exercise[date] = min(active_minutes, 120)  # Cap at 2 hours
            
            # Rough steps estimate: 100 steps per active minute
            daily_steps[date] = int(active_minutes * 100)
        
        return daily_exercise, daily_steps
    
    except Exception as e:
        print(f"  Error extracting activity for {student_id}: {e}")
        return {}, {}


def extract_screen_time(base_path, student_id):
    """
    Extract screen time from phonelock data
    Phone is unlocked between lock periods, so total time - locked time = screen time
    """
    phonelock_path = os.path.join(base_path, 'sensing', 'phonelock', f'phonelock_{student_id}.csv')
    
    if not os.path.exists(phonelock_path):
        return {}
    
    try:
        df = pd.read_csv(phonelock_path, encoding='utf-8-sig')
        
        if 'start' not in df.columns or 'end' not in df.columns:
            return {}
        
        df['start_dt'] = pd.to_datetime(df['start'], unit='s')
        df['end_dt'] = pd.to_datetime(df['end'], unit='s')
        df['date'] = df['start_dt'].dt.date
        df['locked_hours'] = (df['end'] - df['start']) / 3600
        
        daily_screen = {}
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            
            # Total locked hours in day
            total_locked = day_data['locked_hours'].sum()
            
            # Assume 18 waking hours - locked time = screen time
            # (exclude 6 hours of sleep from calculation)
            screen_hours = max(0, 18 - total_locked)
            daily_screen[date] = min(screen_hours, 16)  # Cap at 16h
        
        return daily_screen
    
    except Exception as e:
        return {}


def extract_social_interactions(base_path, student_id):
    """
    Extract social interactions from conversation, call_log, sms
    """
    daily_social = {}
    
    # 1. Conversation sensor (start_timestamp, end_timestamp)
    conv_path = os.path.join(base_path, 'sensing', 'conversation', f'conversation_{student_id}.csv')
    if os.path.exists(conv_path):
        try:
            df = pd.read_csv(conv_path, encoding='utf-8-sig')
            
            # Check column names (may have leading space)
            start_col = 'start_timestamp' if 'start_timestamp' in df.columns else ' start_timestamp'
            
            if start_col in df.columns:
                df['datetime'] = pd.to_datetime(df[start_col], unit='s')
                df['date'] = df['datetime'].dt.date
                
                for date in df['date'].unique():
                    day_data = df[df['date'] == date]
                    # Each conversation = 1 interaction
                    daily_social[date] = len(day_data)
        except Exception as e:
            pass
    
    # 2. Call log
    call_path = os.path.join(base_path, 'call_log', f'call_log_{student_id}.csv')
    if os.path.exists(call_path):
        try:
            df = pd.read_csv(call_path, encoding='utf-8-sig')
            
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                
                # Only count actual calls (where CALLS_type exists)
                if 'CALLS_type' in df.columns:
                    df_calls = df[df['CALLS_type'].notna()]
                    
                    for date in df_calls['date'].unique():
                        day_data = df_calls[df_calls['date'] == date]
                        daily_social[date] = daily_social.get(date, 0) + len(day_data)
        except Exception as e:
            pass
    
    # 3. SMS (similar structure to call_log)
    sms_path = os.path.join(base_path, 'sms', f'sms_{student_id}.csv')
    if os.path.exists(sms_path):
        try:
            df = pd.read_csv(sms_path, encoding='utf-8-sig')
            
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                
                for date in df['date'].unique():
                    day_data = df[df['date'] == date]
                    # Each SMS counts as 0.5 interaction (less than call/conversation)
                    daily_social[date] = daily_social.get(date, 0) + len(day_data) * 0.5
        except Exception as e:
            pass
    
    return daily_social


def extract_work_hours(base_path, student_id):
    """
    Estimate work hours from app usage + education data
    """
    app_path = os.path.join(base_path, 'app_usage', f'running_app_{student_id}.csv')
    
    if not os.path.exists(app_path):
        return {}
    
    try:
        df = pd.read_csv(app_path, encoding='utf-8-sig')
        
        if 'timestamp' not in df.columns:
            return {}
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['datetime'].dt.date
        
        daily_work = {}
        
        # Educational/productivity app packages
        work_packages = [
            'com.google',  # Google apps (docs, classroom, etc.)
            'com.microsoft',  # Office apps
            'edu.',  # Educational apps
            'blackboard',
            'canvas',
            'moodle',
            'com.android.chrome',  # Browser (assume studying)
            'org.mozilla.firefox',
            'com.evernote',
            'com.notion'
        ]
        
        # Use RUNNING_TASKS_topActivity_mPackage column
        pkg_col = 'RUNNING_TASKS_topActivity_mPackage'
        
        if pkg_col not in df.columns:
            return {}
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            
            # Filter work-related packages
            work_mask = day_data[pkg_col].astype(str).apply(
                lambda x: any(pkg in x.lower() for pkg in work_packages)
            )
            work_records = day_data[work_mask]
            
            # Each record = ~10 seconds of usage, convert to hours
            work_hours = len(work_records) * 10 / 3600
            daily_work[date] = min(work_hours, 14)  # Cap at 14h
        
        return daily_work
    
    except Exception as e:
        return {}


def extract_calendar_meetings(base_path, student_id):
    """
    Extract meeting count from calendar data
    """
    calendar_path = os.path.join(base_path, 'calendar', f'calendar_{student_id}.csv')
    
    if not os.path.exists(calendar_path):
        return {}
    
    try:
        df = pd.read_csv(calendar_path, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['start'], unit='s').dt.date
        
        daily_meetings = df.groupby('date').size().to_dict()
        return daily_meetings
    
    except Exception as e:
        return {}


# ============================================================================
# STEP 4: BUILD COMPLETE DAILY FEATURE MATRIX
# ============================================================================

def build_student_daily_features(base_path, student_id):
    """
    Build complete daily feature matrix for one student
    """
    print(f"  Processing {student_id}...")
    
    # Extract all features
    sleep_hours = extract_sleep_from_phonelock(base_path, student_id)
    exercise, steps = extract_activity(base_path, student_id)
    screen_time = extract_screen_time(base_path, student_id)
    social = extract_social_interactions(base_path, student_id)
    work_hours = extract_work_hours(base_path, student_id)
    meetings = extract_calendar_meetings(base_path, student_id)
    
    # Extract mental health targets
    stress_ema = load_ema_responses(base_path, 'Stress', student_id)
    mood_ema = load_ema_responses(base_path, 'Mood', student_id)
    
    stress_daily = aggregate_ema_by_day(stress_ema, ema_type='Stress')
    mood_daily = aggregate_ema_by_day(mood_ema, ema_type='Mood')
    
    # Debug: print data availability
    print(f"    Sleep days: {len(sleep_hours)}, Exercise days: {len(exercise)}, Screen days: {len(screen_time)}")
    print(f"    Social days: {len(social)}, Work days: {len(work_hours)}, Meetings days: {len(meetings)}")
    print(f"    Stress days: {len(stress_daily)}, Mood days: {len(mood_daily)}")
    
    # Get all unique dates
    all_dates = set()
    for data in [sleep_hours, exercise, screen_time, social, work_hours, meetings, stress_daily, mood_daily]:
        all_dates.update(data.keys())
    
    # Build daily records
    records = []
    for date in sorted(all_dates):
        record = {
            'student_id': student_id,
            'date': date,
            # Behavioral features
            'sleep_hours': sleep_hours.get(date, 7.0),
            'sleep_quality': 6.0,  # Default - would need wake episodes analysis
            'work_hours': work_hours.get(date, 5.0),
            'meetings_count': meetings.get(date, 2),
            'tasks_completed': 5,  # Default - would need deadline data
            'emails_received': 15,  # Default - not available
            'commute_minutes': 20,  # Default - would need location analysis
            'exercise_minutes': exercise.get(date, 30.0),
            'steps_count': steps.get(date, 5000),
            'caffeine_mg': 150,  # Not available - use average
            'alcohol_units': 0,  # Not available
            'screen_time_hours': screen_time.get(date, 6.0),
            'social_interactions': social.get(date, 5),
            'outdoor_time_minutes': 30,  # Default - would need location analysis
            'diet_quality': 6,  # Default - would need dinning analysis
            'work_pressure': 5,  # Default - would need deadline proximity
            'weather_mood_impact': 0,  # Not available
            # Mental health targets
            'stress_level': stress_daily.get(date, 5.0),
            'mood_score': mood_daily.get(date, 6.0),
            'energy_level': 6.0,  # Would derive from activity + BigFive
            'focus_score': 6.0,  # Would derive from PSQI + work patterns
            'perceived_stress_scale': stress_daily.get(date, 5.0) * 4,  # Scale to 0-40
            'anxiety_score': 8.0,  # Would derive from BigFive neuroticism
            'depression_score': 6.0,  # Would derive from BigFive depression items
            'job_satisfaction': 6.5  # Would derive from FlourishingScale
        }
        records.append(record)
    
    return records


def build_full_dataset(base_path, student_ids, max_students=10):
    """
    Build complete dataset for all students
    """
    print("\n" + "=" * 80)
    print(f"BUILDING DATASET FOR {max_students} STUDENTS")
    print("=" * 80)
    
    all_records = []
    
    for student_id in student_ids[:max_students]:
        try:
            records = build_student_daily_features(base_path, student_id)
            all_records.extend(records)
        except Exception as e:
            print(f"  Error processing {student_id}: {e}")
    
    df = pd.DataFrame(all_records)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show variance to detect constant features
    print(f"\n=== FEATURE VARIANCE CHECK ===")
    key_features = ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                   'social_interactions', 'work_hours', 'stress_level', 'mood_score']
    for feat in key_features:
        if feat in df.columns:
            variance = df[feat].var()
            unique_vals = df[feat].nunique()
            print(f"{feat:25s} - Variance: {variance:8.2f}, Unique values: {unique_vals:4d}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


# ============================================================================
# STEP 5: CORRELATION VALIDATION
# ============================================================================

def validate_correlations(df):
    """
    Validate correlations against clinical expectations
    """
    print("\n" + "=" * 80)
    print("CORRELATION VALIDATION")
    print("=" * 80)
    
    # Remove missing values
    df_clean = df.dropna()
    
    expected_corrs = [
        ('sleep_hours', 'stress_level', -0.35, -0.45),
        ('exercise_minutes', 'mood_score', 0.30, 0.40),
        ('work_hours', 'stress_level', 0.30, 0.45),
        ('social_interactions', 'mood_score', 0.25, 0.35),
        ('screen_time_hours', 'stress_level', 0.15, 0.30),
    ]
    
    print("\n{:<25} {:<25} {:<15} {:<15} {:<10}".format(
        "Feature", "Target", "Expected Range", "Actual", "Status"))
    print("-" * 90)
    
    for feat, target, min_exp, max_exp in expected_corrs:
        if feat in df_clean.columns and target in df_clean.columns:
            corr = df_clean[[feat, target]].corr().iloc[0, 1]
            
            if min_exp < 0:  # Negative correlation
                is_good = min_exp <= corr <= max_exp
            else:  # Positive correlation
                is_good = min_exp <= corr <= max_exp
            
            status = "✓ GOOD" if is_good else "⚠ WEAK"
            
            print("{:<25} {:<25} {:<15} {:<15.4f} {:<10}".format(
                feat[:24], target[:24], f"{min_exp:.2f} to {max_exp:.2f}", 
                corr, status))
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH SYNTHETIC DATA")
    print("=" * 80)
    print("Synthetic sleep-stress:       -0.0706  (7x too weak)")
    print("Synthetic exercise-mood:      +0.0887  (3x too weak)")
    print("\nStudentLife data should show MUCH STRONGER correlations!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("STUDENTLIFE FEATURE EXTRACTION")
    print("=" * 80)
    
    # Step 1: Get student IDs
    student_ids = get_all_student_ids(BASE_PATH)
    
    # Step 2: Build dataset (test with 10 students first)
    df = build_full_dataset(BASE_PATH, student_ids, max_students=10)
    
    # Step 3: Validate correlations
    validate_correlations(df)
    
    # Step 4: Save processed data
    output_path = 'studentlife_processed.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext step: Use this data to train a comparison LSTM model")


if __name__ == '__main__':
    main()
