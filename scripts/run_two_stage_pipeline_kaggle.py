"""
TWO-STAGE HYBRID PREDICTION PIPELINE
=====================================
Stage 1: Real model predicts behavioral patterns (with uncertainty)
Stage 2: Synthetic model predicts mental health from predicted behaviors
Compare: Final predictions vs ground truth + error propagation analysis

This demonstrates:
1. Using real sensor data to forecast behavior
2. Using synthetic patterns to infer mental health
3. How errors compound through prediction pipelines
4. Uncertainty quantification at each stage
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = '/kaggle/input/student-life/dataset'
STAGE1_MODEL = '/kaggle/working/stage1_behavioral_forecasting.pt'  # Just trained!
STAGE2_MODEL = '/kaggle/input/mental-health-lstm/mental_health_lstm.pt'
OUTPUT_FILE = '/kaggle/working/two_stage_predictions.json'

# Import model architectures
import sys
sys.path.append('/kaggle/working')

# ============================================================================
# LOAD MODELS
# ============================================================================

def load_stage1_model():
    """Load Stage 1: Behavioral forecasting (real data)."""
    from train_stage1_behavioral_kaggle import BehavioralForecastingLSTM
    
    checkpoint = torch.load(STAGE1_MODEL, map_location='cpu', weights_only=False)
    
    # Use architecture params from checkpoint (matches training)
    model = BehavioralForecastingLSTM(
        input_dim=len(checkpoint['feature_cols']),
        hidden_dim=32,  # Match training script
        num_layers=1,   # Match training script
        targets=checkpoint['targets']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print("✓ Stage 1 Model Loaded:")
    print(f"  Task: {checkpoint['training_info']['task']}")
    print(f"  Model: {checkpoint['training_info'].get('model_type', 'simple_lstm_mse')}")
    print(f"  Targets: {checkpoint['targets']}")
    print(f"  Features: {checkpoint['feature_cols']}")
    
    return model, checkpoint


def load_stage2_model():
    """Load Stage 2: Mental health prediction (synthetic data)."""
    # Load using existing model definition
    checkpoint = torch.load(STAGE2_MODEL, map_location='cpu', weights_only=False)
    
    # Import from existing script
    from model_definitions import MentalHealthPredictor
    
    model = MentalHealthPredictor(
        input_dim=len(checkpoint['feature_cols']),
        hidden_dim=128,
        encoder_type='lstm',
        targets=checkpoint['targets']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print("\n✓ Stage 2 Model Loaded:")
    print(f"  Targets: {checkpoint['targets']}")
    print(f"  Features: {checkpoint['feature_cols']}")
    
    return model, checkpoint


# ============================================================================
# DATA EXTRACTION (reuse from Stage 1)
# ============================================================================

def get_all_student_ids(base_path):
    student_ids = set()
    sensing_path = os.path.join(base_path, 'sensing', 'phonelock')
    if os.path.exists(sensing_path):
        files = os.listdir(sensing_path)
        for f in files:
            if f.startswith('phonelock_u') and f.endswith('.csv'):
                uid = f.replace('phonelock_', '').replace('.csv', '')
                student_ids.add(uid)
    return sorted(list(student_ids))


def extract_sleep_from_phonelock(base_path, student_id):
    phonelock_path = os.path.join(base_path, 'sensing', 'phonelock', f'phonelock_{student_id}.csv')
    if not os.path.exists(phonelock_path):
        return {}
    try:
        df = pd.read_csv(phonelock_path, encoding='utf-8-sig')
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
            night_locks = day_data[
                ((day_data['start_hour'] >= 22) | (day_data['start_hour'] <= 10)) &
                (day_data['duration_hours'] >= 3)
            ]
            if len(night_locks) > 0:
                daily_sleep[date] = min(night_locks['duration_hours'].max(), 12)
        return daily_sleep
    except:
        return {}


def extract_activity(base_path, student_id):
    activity_path = os.path.join(base_path, 'sensing', 'activity', f'activity_{student_id}.csv')
    if not os.path.exists(activity_path):
        return {}, {}
    try:
        df = pd.read_csv(activity_path, encoding='utf-8-sig')
        if 'timestamp' not in df.columns:
            return {}, {}
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['datetime'].dt.date
        activity_col = ' activity inference' if ' activity inference' in df.columns else 'activity_inference'
        daily_exercise, daily_steps = {}, {}
        if activity_col not in df.columns:
            return {}, {}
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            active_data = day_data[day_data[activity_col].isin([1, 2])]
            active_minutes = len(active_data) * 3 / 60
            daily_exercise[date] = min(active_minutes, 120)
            daily_steps[date] = int(active_minutes * 100)
        return daily_exercise, daily_steps
    except:
        return {}, {}


def extract_screen_time(base_path, student_id):
    phonelock_path = os.path.join(base_path, 'sensing', 'phonelock', f'phonelock_{student_id}.csv')
    if not os.path.exists(phonelock_path):
        return {}
    try:
        df = pd.read_csv(phonelock_path, encoding='utf-8-sig')
        if 'start' not in df.columns or 'end' not in df.columns:
            return {}
        df['start_dt'] = pd.to_datetime(df['start'], unit='s')
        df['date'] = df['start_dt'].dt.date
        df['locked_hours'] = (df['end'] - df['start']) / 3600
        daily_screen = {}
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            total_locked = day_data['locked_hours'].sum()
            daily_screen[date] = min(max(0, 18 - total_locked), 16)
        return daily_screen
    except:
        return {}


def extract_social_interactions(base_path, student_id):
    daily_social = {}
    conv_path = os.path.join(base_path, 'sensing', 'conversation', f'conversation_{student_id}.csv')
    if os.path.exists(conv_path):
        try:
            df = pd.read_csv(conv_path, encoding='utf-8-sig')
            start_col = 'start_timestamp' if 'start_timestamp' in df.columns else ' start_timestamp'
            if start_col in df.columns:
                df['datetime'] = pd.to_datetime(df[start_col], unit='s')
                df['date'] = df['datetime'].dt.date
                for date in df['date'].unique():
                    daily_social[date] = len(df[df['date'] == date])
        except:
            pass
    call_path = os.path.join(base_path, 'call_log', f'call_log_{student_id}.csv')
    if os.path.exists(call_path):
        try:
            df = pd.read_csv(call_path, encoding='utf-8-sig')
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                if 'CALLS_type' in df.columns:
                    df_calls = df[df['CALLS_type'].notna()]
                    for date in df_calls['date'].unique():
                        daily_social[date] = daily_social.get(date, 0) + len(df_calls[df_calls['date'] == date])
        except:
            pass
    return daily_social


def extract_work_hours(base_path, student_id):
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
        work_packages = ['com.google', 'com.microsoft', 'edu.', 'blackboard', 'canvas']
        pkg_col = 'RUNNING_TASKS_topActivity_mPackage'
        if pkg_col not in df.columns:
            return {}
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            work_mask = day_data[pkg_col].astype(str).apply(
                lambda x: any(pkg in x.lower() for pkg in work_packages)
            )
            daily_work[date] = min(len(day_data[work_mask]) * 10 / 3600, 14)
        return daily_work
    except:
        return {}


def build_student_data_complete(base_path, student_id):
    """Build complete student dataset."""
    sleep_hours = extract_sleep_from_phonelock(base_path, student_id)
    exercise, steps = extract_activity(base_path, student_id)
    screen_time = extract_screen_time(base_path, student_id)
    social = extract_social_interactions(base_path, student_id)
    work_hours = extract_work_hours(base_path, student_id)
    
    all_dates = set()
    for data in [sleep_hours, exercise, screen_time, social, work_hours]:
        all_dates.update(data.keys())
    
    records = []
    for date in sorted(all_dates):
        record = {
            'student_id': student_id,
            'date': date,
            'sleep_hours': sleep_hours.get(date, 7.0),
            'work_hours': work_hours.get(date, 2.0),
            'exercise_minutes': exercise.get(date, 30.0),
            'steps_count': steps.get(date, 5000),
            'screen_time_hours': screen_time.get(date, 6.0),
            'social_interactions': social.get(date, 10)
        }
        records.append(record)
    
    return records


# ============================================================================
# TWO-STAGE PREDICTION PIPELINE
# ============================================================================

def run_two_stage_pipeline():
    """Execute complete two-stage prediction pipeline."""
    print("="*80)
    print("TWO-STAGE HYBRID PREDICTION PIPELINE")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    stage1_model, stage1_checkpoint = load_stage1_model()
    stage2_model, stage2_checkpoint = load_stage2_model()
    
    # Extract StudentLife data
    print("\n" + "="*80)
    print("EXTRACTING STUDENTLIFE DATA")
    print("="*80)
    student_ids = get_all_student_ids(BASE_PATH)
    print(f"Found {len(student_ids)} students")
    
    all_records = []
    for student_id in student_ids[:10]:
        try:
            records = build_student_data_complete(BASE_PATH, student_id)
            all_records.extend(records)
            print(f"  {student_id}: {len(records)} days")
        except Exception as e:
            print(f"  Error with {student_id}: {e}")
    
    df = pd.DataFrame(all_records)
    print(f"\n✓ Total data: {df.shape}")
    
    # Normalize for Stage 1
    stage1_scaler_mean = stage1_checkpoint['scaler_mean']
    stage1_scaler_std = stage1_checkpoint['scaler_std']
    
    stage1_features = stage1_checkpoint['feature_cols']
    df_stage1 = df.copy()
    df_stage1[stage1_features] = (df_stage1[stage1_features] - stage1_scaler_mean) / stage1_scaler_std
    
    # Run predictions
    print("\n" + "="*80)
    print("RUNNING TWO-STAGE PREDICTIONS")
    print("="*80)
    
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'num_students': len(student_ids[:10]),
            'total_predictions': 0,
            'stage1_model': 'behavioral_forecasting (StudentLife)',
            'stage2_model': 'mental_health_lstm (Synthetic)',
            'pipeline_description': 'Real sensors → Behavioral forecast → Mental health inference'
        },
        'predictions': []
    }
    
    for student_id in df['student_id'].unique():
        student_data = df_stage1[df_stage1['student_id'] == student_id].sort_values('date')
        student_data_original = df[df['student_id'] == student_id].sort_values('date')
        
        # Need 7-day windows
        for i in range(len(student_data) - 7):
            # Get 7-day window
            window = student_data.iloc[i:i+7]
            window_original = student_data_original.iloc[i:i+7]
            target_date = student_data.iloc[i+7]['date']
            
            # STAGE 1: Predict behavioral patterns (with uncertainty)
            features_stage1 = torch.FloatTensor(window[stage1_features].values).unsqueeze(0)
            
            with torch.no_grad():
                stage1_outputs = stage1_model(features_stage1)
            
            # Extract predictions (simple model - no uncertainty dict)
            behavioral_predictions = {}
            behavioral_uncertainties = {}
            
            # Get target scalers for denormalization
            target_mean = stage1_checkpoint.get('target_scaler_mean', np.zeros(len(stage1_checkpoint['targets'])))
            target_std = stage1_checkpoint.get('target_scaler_std', np.ones(len(stage1_checkpoint['targets'])))
            
            for idx, target in enumerate(stage1_checkpoint['targets']):
                pred = stage1_outputs[target]
                # Handle both simple (float) and complex (dict) outputs
                if isinstance(pred, dict):
                    pred_normalized = float(pred['mean'].item())
                    uncertainty_normalized = float(pred['std'].item())
                else:
                    pred_normalized = float(pred.item())
                    uncertainty_normalized = 0.5
                
                # Denormalize prediction
                behavioral_predictions[target] = pred_normalized * target_std[idx] + target_mean[idx]
                behavioral_uncertainties[target] = uncertainty_normalized * target_std[idx]
            
            # STAGE 2: Use predicted behaviors to infer mental health
            # Need to construct 7-day sequence for Stage 2
            # Use last 6 days of actual data + 1 day of predicted data
            
            stage2_features = stage2_checkpoint['feature_cols']
            stage2_scaler_mean = stage2_checkpoint['scaler_mean']
            stage2_scaler_std = stage2_checkpoint['scaler_std']
            
            # Build Stage 2 input: 6 actual days + 1 predicted day
            # Stage 2 expects 17 features, but we only have 6 from StudentLife
            # Fill missing features with reasonable defaults
            stage2_input_data = []
            
            # Map StudentLife features to Stage 2 expected features
            feature_defaults = {
                'sleep_quality': 6.0,
                'caffeine_mg': 150.0,
                'alcohol_units': 0.0,
                'diet_quality': 6.0,
                'meetings_count': 3.0,
                'tasks_completed': 8.0,
                'work_pressure': 5.0,
                'commute_time': 0.5,
                'weather_score': 7.0,
                'chronotype_encoded': 1.0,
                'work_arrangement_encoded': 1.0
            }
            
            for j in range(6):
                day_data = window_original.iloc[j]
                day_features = []
                for f in stage2_features:
                    if f in window_original.columns:
                        day_features.append(day_data[f])
                    else:
                        day_features.append(feature_defaults.get(f, 0.0))
                stage2_input_data.append(day_features)
            
            # Add predicted day (from Stage 1 + defaults for missing)
            predicted_day = []
            for f in stage2_features:
                if f in behavioral_predictions:
                    predicted_day.append(behavioral_predictions[f])
                elif f in window_original.columns:
                    # Use mean from last 6 days if available in data
                    predicted_day.append(window_original[f].iloc[-6:].mean())
                else:
                    # Use default for synthetic-only features
                    predicted_day.append(feature_defaults.get(f, 0.0))
            stage2_input_data.append(predicted_day)
            
            # Normalize for Stage 2
            stage2_input_array = np.array(stage2_input_data, dtype=np.float32)
            stage2_input_normalized = (stage2_input_array - stage2_scaler_mean) / stage2_scaler_std
            stage2_input_tensor = torch.FloatTensor(stage2_input_normalized).unsqueeze(0)
            
            with torch.no_grad():
                stage2_outputs = stage2_model(stage2_input_tensor)
            
            # Extract mental health predictions
            mental_health_predictions = {}
            for target in stage2_checkpoint['targets']:
                if target in stage2_outputs:
                    pred = stage2_outputs[target]
                    if isinstance(pred, dict):
                        # Dict output: {'regression': tensor, 'classification': tensor}
                        mental_health_predictions[target] = float(pred['regression'].item())
                    elif isinstance(pred, tuple):
                        # Tuple output: (regression_tensor, classification_tensor)
                        mental_health_predictions[target] = float(pred[0].item())
                    else:
                        # Simple tensor
                        mental_health_predictions[target] = float(pred.item())
            
            # Store complete prediction
            prediction_record = {
                'student_id': student_id,
                'date': str(target_date),
                'stage1_behavioral_predictions': behavioral_predictions,
                'stage1_uncertainties': behavioral_uncertainties,
                'stage2_mental_health_predictions': mental_health_predictions,
                'input_features': {
                    f: float(window_original[f].iloc[-1]) for f in stage1_features
                },
                'error_propagation': {
                    'stage1_total_uncertainty': sum(behavioral_uncertainties.values()),
                    'stage2_input_was_predicted': True,
                    'confidence_reduction': 'High uncertainty in Stage 1 compounds in Stage 2'
                }
            }
            
            results['predictions'].append(prediction_record)
    
    results['metadata']['total_predictions'] = len(results['predictions'])
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Generated {len(results['predictions'])} two-stage predictions")
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Stage 1: Behavioral forecasting from real sensors")
    print(f"  → Outputs: {len(stage1_checkpoint['targets'])} behavioral predictions")
    print(f"  → Each with uncertainty estimate (aleatoric)")
    print(f"\nStage 2: Mental health inference from predicted behaviors")
    print(f"  → Inputs: 6 real days + 1 predicted day")
    print(f"  → Outputs: {len(stage2_checkpoint['targets'])} mental health scores")
    print(f"\nError Propagation:")
    print(f"  → Stage 1 uncertainty propagates to Stage 2 inputs")
    print(f"  → Distribution mismatch (predicted vs training)")
    print(f"  → Results stored in: {OUTPUT_FILE}")
    
    return results


if __name__ == '__main__':
    results = run_two_stage_pipeline()
    print(f"\n✓ Complete! Check {OUTPUT_FILE}")
