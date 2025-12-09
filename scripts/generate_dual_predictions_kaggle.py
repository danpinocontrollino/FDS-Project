"""
Dual Model Prediction Comparison - KAGGLE VERSION
==================================================
Runs BOTH models on the same StudentLife students and compares predictions:
1. Synthetic-trained model (mental_health_lstm.pt) - 1.5M records
2. Real-trained model (mental_health_lstm_studentlife.pt) - 674 records

Creates confrontation data showing:
- Same behavioral input → different predictions
- Which model is closer to actual EMA responses
- Impact of training data quality

Output: dual_predictions_comparison.json

HOW TO RUN ON KAGGLE:
1. Upload BOTH model files:
   - mental_health_lstm.pt (synthetic-trained)
   - mental_health_lstm_studentlife.pt (real-trained)
2. Add StudentLife dataset
3. Run this script
4. Download dual_predictions_comparison.json

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import os
from datetime import datetime

# ============================================================================
# KAGGLE PATHS
# ============================================================================

BASE_PATH = '/kaggle/input/student-life/dataset'
OUTPUT_PATH = '/kaggle/working/dual_predictions_comparison.json'

# Auto-detect model locations
def find_model_path(model_name):
    """Try multiple locations for models."""
    possible_paths = [
        f'/kaggle/input/mental-health-models/{model_name}',  # Dataset upload
        f'/kaggle/working/{model_name}',  # Direct upload
        f'/kaggle/input/mental-health/{model_name}',  # Alternative dataset name
        f'models/saved/{model_name}'  # Local path
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"  ✓ Found {model_name} at: {path}")
            return path
    
    # Show what we checked
    print(f"  ✗ Could not find {model_name}. Checked:")
    for path in possible_paths:
        print(f"    - {path}")
    raise FileNotFoundError(f"Could not find {model_name} in any expected location.")

# These will be set in run_dual_predictions()
SYNTHETIC_MODEL_PATH = None
REAL_MODEL_PATH = None

# ============================================================================
# MODEL DEFINITION - Support BOTH architectures
# ============================================================================

class PredictionHead(nn.Module):
    """Prediction head WITH shared layer (original synthetic model)."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regression = nn.Linear(hidden_dim, 1)
        self.classification = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        reg = self.regression(h).squeeze(-1)
        cls = self.classification(h).squeeze(-1)
        return reg, cls


class SimplePredictionHead(nn.Module):
    """Simple prediction head WITHOUT shared layer (StudentLife model)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.regression = nn.Linear(input_dim, 1)
        self.classification = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor):
        reg = self.regression(x).squeeze(-1)
        cls = self.classification(x).squeeze(-1)
        return reg, cls


class MentalHealthPredictor(nn.Module):
    """Multi-task LSTM - ORIGINAL architecture with shared_repr."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 encoder_type='lstm', targets=None):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        
        if targets is None:
            targets = ['stress_level', 'mood_score', 'energy_level', 'focus_score',
                      'perceived_stress_scale', 'anxiety_score', 'depression_score', 
                      'job_satisfaction']
        
        self.targets = targets
        
        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                  batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        elif encoder_type == 'gru':
            self.encoder = nn.GRU(input_dim, hidden_dim, num_layers,
                                 batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        # Shared representation layer (LayerNorm + Linear)
        self.shared_repr = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multiple prediction heads
        self.heads = nn.ModuleDict({
            target: PredictionHead(hidden_dim, 64) for target in targets
        })
    
    def forward(self, x):
        if self.encoder_type in ['lstm', 'gru']:
            encoded, _ = self.encoder(x)
            last_hidden = encoded[:, -1, :]
        else:
            last_hidden = self.encoder(x)
        
        shared = self.shared_repr(last_hidden)
        
        outputs = {}
        for target, head in self.heads.items():
            outputs[target] = head(shared)
        
        return outputs


class SimpleMentalHealthPredictor(nn.Module):
    """Multi-task LSTM - SIMPLE architecture without shared_repr (StudentLife training)."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 encoder_type='lstm', targets=None):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        
        if targets is None:
            targets = ['stress_level', 'mood_score', 'energy_level', 'focus_score',
                      'perceived_stress_scale', 'anxiety_score', 'depression_score', 
                      'job_satisfaction']
        
        self.targets = targets
        
        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                  batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        elif encoder_type == 'gru':
            self.encoder = nn.GRU(input_dim, hidden_dim, num_layers,
                                 batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        # Simple prediction heads (no shared layer before)
        self.prediction_heads = nn.ModuleDict({
            target: SimplePredictionHead(hidden_dim) for target in targets
        })
    
    def forward(self, x):
        if self.encoder_type in ['lstm', 'gru']:
            encoded, _ = self.encoder(x)
            last_hidden = encoded[:, -1, :]
        else:
            last_hidden = self.encoder(x)
        
        outputs = {}
        for target, head in self.prediction_heads.items():
            outputs[target] = head(last_hidden)
        
        return outputs


def detect_model_architecture(checkpoint):
    """Detect which architecture a checkpoint uses."""
    state_dict = checkpoint['model_state']
    
    # Check for shared_repr layer (original architecture)
    if any('shared_repr' in key for key in state_dict.keys()):
        return 'original'
    # Check for prediction_heads (simple architecture)
    elif any('prediction_heads' in key for key in state_dict.keys()):
        return 'simple'
    else:
        raise ValueError("Unknown model architecture in checkpoint")


# ============================================================================
# DATA EXTRACTION (reuse from training script)
# ============================================================================

def get_all_student_ids(base_path):
    student_ids = set()
    sensing_path = os.path.join(base_path, 'sensing', 'phonelock')
    if os.path.exists(sensing_path):
        for f in os.listdir(sensing_path):
            if f.startswith('phonelock_u') and f.endswith('.csv'):
                student_ids.add(f.replace('phonelock_', '').replace('.csv', ''))
    return sorted(list(student_ids))


def load_ema_responses(base_path, ema_type, student_id):
    ema_path = os.path.join(base_path, 'EMA', 'response', ema_type, f'{ema_type}_{student_id}.json')
    if not os.path.exists(ema_path):
        return []
    try:
        with open(ema_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except:
        return []


def aggregate_ema_by_day(ema_responses, ema_type='Stress'):
    daily_values = {}
    for entry in ema_responses:
        if 'resp_time' not in entry:
            continue
        try:
            dt = datetime.fromtimestamp(entry['resp_time'])
            date_key = dt.date()
            if ema_type == 'Mood':
                happy = int(entry.get('happy', 3)) if 'happy' in entry else 3
                sad = int(entry.get('sad', 3)) if 'sad' in entry else 3
                numeric_value = min(10, max(0, (happy - sad + 5)))
            else:
                numeric_value = 5.0
            if date_key not in daily_values:
                daily_values[date_key] = []
            daily_values[date_key].append(numeric_value)
        except:
            continue
    return {date: np.mean(values) for date, values in daily_values.items()}


def extract_sleep_from_phonelock(base_path, student_id):
    phonelock_path = os.path.join(base_path, 'sensing', 'phonelock', f'phonelock_{student_id}.csv')
    if not os.path.exists(phonelock_path):
        return {}
    try:
        df = pd.read_csv(phonelock_path, encoding='utf-8-sig')
        if 'start' not in df.columns or 'end' not in df.columns:
            return {}
        df['start_dt'] = pd.to_datetime(df['start'], unit='s')
        df['duration_hours'] = (df['end'] - df['start']) / 3600
        df['date'] = df['start_dt'].dt.date
        df['start_hour'] = df['start_dt'].dt.hour
        daily_sleep = {}
        for date in df['date'].unique():
            day_data = df[df['date'] == date].copy()
            night_locks = day_data[((day_data['start_hour'] >= 22) | (day_data['start_hour'] <= 10)) & (day_data['duration_hours'] >= 3)]
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
        if activity_col not in df.columns:
            return {}, {}
        daily_exercise, daily_steps = {}, {}
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
            daily_screen[date] = min(max(0, 18 - day_data['locked_hours'].sum()), 16)
        return daily_screen
    except:
        return {}


def extract_social_interactions(base_path, student_id):
    daily_social = {}
    # Simplified - just conversation for brevity
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
        pkg_col = 'RUNNING_TASKS_topActivity_mPackage'
        if pkg_col not in df.columns:
            return {}
        work_packages = ['com.google', 'edu.', 'blackboard', 'com.android.chrome']
        daily_work = {}
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            work_mask = day_data[pkg_col].astype(str).apply(lambda x: any(pkg in x.lower() for pkg in work_packages))
            daily_work[date] = min(len(day_data[work_mask]) * 10 / 3600, 14)
        return daily_work
    except:
        return {}


def build_student_daily_features(base_path, student_id):
    sleep_hours = extract_sleep_from_phonelock(base_path, student_id)
    exercise, steps = extract_activity(base_path, student_id)
    screen_time = extract_screen_time(base_path, student_id)
    social = extract_social_interactions(base_path, student_id)
    work_hours = extract_work_hours(base_path, student_id)
    stress_ema = load_ema_responses(base_path, 'Stress', student_id)
    mood_ema = load_ema_responses(base_path, 'Mood', student_id)
    stress_daily = aggregate_ema_by_day(stress_ema, ema_type='Stress')
    mood_daily = aggregate_ema_by_day(mood_ema, ema_type='Mood')
    
    all_dates = set()
    for data in [sleep_hours, exercise, screen_time, social, work_hours, stress_daily, mood_daily]:
        all_dates.update(data.keys())
    
    records = []
    for date in sorted(all_dates):
        record = {
            'student_id': student_id, 'date': date,
            'sleep_hours': sleep_hours.get(date, 7.0), 'sleep_quality': 6.0,
            'work_hours': work_hours.get(date, 5.0), 'meetings_count': 2,
            'tasks_completed': 5, 'emails_received': 15, 'commute_minutes': 20,
            'exercise_minutes': exercise.get(date, 30.0), 'steps_count': steps.get(date, 5000),
            'caffeine_mg': 150, 'alcohol_units': 0, 'screen_time_hours': screen_time.get(date, 6.0),
            'social_interactions': social.get(date, 5), 'outdoor_time_minutes': 30,
            'diet_quality': 6, 'work_pressure': 5, 'weather_mood_impact': 0,
            'stress_level': stress_daily.get(date, 5.0), 'mood_score': mood_daily.get(date, 6.0),
            'energy_level': 6.0, 'focus_score': 6.0, 'perceived_stress_scale': stress_daily.get(date, 5.0) * 4,
            'anxiety_score': 8.0, 'depression_score': 6.0, 'job_satisfaction': 6.5
        }
        records.append(record)
    return records


# ============================================================================
# DUAL PREDICTION
# ============================================================================

def run_dual_predictions():
    """Main comparison function."""
    global SYNTHETIC_MODEL_PATH, REAL_MODEL_PATH
    
    print("="*80)
    print("DUAL MODEL COMPARISON")
    print("="*80)
    
    # Find models first
    print("\nLocating models...")
    SYNTHETIC_MODEL_PATH = find_model_path('mental_health_lstm.pt')
    REAL_MODEL_PATH = find_model_path('mental_health_lstm_studentlife.pt')
    
    # Load both models
    print("\nLoading models...")
    
    print("  1. Synthetic-trained model...")
    synthetic_checkpoint = torch.load(SYNTHETIC_MODEL_PATH, map_location='cpu', weights_only=False)
    synthetic_arch = detect_model_architecture(synthetic_checkpoint)
    print(f"     Architecture: {synthetic_arch}")
    
    if synthetic_arch == 'original':
        synthetic_model = MentalHealthPredictor(
            input_dim=len(synthetic_checkpoint['feature_cols']),
            hidden_dim=128,
            encoder_type='lstm',
            targets=synthetic_checkpoint['targets']
        )
    else:
        synthetic_model = SimpleMentalHealthPredictor(
            input_dim=len(synthetic_checkpoint['feature_cols']),
            hidden_dim=128,
            encoder_type='lstm',
            targets=synthetic_checkpoint['targets']
        )
    
    synthetic_model.load_state_dict(synthetic_checkpoint['model_state'])
    synthetic_model.eval()
    print(f"     ✓ Trained on: Synthetic Kaggle (1.5M records)")
    
    print("  2. Real-trained model...")
    real_checkpoint = torch.load(REAL_MODEL_PATH, map_location='cpu', weights_only=False)
    real_arch = detect_model_architecture(real_checkpoint)
    print(f"     Architecture: {real_arch}")
    
    if real_arch == 'original':
        real_model = MentalHealthPredictor(
            input_dim=len(real_checkpoint['feature_cols']),
            hidden_dim=128,
            encoder_type='lstm',
            targets=real_checkpoint['targets']
        )
    else:
        real_model = SimpleMentalHealthPredictor(
            input_dim=len(real_checkpoint['feature_cols']),
            hidden_dim=128,
            encoder_type='lstm',
            targets=real_checkpoint['targets']
        )
    
    real_model.load_state_dict(real_checkpoint['model_state'])
    real_model.eval()
    print(f"     ✓ Trained on: StudentLife (674 records)")
    
    # Extract StudentLife data
    print("\nExtracting StudentLife data...")
    student_ids = get_all_student_ids(BASE_PATH)
    print(f"Processing {len(student_ids[:10])} students...")
    
    all_records = []
    for student_id in student_ids[:10]:
        records = build_student_daily_features(BASE_PATH, student_id)
        all_records.extend(records)
        print(f"  {student_id}: {len(records)} days")
    
    df = pd.DataFrame(all_records)
    feature_cols = synthetic_checkpoint['feature_cols']
    
    # Run dual predictions
    print(f"\nRunning dual predictions...")
    comparison_results = []
    
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id].sort_values('date')
        
        for i in range(6, len(student_data)):
            window = student_data.iloc[max(0, i-6):i+1]
            if len(window) < 7:
                continue
            
            # Extract features
            feature_matrix = []
            for _, row in window.iterrows():
                day_features = [row.get(feat, 0) for feat in feature_cols]
                feature_matrix.append(day_features)
            
            feature_array = np.array(feature_matrix)
            
            # Predict with BOTH models
            # Synthetic model
            normalized_synthetic = (feature_array - synthetic_checkpoint['scaler_mean']) / synthetic_checkpoint['scaler_std']
            normalized_synthetic = np.nan_to_num(normalized_synthetic, nan=0.0)
            X_synthetic = torch.FloatTensor(normalized_synthetic).unsqueeze(0)
            
            with torch.no_grad():
                outputs_synthetic = synthetic_model(X_synthetic)
            
            # Real model
            normalized_real = (feature_array - real_checkpoint['scaler_mean']) / real_checkpoint['scaler_std']
            normalized_real = np.nan_to_num(normalized_real, nan=0.0)
            X_real = torch.FloatTensor(normalized_real).unsqueeze(0)
            
            with torch.no_grad():
                outputs_real = real_model(X_real)
            
            # Extract predictions
            date_str = str(window.iloc[-1]['date'])
            actual_row = window.iloc[-1]
            
            comparison = {
                'student_id': student_id,
                'date': date_str,
                'behavioral_features': {feat: float(actual_row[feat]) for feat in feature_cols},
                'predictions': {}
            }
            
            for target in synthetic_checkpoint['targets']:
                synthetic_value = outputs_synthetic[target][0].item() if target in outputs_synthetic else None
                real_value = outputs_real[target][0].item() if target in outputs_real else None
                actual_value = float(actual_row[target])
                
                # Check if actual value is NOT a default (5.0, 6.0, 6.5)
                has_actual = actual_value not in [5.0, 6.0, 6.5, 8.0, 20.0]
                
                comparison['predictions'][target] = {
                    'synthetic_prediction': synthetic_value,
                    'real_prediction': real_value,
                    'actual_value': actual_value if has_actual else None,
                    'synthetic_error': abs(synthetic_value - actual_value) if has_actual and synthetic_value else None,
                    'real_error': abs(real_value - actual_value) if has_actual and real_value else None,
                    'winner': 'real' if has_actual and real_value and (abs(real_value - actual_value) < abs(synthetic_value - actual_value)) else 'synthetic' if has_actual else None
                }
            
            comparison_results.append(comparison)
    
    # Calculate summary statistics
    print(f"\n✓ Generated {len(comparison_results)} comparisons")
    
    # Count wins
    synthetic_wins = 0
    real_wins = 0
    ties = 0
    
    for comp in comparison_results:
        for target, pred_data in comp['predictions'].items():
            if pred_data['winner'] == 'synthetic':
                synthetic_wins += 1
            elif pred_data['winner'] == 'real':
                real_wins += 1
            elif pred_data['actual_value'] is not None:
                ties += 1
    
    total_comparisons = synthetic_wins + real_wins + ties
    
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_comparisons': len(comparison_results),
        'total_predictions_with_ground_truth': total_comparisons,
        'model_performance': {
            'synthetic_model': {
                'wins': synthetic_wins,
                'win_rate': synthetic_wins / total_comparisons if total_comparisons > 0 else 0,
                'training_data': 'Synthetic Kaggle (1.5M records)'
            },
            'real_model': {
                'wins': real_wins,
                'win_rate': real_wins / total_comparisons if total_comparisons > 0 else 0,
                'training_data': 'StudentLife (674 records)'
            },
            'ties': ties
        },
        'comparisons': comparison_results
    }
    
    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults:")
    print(f"  Synthetic Model Wins: {synthetic_wins}/{total_comparisons} ({synthetic_wins/total_comparisons*100:.1f}%)")
    print(f"  Real Model Wins:      {real_wins}/{total_comparisons} ({real_wins/total_comparisons*100:.1f}%)")
    print(f"  Ties:                 {ties}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("Download this file for interactive comparison viewer!")


if __name__ == '__main__':
    run_dual_predictions()
