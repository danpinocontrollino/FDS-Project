"""
Train LSTM on StudentLife Dataset - KAGGLE VERSION
===================================================
Trains a NEW LSTM model on real StudentLife data (674 records, 10 students).
This model will be compared against the synthetic-trained model to show
the impact of training data quality on predictions.

Run on Kaggle with StudentLife dataset loaded.

Key Differences from Synthetic Training:
- Only 674 records (vs 1.5M synthetic)
- Real behavioral patterns with variance
- Sparse mental health targets (mostly defaults)
- Focus on behavioral features, limited ground truth

Output: mental_health_lstm_studentlife.pt
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# ============================================================================
# KAGGLE PATHS
# ============================================================================

BASE_PATH = '/kaggle/input/student-life/dataset'
OUTPUT_DIR = '/kaggle/working'
MODEL_OUTPUT = os.path.join(OUTPUT_DIR, 'mental_health_lstm_studentlife.pt')

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class PredictionHead(nn.Module):
    """Individual prediction head for one target."""
    def __init__(self, input_dim, target_name):
        super().__init__()
        self.target_name = target_name
        self.regression = nn.Linear(input_dim, 1)
        self.classification = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        reg_output = self.regression(x)
        cls_output = self.classification(x)
        return reg_output.squeeze(-1), cls_output.squeeze(-1)


class MentalHealthPredictor(nn.Module):
    """Multi-task LSTM for mental health prediction."""
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
        
        self.prediction_heads = nn.ModuleDict({
            target: PredictionHead(hidden_dim, target) for target in targets
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


# ============================================================================
# DATA EXTRACTION (from kaggle_studentlife_extraction.py)
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


def load_ema_responses(base_path, ema_type, student_id):
    ema_path = os.path.join(base_path, 'EMA', 'response', ema_type, 
                           f'{ema_type}_{student_id}.json')
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
        df['end_dt'] = pd.to_datetime(df['end'], unit='s')
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
    # Conversation
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
    # Calls
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
    # SMS
    sms_path = os.path.join(base_path, 'sms', f'sms_{student_id}.csv')
    if os.path.exists(sms_path):
        try:
            df = pd.read_csv(sms_path, encoding='utf-8-sig')
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                for date in df['date'].unique():
                    daily_social[date] = daily_social.get(date, 0) + len(df[df['date'] == date]) * 0.5
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
        work_packages = ['com.google', 'com.microsoft', 'edu.', 'blackboard', 'canvas', 'moodle', 'com.android.chrome', 'org.mozilla.firefox', 'com.evernote', 'com.notion']
        pkg_col = 'RUNNING_TASKS_topActivity_mPackage'
        if pkg_col not in df.columns:
            return {}
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
# DATASET CLASS
# ============================================================================

class StudentLifeDataset(Dataset):
    """PyTorch Dataset for StudentLife sequences."""
    def __init__(self, df, feature_cols, target_cols, sequence_length=7):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.sequence_length = sequence_length
        self.samples = self._create_sequences()
    
    def _create_sequences(self):
        samples = []
        for student_id in self.df['student_id'].unique():
            student_data = self.df[self.df['student_id'] == student_id].sort_values('date')
            for i in range(len(student_data) - self.sequence_length + 1):
                window = student_data.iloc[i:i+self.sequence_length]
                features = window[self.feature_cols].values.astype(np.float32)
                targets = window.iloc[-1][self.target_cols].values.astype(np.float32)
                samples.append((features, targets))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, targets = self.samples[idx]
        return torch.FloatTensor(features), torch.FloatTensor(targets)


# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    """Main training function."""
    print("="*80)
    print("TRAINING LSTM ON STUDENTLIFE DATA")
    print("="*80)
    
    # Extract data
    print("\nExtracting StudentLife data...")
    student_ids = get_all_student_ids(BASE_PATH)
    print(f"Found {len(student_ids)} students")
    
    all_records = []
    for student_id in student_ids[:10]:
        try:
            records = build_student_daily_features(BASE_PATH, student_id)
            all_records.extend(records)
            print(f"  {student_id}: {len(records)} days")
        except Exception as e:
            print(f"  Error with {student_id}: {e}")
    
    df = pd.DataFrame(all_records)
    print(f"\n✓ Dataset: {df.shape}")
    
    # Define features and targets
    feature_cols = ['sleep_hours', 'sleep_quality', 'work_hours', 'meetings_count',
                   'tasks_completed', 'emails_received', 'commute_minutes', 'exercise_minutes',
                   'steps_count', 'caffeine_mg', 'alcohol_units', 'screen_time_hours',
                   'social_interactions', 'outdoor_time_minutes', 'diet_quality',
                   'work_pressure', 'weather_mood_impact']
    
    target_cols = ['stress_level', 'mood_score', 'energy_level', 'focus_score',
                  'perceived_stress_scale', 'anxiety_score', 'depression_score', 'job_satisfaction']
    
    # Normalize features
    scaler_mean = df[feature_cols].mean().values
    scaler_std = df[feature_cols].std().values
    scaler_std[scaler_std == 0] = 1.0  # Avoid division by zero
    
    df[feature_cols] = (df[feature_cols] - scaler_mean) / scaler_std
    
    # Split by student (80/20)
    student_ids_list = df['student_id'].unique()
    train_students, val_students = train_test_split(student_ids_list, test_size=0.2, random_state=42)
    
    train_df = df[df['student_id'].isin(train_students)]
    val_df = df[df['student_id'].isin(val_students)]
    
    print(f"\nTrain: {len(train_df)} records ({len(train_students)} students)")
    print(f"Val: {len(val_df)} records ({len(val_students)} students)")
    
    # Create datasets
    train_dataset = StudentLifeDataset(train_df, feature_cols, target_cols)
    val_dataset = StudentLifeDataset(val_df, feature_cols, target_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    model = MentalHealthPredictor(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_layers=2,
        encoder_type='lstm',
        targets=target_cols
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"\n✓ Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    print(f"\nStarting training...")
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = 0
            for i, target_name in enumerate(target_cols):
                if target_name in outputs:
                    reg_out, _ = outputs[target_name]
                    loss += criterion(reg_out, targets[:, i])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = 0
                for i, target_name in enumerate(target_cols):
                    if target_name in outputs:
                        reg_out, _ = outputs[target_name]
                        loss += criterion(reg_out, targets[:, i])
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            checkpoint = {
                'model_state': model.state_dict(),
                'feature_cols': feature_cols,
                'targets': target_cols,
                'scaler_mean': scaler_mean,
                'scaler_std': scaler_std,
                'epoch': epoch,
                'val_loss': val_loss,
                'training_info': {
                    'dataset': 'StudentLife',
                    'num_records': len(df),
                    'num_students': len(student_ids_list),
                    'trained_on': datetime.now().isoformat()
                }
            }
            torch.save(checkpoint, MODEL_OUTPUT)
    
    print(f"\n{'='*80}")
    print(f"✓ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved: {MODEL_OUTPUT}")
    print("\nDownload this model and compare against synthetic-trained model!")


if __name__ == '__main__':
    train_model()
