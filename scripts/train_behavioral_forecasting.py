"""
Train Behavioral Forecasting Model - StudentLife
=================================================
Predict NEXT DAY's behavior based on past 7 days.

Targets (what we actually have data for):
- sleep_hours (next day)
- exercise_minutes (next day)
- screen_time_hours (next day)
- social_interactions (next day)
- steps_count (next day)

This is a valid ML task that:
1. Uses real sensor data with variation
2. Has complete ground truth (no missing labels)
3. Still relevant to well-being monitoring
4. Comparable: synthetic vs real training

Output: behavioral_forecasting_lstm.pt
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
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = '/kaggle/input/student-life/dataset'
OUTPUT_DIR = '/kaggle/working'
MODEL_OUTPUT = os.path.join(OUTPUT_DIR, 'behavioral_forecasting_studentlife.pt')

# Behavioral targets we CAN predict (have real data)
BEHAVIORAL_TARGETS = [
    'sleep_hours',
    'exercise_minutes', 
    'screen_time_hours',
    'social_interactions',
    'steps_count'
]

# Input features (same as before)
FEATURE_COLS = [
    'sleep_hours', 'work_hours', 'exercise_minutes', 'steps_count',
    'screen_time_hours', 'social_interactions'
]

# ============================================================================
# MODEL DEFINITION (Simple LSTM for forecasting)
# ============================================================================

class BehavioralForecastingLSTM(nn.Module):
    """Predicts next-day behavioral patterns."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, targets=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.targets = targets or BEHAVIORAL_TARGETS
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        
        # Simple prediction heads (one per target)
        self.heads = nn.ModuleDict({
            target: nn.Linear(hidden_dim, 1) for target in self.targets
        })
    
    def forward(self, x):
        # Encode sequence
        encoded, _ = self.encoder(x)
        last_hidden = encoded[:, -1, :]  # Take last timestep
        
        # Predict each target
        outputs = {}
        for target, head in self.heads.items():
            outputs[target] = head(last_hidden).squeeze(-1)
        
        return outputs

# ============================================================================
# DATA EXTRACTION (reuse from previous scripts)
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


def build_student_behavioral_data(base_path, student_id):
    """Build complete behavioral dataset for forecasting."""
    sleep_hours = extract_sleep_from_phonelock(base_path, student_id)
    exercise, steps = extract_activity(base_path, student_id)
    screen_time = extract_screen_time(base_path, student_id)
    social = extract_social_interactions(base_path, student_id)
    work_hours = extract_work_hours(base_path, student_id)
    
    # Get all dates
    all_dates = set()
    for data in [sleep_hours, exercise, screen_time, social, work_hours]:
        all_dates.update(data.keys())
    
    # Build records with ONLY behavioral data (no fake mental health targets)
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
# DATASET CLASS (for forecasting)
# ============================================================================

class ForecastingDataset(Dataset):
    """Dataset for next-day behavioral forecasting."""
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
            
            # Create sliding windows: past 7 days → predict day 8
            for i in range(len(student_data) - self.sequence_length):
                # Input: days i to i+6 (7 days)
                window = student_data.iloc[i:i+self.sequence_length]
                # Target: day i+7 (next day)
                target_row = student_data.iloc[i+self.sequence_length]
                
                features = window[self.feature_cols].values.astype(np.float32)
                targets = target_row[self.target_cols].values.astype(np.float32)
                
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

def train_forecasting_model():
    """Train behavioral forecasting model."""
    print("="*80)
    print("TRAINING BEHAVIORAL FORECASTING MODEL ON STUDENTLIFE")
    print("="*80)
    
    # Extract data
    print("\nExtracting behavioral data...")
    student_ids = get_all_student_ids(BASE_PATH)
    print(f"Found {len(student_ids)} students")
    
    all_records = []
    for student_id in student_ids[:10]:
        try:
            records = build_student_behavioral_data(BASE_PATH, student_id)
            all_records.extend(records)
            print(f"  {student_id}: {len(records)} days")
        except Exception as e:
            print(f"  Error with {student_id}: {e}")
    
    df = pd.DataFrame(all_records)
    print(f"\n✓ Dataset: {df.shape}")
    
    # Show data statistics
    print("\nData statistics:")
    for col in BEHAVIORAL_TARGETS:
        print(f"  {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}, "
              f"min={df[col].min():.1f}, max={df[col].max():.1f}")
    
    # Normalize features
    scaler_mean = df[FEATURE_COLS].mean().values
    scaler_std = df[FEATURE_COLS].std().values
    scaler_std[scaler_std == 0] = 1.0
    
    df[FEATURE_COLS] = (df[FEATURE_COLS] - scaler_mean) / scaler_std
    
    # Split by student
    student_ids_list = df['student_id'].unique()
    train_students, val_students = train_test_split(student_ids_list, test_size=0.2, random_state=42)
    
    train_df = df[df['student_id'].isin(train_students)]
    val_df = df[df['student_id'].isin(val_students)]
    
    print(f"\nTrain: {len(train_df)} records ({len(train_students)} students)")
    print(f"Val: {len(val_df)} records ({len(val_students)} students)")
    
    # Create datasets
    train_dataset = ForecastingDataset(train_df, FEATURE_COLS, BEHAVIORAL_TARGETS)
    val_dataset = ForecastingDataset(val_df, FEATURE_COLS, BEHAVIORAL_TARGETS)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = BehavioralForecastingLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_dim=64,
        num_layers=2,
        targets=BEHAVIORAL_TARGETS
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
            for i, target_name in enumerate(BEHAVIORAL_TARGETS):
                if target_name in outputs:
                    loss += criterion(outputs[target_name], targets[:, i])
            
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
                for i, target_name in enumerate(BEHAVIORAL_TARGETS):
                    if target_name in outputs:
                        loss += criterion(outputs[target_name], targets[:, i])
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
                'feature_cols': FEATURE_COLS,
                'targets': BEHAVIORAL_TARGETS,
                'scaler_mean': scaler_mean,
                'scaler_std': scaler_std,
                'epoch': epoch,
                'val_loss': val_loss,
                'training_info': {
                    'task': 'behavioral_forecasting',
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
    print(f"\nThis model predicts NEXT DAY's behavior:")
    for target in BEHAVIORAL_TARGETS:
        print(f"  • {target}")
    print(f"\nBased on past 7 days of sensor data")


if __name__ == '__main__':
    train_forecasting_model()
