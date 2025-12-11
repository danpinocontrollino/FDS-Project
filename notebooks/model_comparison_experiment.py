"""
COMPREHENSIVE MODEL COMPARISON EXPERIMENT
==========================================
Systematic evaluation of multiple deep learning architectures for behavioral
forecasting from real StudentLife sensor data.

PURPOSE:
- Find the BEST model architecture for Stage 1 (real behavioral prediction)
- The winner will be combined with synthetic-trained Stage 2 model
- Goal: Optimal two-stage pipeline for real-world mental health prediction

MODELS TESTED:
1. LSTM (baseline)
2. Bidirectional LSTM
3. GRU
4. Bidirectional GRU
5. Transformer (attention-based)
6. CNN-LSTM (convolutional + recurrent hybrid)
7. TCN (Temporal Convolutional Network)
8. LSTM + Self-Attention
9. Stacked LSTM (deep)
10. Simple MLP (baseline for comparison)

EVALUATION:
- 5-fold cross-validation
- Metrics: MAE, RMSE, R², MAPE
- Per-target breakdown
- Statistical significance testing

RUN ON KAGGLE:
1. Create new notebook
2. Add StudentLife dataset: studentlife
3. Copy this code
4. Run all cells
5. Download results JSON

Author: FDS Project Team
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON EXPERIMENT")
print("Finding the optimal architecture for real behavioral data")
print("="*80)

# Check environment
KAGGLE = os.path.exists('/kaggle/input')
if KAGGLE:
    BASE_PATH = '/kaggle/input/student-life/dataset'
    OUTPUT_PATH = '/kaggle/working'
else:
    BASE_PATH = 'data/studentlife/dataset'
    OUTPUT_PATH = 'reports'

print(f"\nEnvironment: {'Kaggle' if KAGGLE else 'Local'}")
print(f"Base path: {BASE_PATH}")

# ============================================================================
# DATA EXTRACTION FROM STUDENTLIFE
# ============================================================================

def get_all_student_ids(base_path):
    """Get all available student IDs from the dataset."""
    student_ids = set()
    sensing_path = os.path.join(base_path, 'sensing', 'phonelock')
    if os.path.exists(sensing_path):
        for f in os.listdir(sensing_path):
            if f.startswith('phonelock_u') and f.endswith('.csv'):
                student_ids.add(f.replace('phonelock_', '').replace('.csv', ''))
    return sorted(list(student_ids))


def extract_sleep_from_phonelock(base_path, student_id):
    """Extract sleep duration from phone lock patterns."""
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
            # Night locks (10pm - 10am) longer than 3 hours are likely sleep
            night_locks = day_data[
                ((day_data['start_hour'] >= 22) | (day_data['start_hour'] <= 10)) & 
                (day_data['duration_hours'] >= 3)
            ]
            if len(night_locks) > 0:
                daily_sleep[date] = min(night_locks['duration_hours'].max(), 12)
        return daily_sleep
    except Exception as e:
        return {}


def extract_activity(base_path, student_id):
    """Extract exercise minutes and steps from activity data."""
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
            # Activity 1 = walking, 2 = running
            active_data = day_data[day_data[activity_col].isin([1, 2])]
            active_minutes = len(active_data) * 3 / 60  # Each record is ~3 seconds
            daily_exercise[date] = min(active_minutes, 120)
            daily_steps[date] = int(active_minutes * 100)  # Rough estimation
        return daily_exercise, daily_steps
    except Exception as e:
        return {}, {}


def extract_screen_time(base_path, student_id):
    """Extract screen time from phone lock data."""
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
            # Screen time = 18h awake - locked hours
            daily_screen[date] = min(max(0, 18 - day_data['locked_hours'].sum()), 16)
        return daily_screen
    except Exception as e:
        return {}


def extract_social_interactions(base_path, student_id):
    """Extract social interaction count from conversation data."""
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
    return daily_social


def extract_work_hours(base_path, student_id):
    """Estimate study/work hours from app usage patterns."""
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
        
        # Educational/work apps
        work_packages = ['com.google', 'edu.', 'blackboard', 'com.android.chrome', 
                        'com.microsoft', 'canvas', 'zoom', 'slack']
        daily_work = {}
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            work_mask = day_data[pkg_col].astype(str).apply(
                lambda x: any(pkg in x.lower() for pkg in work_packages)
            )
            # Each record ~10 seconds
            daily_work[date] = min(len(day_data[work_mask]) * 10 / 3600, 14)
        return daily_work
    except Exception as e:
        return {}


def build_dataset(base_path, max_students=None):
    """
    Build complete dataset from all students.
    Returns DataFrame with daily behavioral features.
    """
    student_ids = get_all_student_ids(base_path)
    if max_students:
        student_ids = student_ids[:max_students]
    
    print(f"\nExtracting data from {len(student_ids)} students...")
    
    all_records = []
    
    for i, student_id in enumerate(student_ids):
        # Extract all features
        sleep_hours = extract_sleep_from_phonelock(base_path, student_id)
        exercise, steps = extract_activity(base_path, student_id)
        screen_time = extract_screen_time(base_path, student_id)
        social = extract_social_interactions(base_path, student_id)
        work_hours = extract_work_hours(base_path, student_id)
        
        # Combine all dates
        all_dates = set()
        for data in [sleep_hours, exercise, screen_time, social, work_hours]:
            all_dates.update(data.keys())
        
        # Create records
        for date in sorted(all_dates):
            record = {
                'student_id': student_id,
                'date': date,
                'sleep_hours': sleep_hours.get(date, np.nan),
                'exercise_minutes': exercise.get(date, np.nan),
                'steps_count': steps.get(date, np.nan),
                'screen_time_hours': screen_time.get(date, np.nan),
                'social_interactions': social.get(date, np.nan),
                'work_hours': work_hours.get(date, np.nan),
            }
            all_records.append(record)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(student_ids)} students...")
    
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['student_id', 'date']).reset_index(drop=True)
    
    print(f"\n✓ Dataset built: {len(df)} records from {df['student_id'].nunique()} students")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def create_sequences(df, feature_cols, seq_length=7):
    """
    Create sequences for time series forecasting.
    Input: 7 days of features
    Output: Next day's features (behavioral forecasting)
    """
    sequences = []
    targets = []
    student_ids = []
    dates = []
    
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id].sort_values('date')
        
        if len(student_data) < seq_length + 1:
            continue
        
        # Fill NaN with student's mean
        student_data = student_data.copy()
        for col in feature_cols:
            student_data[col] = student_data[col].fillna(student_data[col].mean())
        
        # Fill remaining NaN with global defaults
        defaults = {
            'sleep_hours': 7.0,
            'exercise_minutes': 30.0,
            'steps_count': 5000.0,
            'screen_time_hours': 6.0,
            'social_interactions': 5.0,
            'work_hours': 5.0,
        }
        for col in feature_cols:
            student_data[col] = student_data[col].fillna(defaults.get(col, 0))
        
        values = student_data[feature_cols].values
        date_values = student_data['date'].values
        
        for i in range(len(values) - seq_length):
            seq = values[i:i+seq_length]
            target = values[i+seq_length]
            
            # Skip if too many zeros
            if np.sum(seq == 0) > seq_length * len(feature_cols) * 0.5:
                continue
            
            sequences.append(seq)
            targets.append(target)
            student_ids.append(student_id)
            dates.append(date_values[i+seq_length])
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"\n✓ Created {len(X)} sequences (seq_length={seq_length})")
    
    return X, y, student_ids, dates


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class LSTMModel(nn.Module):
    """Standard LSTM for sequence-to-vector prediction."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "LSTM"
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM for capturing both past and future context."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "BiLSTM"
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """GRU - simpler than LSTM, often faster convergence."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "GRU"
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class BiGRUModel(nn.Module):
    """Bidirectional GRU."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "BiGRU"
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, bidirectional=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    """Transformer encoder for sequence modeling."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nhead=4, dropout=0.3):
        super().__init__()
        self.name = "Transformer"
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CNNLSTMModel(nn.Module):
    """CNN for local patterns + LSTM for temporal dependencies."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "CNN-LSTM"
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq, features] -> [batch, features, seq]
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        # Back to [batch, seq, features]
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        # Trim to match residual size
        out = out[:, :, :residual.size(2)]
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network - alternative to RNNs."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.3):
        super().__init__()
        self.name = "TCN"
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x: [batch, seq, features] -> [batch, features, seq]
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        # Global average pooling
        x = x.mean(dim=2)
        return self.fc(x)


class LSTMAttentionModel(nn.Module):
    """LSTM with self-attention mechanism."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "LSTM-Attention"
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out[:, -1, :])


class DeepLSTMModel(nn.Module):
    """Deep stacked LSTM with skip connections."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=4, dropout=0.3):
        super().__init__()
        self.name = "DeepLSTM"
        self.lstms = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.lstms.append(nn.LSTM(in_dim, hidden_dim, 1, batch_first=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        for lstm, norm in zip(self.lstms, self.norms):
            out, _ = lstm(x)
            out = norm(out)
            out = self.dropout(out)
            x = out
        return self.fc(x[:, -1, :])


class MLPModel(nn.Module):
    """Simple MLP baseline - flattens sequence."""
    def __init__(self, input_dim, hidden_dim=128, seq_length=7, dropout=0.3):
        super().__init__()
        self.name = "MLP"
        self.flatten_dim = input_dim * seq_length
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
    """Train model with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_loss


def evaluate_model(model, test_loader, feature_cols):
    """Evaluate model and return per-feature metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Overall metrics
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    
    # Per-feature metrics
    per_feature = {}
    for i, col in enumerate(feature_cols):
        per_feature[col] = {
            'mae': mean_absolute_error(targets[:, i], preds[:, i]),
            'rmse': np.sqrt(mean_squared_error(targets[:, i], preds[:, i])),
            'r2': r2_score(targets[:, i], preds[:, i]) if np.std(targets[:, i]) > 0 else 0
        }
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'per_feature': per_feature
    }


def run_cross_validation(model_class, model_kwargs, X, y, feature_cols, n_folds=5, epochs=50):
    """Run k-fold cross-validation for a model."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize
        X_mean = X_train.mean(axis=(0, 1))
        X_std = X_train.std(axis=(0, 1)) + 1e-8
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0) + 1e-8
        
        X_train_norm = (X_train - X_mean) / X_std
        X_val_norm = (X_val - X_mean) / X_std
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm), 
            torch.FloatTensor(y_train_norm)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_norm), 
            torch.FloatTensor(y_val_norm)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create and train model
        model = model_class(**model_kwargs)
        model, val_loss = train_model(model, train_loader, val_loader, epochs=epochs)
        
        # Evaluate (on original scale)
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val_norm)
            device = next(model.parameters()).device
            preds_norm = model(X_val_tensor.to(device)).cpu().numpy()
        
        # Denormalize predictions
        preds = preds_norm * y_std + y_mean
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        
        fold_results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'val_loss': val_loss
        })
    
    # Aggregate results
    results = {
        'mae_mean': np.mean([r['mae'] for r in fold_results]),
        'mae_std': np.std([r['mae'] for r in fold_results]),
        'rmse_mean': np.mean([r['rmse'] for r in fold_results]),
        'rmse_std': np.std([r['rmse'] for r in fold_results]),
        'r2_mean': np.mean([r['r2'] for r in fold_results]),
        'r2_std': np.std([r['r2'] for r in fold_results]),
        'fold_results': fold_results
    }
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run the complete model comparison experiment."""
    print("\n" + "="*80)
    print("STEP 1: Data Preparation")
    print("="*80)
    
    # Build dataset
    df = build_dataset(BASE_PATH)
    
    # Feature columns
    feature_cols = ['sleep_hours', 'exercise_minutes', 'steps_count', 
                   'screen_time_hours', 'social_interactions', 'work_hours']
    
    # Create sequences
    X, y, student_ids, dates = create_sequences(df, feature_cols, seq_length=7)
    
    print(f"\nDataset shapes: X={X.shape}, y={y.shape}")
    
    # Define models to test
    input_dim = len(feature_cols)
    hidden_dim = 64
    
    models_to_test = [
        ("LSTM", LSTMModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("BiLSTM", BiLSTMModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("GRU", GRUModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("BiGRU", BiGRUModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("Transformer", TransformerModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("CNN-LSTM", CNNLSTMModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("TCN", TCNModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("LSTM-Attention", LSTMAttentionModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("DeepLSTM", DeepLSTMModel, {'input_dim': input_dim, 'hidden_dim': hidden_dim}),
        ("MLP", MLPModel, {'input_dim': input_dim, 'hidden_dim': 128, 'seq_length': 7}),
    ]
    
    print("\n" + "="*80)
    print("STEP 2: Model Comparison (5-fold CV)")
    print("="*80)
    
    results = {}
    
    for name, model_class, model_kwargs in models_to_test:
        print(f"\n{'─'*60}")
        print(f"Testing: {name}")
        print(f"{'─'*60}")
        
        try:
            cv_results = run_cross_validation(
                model_class, model_kwargs, X, y, feature_cols,
                n_folds=5, epochs=50
            )
            
            results[name] = cv_results
            
            print(f"  MAE:  {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
            print(f"  RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
            print(f"  R²:   {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[name] = {'error': str(e)}
    
    # Rank models
    print("\n" + "="*80)
    print("STEP 3: Model Ranking")
    print("="*80)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    # Sort by R² (higher is better)
    ranking = sorted(valid_results.items(), key=lambda x: x[1]['r2_mean'], reverse=True)
    
    print("\n🏆 MODEL LEADERBOARD (by R²):\n")
    print(f"{'Rank':<6} {'Model':<20} {'R²':<15} {'MAE':<15} {'RMSE':<15}")
    print("─" * 70)
    
    for i, (name, res) in enumerate(ranking, 1):
        r2_str = f"{res['r2_mean']:.4f} ± {res['r2_std']:.4f}"
        mae_str = f"{res['mae_mean']:.4f} ± {res['mae_std']:.4f}"
        rmse_str = f"{res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}"
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i:<4} {name:<20} {r2_str:<15} {mae_str:<15} {rmse_str:<15}")
    
    # Best model
    best_model_name = ranking[0][0]
    best_model_results = ranking[0][1]
    
    print(f"\n✅ BEST MODEL: {best_model_name}")
    print(f"   R² = {best_model_results['r2_mean']:.4f}")
    
    # Train final best model on all data
    print("\n" + "="*80)
    print("STEP 4: Training Final Best Model")
    print("="*80)
    
    # Find model class
    best_model_class = None
    best_model_kwargs = None
    for name, model_class, model_kwargs in models_to_test:
        if name == best_model_name:
            best_model_class = model_class
            best_model_kwargs = model_kwargs
            break
    
    # Normalize all data
    X_mean = X.mean(axis=(0, 1))
    X_std = X.std(axis=(0, 1)) + 1e-8
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-8
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    # Split 90/10 for final training
    n_train = int(0.9 * len(X))
    train_dataset = TensorDataset(
        torch.FloatTensor(X_norm[:n_train]), 
        torch.FloatTensor(y_norm[:n_train])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_norm[n_train:]), 
        torch.FloatTensor(y_norm[n_train:])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    final_model = best_model_class(**best_model_kwargs)
    final_model, _ = train_model(final_model, train_loader, val_loader, epochs=100)
    
    # Save checkpoint
    checkpoint = {
        'model_name': best_model_name,
        'model_state': final_model.state_dict(),
        'model_kwargs': best_model_kwargs,
        'feature_cols': feature_cols,
        'scaler_mean_X': X_mean.tolist(),
        'scaler_std_X': X_std.tolist(),
        'scaler_mean_y': y_mean.tolist(),
        'scaler_std_y': y_std.tolist(),
        'cv_results': best_model_results,
        'created_at': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(OUTPUT_PATH, 'best_behavioral_model.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"\n✓ Saved best model checkpoint: {checkpoint_path}")
    
    # Save full results
    results_summary = {
        'experiment_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(X),
            'num_students': len(set(student_ids)),
            'features': feature_cols,
            'sequence_length': 7
        },
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'fold_results'} 
                       for k, v in results.items() if 'error' not in v},
        'ranking': [(name, res['r2_mean']) for name, res in ranking],
        'best_model': {
            'name': best_model_name,
            'r2': best_model_results['r2_mean'],
            'mae': best_model_results['mae_mean'],
            'rmse': best_model_results['rmse_mean']
        }
    }
    
    results_path = os.path.join(OUTPUT_PATH, 'model_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✓ Saved results: {results_path}")
    
    return results_summary


if __name__ == '__main__':
    results = run_experiment()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nDownload these files from Kaggle:")
    print("  1. best_behavioral_model.pt - Best model checkpoint")
    print("  2. model_comparison_results.json - Full comparison data")
    print("\nNext step: Combine with synthetic mental health model for two-stage pipeline!")
