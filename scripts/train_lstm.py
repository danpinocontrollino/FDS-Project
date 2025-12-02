"""
================================================================================
TRAIN SEQUENCE MODELS (LSTM / GRU / CNN)
================================================================================
This script trains recurrent and convolutional neural networks to classify
burnout levels from 7-day behavioral time series windows.

ARCHITECTURE OPTIONS:
---------------------
1. LSTM (Long Short-Term Memory):
   - Learns long-term dependencies in sequential data
   - Has "gates" to control information flow (forget, input, output)
   - Best when: temporal patterns span multiple days

2. GRU (Gated Recurrent Unit):
   - Simplified LSTM with fewer parameters
   - Often performs similarly to LSTM but trains faster
   - Best when: dataset is smaller or faster training is needed

3. CNN (1D Convolutional Network):
   - Learns local patterns with sliding filters
   - Parallelizable (faster than RNNs on GPU)
   - Best when: local patterns (2-3 day trends) are important

WHY SEQUENCE MODELS?
-------------------
Burnout develops over time. A single day's data might not capture the trend.
By looking at 7 consecutive days, we can detect:
- Declining sleep quality over the week
- Increasing stress accumulation
- Work-life balance deterioration patterns

STRATIFIED USER SAMPLING:
------------------------
The full dataset has 1000 users × 365 days ≈ 1.47M sequences.
For CPU training, we support sampling a fraction of USERS (not sequences)
to preserve complete behavioral patterns per user.

Usage:
    python scripts/train_lstm.py --model lstm --window 7 --epochs 40
    python scripts/train_lstm.py --model gru --window 7 --epochs 40
    python scripts/train_lstm.py --model cnn --window 7 --epochs 40
    python scripts/train_lstm.py --model lstm --sample-users 0.1  # 10% of users

For Transformer models, use scripts/train_transformer.py instead.

Input:  data/processed/daily_with_burnout.parquet (from create_burnout_labels.py)
Output: models/saved/{model}_sequence.pt

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path("data/processed")
DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"
MODEL_DIR = Path("models/saved")

# Daily behavioral features used for sequence modeling
# These capture the key dimensions of work-life balance:
# Ordered by correlation with burnout_level (highest first)
FEATURE_COLS = [
    # Highest correlation features
    "stress_level",          # Self-reported stress (1-10) - corr: 0.345
    "commute_minutes",       # Daily commute time - corr: 0.340
    "exercise_minutes",      # Physical activity - corr: 0.222
    "work_hours",            # Hours worked that day - corr: 0.201
    "mood_score",            # Emotional state (1-10) - corr: 0.175
    "sleep_quality",         # Subjective sleep quality (1-10) - corr: 0.142
    "emails_received",       # Email volume (workload proxy) - corr: 0.123
    "caffeine_mg",           # Stimulant consumption - corr: 0.116
    "energy_level",          # Fatigue indicator (1-10) - corr: 0.112
    "sleep_hours",           # Total sleep duration - corr: 0.094
    "focus_score",           # Cognitive performance (1-10) - corr: 0.090
    
    # Secondary features (lower correlation but potentially useful)
    "meetings_count",        # Number of meetings (interruption proxy)
    "tasks_completed",       # Productivity metric
    "steps_count",           # Daily movement
    "alcohol_units",         # Depressant consumption
    "screen_time_hours",     # Digital exposure
    
    # Social & lifestyle (NEW - high value for burnout)
    "social_interactions",   # Social connection indicator
    "outdoor_time_minutes",  # Recovery/nature exposure
    "diet_quality",          # Nutrition quality score
    
    # Work environment (categorical -> numeric)
    "work_pressure",         # Converted: low=0, medium=1, high=2
]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model training configuration.
    
    Key arguments:
    - --model: Architecture type (lstm, gru, cnn)
    - --window: Sequence length in days (default 7 = weekly patterns)
    - --sample-users: Fraction of users for CPU optimization
    """
    parser = argparse.ArgumentParser(description="Train burnout sequence models")
    parser.add_argument("--model", choices=["lstm", "gru", "cnn", "lstm_attn"], default="lstm",
                        help="Model architecture: lstm, gru, cnn, or lstm_attn (with attention)")
    parser.add_argument("--window", type=int, default=7,
                        help="Sequence window size in days (default: 7)")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512, 
                        help="Mini-batch size (larger = faster on CPU)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--sample-users", type=float, default=0.1,
                        help="Fraction of USERS to include (default 0.1 = 10%% for CPU training)")
    parser.add_argument("--forecast-horizon", type=int, default=7,
                        help="Days ahead to predict burnout (default 7 = next week)")
    return parser.parse_args()


# ============================================================================
# DATA LOADING & SEQUENCE CREATION
# ============================================================================

def load_daily(window: int, sample_users: float = 1.0, forecast_horizon: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load daily data and create sliding window sequences for FORECASTING.
    
    CRITICAL CHANGE: We now predict FUTURE burnout, not same-period burnout.
    This prevents data leakage and creates a realistic early warning system.
    
    Process:
    1. Load daily records (1000 users × 365 days)
    2. Optional: Stratified user sampling for CPU optimization
    3. Create sliding windows: each 7-day window → 1 sequence
    4. Label = burnout level FORECAST_HORIZON days AFTER the window ends
    
    Args:
        window: Number of days in each sequence (default 7)
        sample_users: Fraction of users to include (for CPU training)
        forecast_horizon: Days into the future to predict (default 7 = next week)
        
    Returns:
        X: Sequences array (N × window × features)
        y: Labels array (N,)
        
    Example (window=7, forecast_horizon=7):
        For user with 365 days:
        - Sequence 1: days 1-7, label = day 14's burnout (1 week ahead)
        - Sequence 2: days 2-8, label = day 15's burnout
        - ...
        This forces the model to learn PREDICTIVE patterns, not correlations.
    """
    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/create_burnout_labels.py first to generate daily_with_burnout.parquet"
        )
    
    # Load and sort by user and date for proper sequence creation
    daily = pd.read_parquet(DAILY_PATH)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Convert categorical work_pressure to numeric
    # Original values: "low", "medium", "high"
    # Numeric encoding: 0, 1, 2 (ordinal - preserves order)
    if daily["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        daily["work_pressure"] = daily["work_pressure"].map(pressure_map).fillna(1).astype(np.float32)

    # ========== STRATIFIED USER SAMPLING ==========
    # For CPU training optimization: sample a fraction of users while
    # preserving the population's burnout class distribution.
    if sample_users < 1.0:
        # Step 1: Determine each user's "dominant" burnout level (mode)
        # This represents the user's typical state
        user_burnout = daily.groupby("user_id")["burnout_level"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()
        )
        user_burnout = user_burnout.reset_index()
        user_burnout.columns = ["user_id", "dominant_burnout"]
        
        n_users_total = len(user_burnout)
        n_users_sample = max(3, int(n_users_total * sample_users))  # at least 3 users
        
        # Step 2: Sample users proportionally from each burnout stratum
        # This ensures we don't accidentally sample only "Low burnout" users
        selected_users = []
        for level in sorted(user_burnout["dominant_burnout"].unique()):
            level_users = user_burnout[user_burnout["dominant_burnout"] == level]["user_id"].values
            # Proportion of this level in population
            level_proportion = len(level_users) / n_users_total
            n_from_level = max(1, int(n_users_sample * level_proportion))
            sampled = np.random.choice(level_users, min(n_from_level, len(level_users)), replace=False)
            selected_users.extend(sampled)
        
        daily = daily[daily["user_id"].isin(selected_users)]
        print(f"Stratified sampling: {len(selected_users):,} users ({sample_users*100:.0f}% of {n_users_total:,})")
        print(f"  Class distribution preserved: {daily['burnout_level'].value_counts(normalize=True).round(3).to_dict()}")

    # ========== SLIDING WINDOW SEQUENCE CREATION (FORECASTING) ==========
    sequences, labels = [], []
    
    for uid, group in daily.groupby("user_id"):
        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        labs = group["burnout_level"].to_numpy(dtype=np.int64)
        
        # Need enough data for window + forecast horizon
        if len(group) < window + forecast_horizon:
            continue
        
        # Create sliding windows with FUTURE labels
        for idx in range(window, len(group) - forecast_horizon + 1):
            seq = feats[idx - window: idx]       # 7-day feature window (days t-6 to t)
            label = labs[idx + forecast_horizon - 1]  # Burnout at day t+7 (1 week ahead)
            
            # Skip windows with missing values
            if np.isnan(seq).any():
                continue
                
            sequences.append(seq)
            labels.append(label)
    
    print(f"Forecasting mode: predicting burnout {forecast_horizon} days ahead")
    return np.stack(sequences), np.array(labels)


# ============================================================================
# DATALOADER CONSTRUCTION
# ============================================================================

def build_loaders(X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Uses random 80/20 split with shuffling (unlike MLP's temporal split).
    Shuffling is OK here because sequences are independent after creation.
    
    Args:
        X: Sequences array (N × window × features)
        y: Labels array (N,)
        batch_size: Mini-batch size
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        y_val: Validation labels for metric computation
    """
    # Random permutation for train/val split
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Wrap in TensorDataset
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    # Create DataLoaders
    # num_workers=0 avoids multiprocessing issues on some systems
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, y_val


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def build_model(model_type: str, input_dim: int, seq_len: int, num_classes: int) -> nn.Module:
    """
    Build the specified sequence model architecture.
    
    Args:
        model_type: "lstm", "gru", "cnn", "lstm_attn" (with attention)
        input_dim: Number of features per timestep
        seq_len: Sequence length (window size)
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    
    # ========== LSTM / GRU WITH ATTENTION ==========
    if model_type == "lstm_attn":
        
        class AttentionLayer(nn.Module):
            """Self-attention layer for sequence weighting."""
            def __init__(self, hidden_dim: int):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            
            def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
                # lstm_output: (batch, seq_len, hidden)
                attn_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
                attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize over sequence
                # Weighted sum of all hidden states
                context = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden)
                return context, attn_weights
        
        class LSTMWithAttention(nn.Module):
            """
            LSTM with Self-Attention for sequence classification.
            
            Instead of just using the last hidden state, attention learns
            which days in the sequence are most important for predicting burnout.
            """
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.2,
                    bidirectional=True,  # Look forward AND backward in time
                )
                self.attention = AttentionLayer(256)  # 128*2 for bidirectional
                self.head = nn.Sequential(
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)  # (batch, seq_len, 256)
                context, _ = self.attention(out)  # (batch, 256)
                return self.head(context)
        
        return LSTMWithAttention()
    
    # ========== LSTM / GRU (standard) ==========
    if model_type in {"lstm", "gru"}:
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        
        class SequenceNet(nn.Module):
            """
            Recurrent neural network for sequence classification.
            
            Architecture:
                Input (batch, seq_len, features)
                    -> 2-layer LSTM/GRU (128 hidden units)
                    -> Take last hidden state
                    -> LayerNorm -> ReLU -> Dropout
                    -> Linear(num_classes)
            """
            def __init__(self) -> None:
                super().__init__()
                # 2-layer RNN with dropout between layers
                self.rnn = rnn_cls(
                    input_size=input_dim,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,  # Input shape: (batch, seq, features)
                    dropout=0.2        # Dropout between RNN layers
                )
                # Classification head
                self.head = nn.Sequential(
                    nn.LayerNorm(128),    # Normalize RNN output
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, input_dim)
                out, _ = self.rnn(x)       # out: (batch, seq_len, 128)
                last_hidden = out[:, -1, :]  # Take last timestep: (batch, 128)
                return self.head(last_hidden)
        
        return SequenceNet()
    
    # ========== 1D CNN ==========
    if model_type == "cnn":
        
        class CNN1D(nn.Module):
            """
            1D Convolutional network for sequence classification.
            
            Architecture:
                Input (batch, seq_len, features) -> transpose -> (batch, features, seq_len)
                    -> Conv1D(64 filters, kernel=3)
                    -> ReLU -> BatchNorm
                    -> Conv1D(128 filters, kernel=3)
                    -> ReLU -> Global Average Pooling
                    -> Dropout -> Linear(num_classes)
            
            The CNN learns local temporal patterns (2-3 day windows)
            through the convolutional filters.
            """
            def __init__(self) -> None:
                super().__init__()
                # Convolutional layers
                # Kernel size 3 = looks at 3 consecutive days
                self.conv = nn.Sequential(
                    nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),  # Global average pooling
                )
                # Classification head
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, features)
                # Conv1d expects: (batch, channels, seq_len)
                x = x.transpose(1, 2)  # -> (batch, features, seq_len)
                feats = self.conv(x)   # -> (batch, 128, 1)
                return self.head(feats)
        
        return CNN1D()
    
    raise ValueError(
        f"Unknown model type: {model_type}. "
        "Use 'lstm', 'gru', 'cnn', or 'lstm_attn'. For transformer, use train_transformer.py"
    )


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Implements standard PyTorch training loop with:
    - CrossEntropy loss for multi-class classification
    - Adam optimizer
    - Model checkpointing (save best validation loss)
    """
    # Select device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load sequences with forecasting
    seq_X, seq_y = load_daily(args.window, args.sample_users, args.forecast_horizon)
    print(f"Total sequences: {len(seq_X):,}")
    
    # ========== FEATURE STANDARDIZATION ==========
    # Normalize features to zero mean and unit variance for better gradient flow
    # Reshape to (N*window, features), compute stats, then reshape back
    original_shape = seq_X.shape  # (N, window, features)
    seq_X_flat = seq_X.reshape(-1, seq_X.shape[2])  # (N*window, features)
    
    # Compute mean and std per feature
    mean = seq_X_flat.mean(axis=0)
    std = seq_X_flat.std(axis=0) + 1e-8  # Avoid division by zero
    
    # Standardize
    seq_X_flat = (seq_X_flat - mean) / std
    seq_X = seq_X_flat.reshape(original_shape)
    print(f"Features standardized (mean≈0, std≈1)")
    
    # Build data loaders
    train_loader, val_loader, y_val = build_loaders(seq_X, seq_y, args.batch_size)
    num_classes = len(np.unique(seq_y))
    
    # Build model
    model = build_model(args.model, seq_X.shape[2], seq_X.shape[1], num_classes).to(DEVICE)

    # Loss with label smoothing (reduces overconfidence, improves generalization)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Tracking
    best_val = float("inf")
    best_acc = 0.0
    history = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{args.model}_sequence.pt"

    # ========== TRAINING LOOP ==========
    for epoch in range(1, args.epochs + 1):
        
        # ----- Training phase -----
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()           # Clear gradients
            loss = criterion(model(xb), yb)  # Forward + loss
            loss.backward()                  # Backward
            optimizer.step()                 # Update weights
            
            train_losses.append(loss.item())
        
        # ----- Validation phase -----
        model.eval()
        val_losses = []
        preds = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        val_loss = float(np.mean(val_losses))
        history.append(val_loss)
        
        # Compute accuracy for this epoch
        epoch_acc = accuracy_score(y_val, preds)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Log progress every epoch (more visibility)
        print(f"Epoch {epoch:3d}: val_loss={val_loss:.4f}, val_acc={epoch_acc:.4f}")
        
        # Save best model (based on accuracy, not just loss)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "model_type": args.model,
                "feature_cols": FEATURE_COLS,
                "window": args.window,
                "best_accuracy": best_acc,
            }, model_path)

    # ========== FINAL METRICS ==========
    accuracy = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    cm = confusion_matrix(y_val, preds)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val:.4f}")
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Final F1 (macro): {f1*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"        Pred→  LOW    MED   HIGH")
    for i, row in enumerate(cm):
        label = ['LOW ', 'MED ', 'HIGH'][i]
        print(f"  {label}:  {row}")
    print(f"\nPer-class report:")
    print(classification_report(y_val, preds, target_names=['LOW', 'MEDIUM', 'HIGH']))
    
    # Update saved model with final metrics
    checkpoint = torch.load(model_path, weights_only=False)
    checkpoint['metrics'] = {
        'val_loss': best_val,
        'val_acc': accuracy,
        'val_f1': f1,
        'confusion_matrix': cm.tolist(),
    }
    torch.save(checkpoint, model_path)
    print(f"\nSaved model with metrics to {model_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main function - parse args and start training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
